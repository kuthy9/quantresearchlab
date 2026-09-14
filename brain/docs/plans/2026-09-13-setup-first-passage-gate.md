# Setup First-Passage Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on out-of-sample Group-5 Setup instances of 2022 H1, whether knowing the Setup lowers the log-loss of "target before failure boundary" beyond a geometry-only baseline, under the pre-registered family verdict of the spec.

**Architecture:** The existing Eye pass (`build_dataset`, Globex-week blocks) gains a path recorder that writes one row per new `PathSequenceStep` with the Setup's geometry, the tape, the nearest unswept levels and the Eye state. A label module scans the raw tape from each instance to its target, its failure boundary or a censor; a feature module builds M₀ (geometry) and M₁ (geometry + Setup); a gate script fits both on K0/K1/K2 × {zone_return, pool_reversal} with the existing purged folds, bootstrap, Holm and `family_verdict`, and writes results, verdict, descriptives and the receipt.

**Tech Stack:** Python 3.12, numpy, pandas, pyarrow, scikit-learn, lightgbm (`.venv` already has them), the existing `brain/research/gate_family.py`, `brain/scripts/predictability_gate.py` (`build_folds`, `session_labels`, `standardize_pair`) and `brain/scripts/build_gate_blocks.py`.

**Spec:** `brain/docs/specs/2026-09-13-setup-first-passage-gate-design.md` — read its §9 and §10 corrections: the target rule, the binary outcome and the pool-column sources in Tasks 2, 4 and 7 below were superseded before run 2.

## Global Constraints

- Work in the checkout `~/Desktop/quant/smc_trader` on branch `brain` (the user's instruction; the `.claude/worktrees/...` worktree is not used). Run every command from that directory with `.venv/bin/python`.
- Data source: `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet` only; sessions 2022-01-03 → 2022-06-06 with the 90 / 20 primary split; the block rule `globex_week_warmup_7d`. Nothing from `rolling_oof` or `sealed_holdout`.
- Unit: one path at one milestone; clocks K0 = {`zone_visible`, `pool_swept`}, K1 = {`reacceptance_held`}, K2 = {`micro_break_observed`}; context kinds `zone_return`, `pool_reversal`.
- Target = nearest unswept level in path direction over 5m ∪ 15m ∪ 1h; failure = `failure_boundary`; horizon `min(240 min, session end)`; same-bar → `failure`; unit ATR₆₀ = ATR₁ₘ·√60.
- M₀ = the nine geometry columns of `setup_features.GEOMETRY_COLUMNS`; M₁ = M₀ + `setup_matrix`; M₂ = M₁ + the 150 Eye-state components, reported only.
- Folds: primary 90 / 20, rolling 60 / 10 step 10, embargo 240 minutes. Verdict: `family_verdict` unchanged, Holm α = 0.10 per model class over judged cells; a cell is judged only with every class ≥ 5 % on primary training rows and ≥ 200 primary OOS rows.
- Nothing is selected on OOS rows. Standardisation, C and early stopping come from training rows only.
- Every new module has tests on synthetic data that run in seconds. The real-data run is Task 8 and is not a test.
- Outputs under `outputs/setup_gate/` (ignored). Commit per task with the attribution line `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`; never commit outputs; never `git add -A` (the checkout holds ignored `* 2.py` iCloud duplicates — leave them).
- Full test command after each task: `.venv/bin/python -m pytest brain/tests -q`.

---

### Task 1: Shared classifier fitting with a time-based purge

**Files:**
- Create: `brain/research/gate_models.py`
- Modify: `brain/scripts/information_gain_gate.py:67-157` (delete the moved definitions, import them)
- Test: `brain/tests/test_gate_models.py`

**Interfaces:**
- Produces:
  - `LOGISTIC_C_GRID = (0.01, 0.1, 1.0)`, `Z_CLIP = 10.0`, `DEFAULT_GAP_ROWS = 60`
  - `class GateModelError(RuntimeError)`
  - `purged_before(rows_before: np.ndarray, boundary: int, *, times: np.ndarray | None, embargo_minutes: int, gap_rows: int) -> np.ndarray` — the subset of `rows_before` (positions, all `< boundary`) whose label window ends before the boundary
  - `purged_after(rows_after: np.ndarray, last_test: int, *, times, embargo_minutes, gap_rows) -> np.ndarray`
  - `full_proba(estimator, x: np.ndarray) -> np.ndarray` (was `_full_proba`)
  - `select_logistic_c(train_x, train_y, *, times=None, embargo_minutes=240) -> float`
  - `fit_predict_proba(model: str, train_x, train_y, test_x, *, c=None, times=None, embargo_minutes=240) -> np.ndarray`
  - `minutes_of(index: pd.DatetimeIndex) -> np.ndarray` (int64 minutes since epoch)
- The old script keeps calling `select_logistic_c(train_x, y)` and `fit_predict_proba(model, ...)` without `times`, which reproduces the row-gap behaviour exactly (`gap_rows = 60`).

- [ ] **Step 1: Write the failing tests**

```python
# brain/tests/test_gate_models.py
"""Classifier fitting shared by the gates: a purge by clock when the rows are
not one per minute, the row gap otherwise."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.gate_models import (
    DEFAULT_GAP_ROWS,
    fit_predict_proba,
    minutes_of,
    purged_after,
    purged_before,
    select_logistic_c,
)


def test_row_gap_purge_matches_the_every_minute_gates() -> None:
    rows = np.arange(100)
    kept = purged_before(rows, 100, times=None, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert kept.tolist() == list(range(40))
    after = purged_after(np.arange(100, 200), 99, times=None, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert after.tolist() == list(range(160, 200))


def test_time_purge_drops_rows_whose_window_reaches_the_boundary() -> None:
    # instances ten minutes apart: a row is kept before the boundary only if
    # its 240-minute window ends by the boundary's clock, and after the test
    # rows only once their windows can no longer reach it.
    index = pd.date_range("2022-01-03T09:30", periods=100, freq="10min", tz="UTC")
    times = minutes_of(index)
    kept = purged_before(np.arange(50), 50, times=times, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert kept.tolist() == list(range(27))  # 10·r + 240 ≤ 500 → r ≤ 26
    after = purged_after(np.arange(50, 100), 49, times=times, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert after.tolist() == list(range(73, 100))  # 10·r ≥ 490 + 240 → r ≥ 73


def test_minutes_of_is_integer_minutes() -> None:
    index = pd.DatetimeIndex(["2022-01-03T09:30Z", "2022-01-03T09:31Z"])
    assert (np.diff(minutes_of(index)) == 1).all()
    assert minutes_of(index).dtype == np.int64


def test_fit_predict_proba_returns_three_columns_with_or_without_times() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 3))
    y = (x[:, 0] > 0).astype(int)  # two classes seen in training
    index = pd.date_range("2022-01-03T09:30", periods=600, freq="1min", tz="UTC")
    plain = fit_predict_proba("logistic", x, y, x[:10], c=1.0)
    timed = fit_predict_proba("logistic", x, y, x[:10], c=1.0, times=minutes_of(index))
    assert plain.shape == (10, 3) and timed.shape == (10, 3)
    assert np.allclose(plain.sum(axis=1), 1.0)
    assert select_logistic_c(x, y, times=minutes_of(index)) in (0.01, 0.1, 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_models.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain.research.gate_models'`

- [ ] **Step 3: Create the module**

```python
# brain/research/gate_models.py
"""Classifier fitting shared by the gates.

Penalty selection and early stopping are purged inside the training window
so the choice cannot see the holdout. The every-minute gates purge by a row
gap (sixty rows = sixty minutes); a gate whose rows are Setup instances
minutes or hours apart purges by clock instead: a row is kept for fitting
when its label window, ``embargo_minutes`` long, ends before the boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.first_passage import CLASS_COUNT
from brain.research.gate_family import log_loss_rows

LOGISTIC_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0)
Z_CLIP = 10.0
# The sixty-minute horizon of the every-minute gates, as rows.
DEFAULT_GAP_ROWS = 60


class GateModelError(RuntimeError):
    pass


def minutes_of(index: pd.DatetimeIndex) -> np.ndarray:
    return (pd.DatetimeIndex(index).asi8 // 60_000_000_000).astype(np.int64)


def purged_before(
    rows_before: np.ndarray, boundary: int, *, times: np.ndarray | None,
    embargo_minutes: int, gap_rows: int,
) -> np.ndarray:
    """Rows (all positioned before ``boundary``) whose label window ends
    before the boundary's clock, or a plain row gap when no clock is given."""

    rows_before = np.asarray(rows_before, dtype=int)
    if times is None:
        return rows_before[rows_before < boundary - gap_rows]
    return rows_before[times[rows_before] + embargo_minutes <= times[boundary]]


def purged_after(
    rows_after: np.ndarray, last_test: int, *, times: np.ndarray | None,
    embargo_minutes: int, gap_rows: int,
) -> np.ndarray:
    """Rows positioned after ``last_test`` far enough that the test rows'
    label windows cannot reach them."""

    rows_after = np.asarray(rows_after, dtype=int)
    if times is None:
        return rows_after[rows_after > last_test + gap_rows]
    return rows_after[times[rows_after] >= times[last_test] + embargo_minutes]


def full_proba(estimator, x: np.ndarray) -> np.ndarray:
    """Probabilities over all three classes even when training saw fewer."""

    proba = np.full((x.shape[0], CLASS_COUNT), 1e-6)
    partial = estimator.predict_proba(x)
    for position, label in enumerate(estimator.classes_):
        proba[:, int(label)] = partial[:, position]
    return proba / proba.sum(axis=1, keepdims=True)


def select_logistic_c(
    train_x: np.ndarray, train_y: np.ndarray, *, times: np.ndarray | None = None,
    embargo_minutes: int = 240,
) -> float:
    """Blocked, purged selection of the L2 penalty, the same protocol
    ``predictability_gate.select_ridge_alpha`` uses for ridge."""

    from sklearn.linear_model import LogisticRegression

    rows = train_x.shape[0]
    blocks = 3
    edges = np.linspace(0, rows, blocks + 1, dtype=int)
    scores: dict[float, list[float]] = {c: [] for c in LOGISTIC_C_GRID}
    for b in range(blocks):
        lo, hi = edges[b], edges[b + 1]
        test = np.arange(lo, hi)
        if test.size == 0:
            continue
        train = np.concatenate(
            [
                purged_before(np.arange(0, lo), lo, times=times, embargo_minutes=embargo_minutes, gap_rows=DEFAULT_GAP_ROWS),
                purged_after(np.arange(hi, rows), hi - 1, times=times, embargo_minutes=embargo_minutes, gap_rows=DEFAULT_GAP_ROWS),
            ]
        )
        if train.size < 200 or test.size < 100 or len(np.unique(train_y[train])) < 2:
            continue
        for c in LOGISTIC_C_GRID:
            fitted = LogisticRegression(C=c, max_iter=500).fit(train_x[train], train_y[train])
            scores[c].append(-float(log_loss_rows(full_proba(fitted, train_x[test]), train_y[test]).mean()))
    means = {c: float(np.mean(v)) for c, v in scores.items() if v}
    if not means:
        return LOGISTIC_C_GRID[0]
    return max(means, key=means.get)


def fit_predict_proba(
    model: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *,
    c: float | None = None, times: np.ndarray | None = None, embargo_minutes: int = 240,
) -> np.ndarray:
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression

        penalty = select_logistic_c(train_x, train_y, times=times, embargo_minutes=embargo_minutes) if c is None else c
        fitted = LogisticRegression(C=penalty, max_iter=500).fit(train_x, train_y)
        return full_proba(fitted, test_x)
    if model == "lightgbm":
        import lightgbm as lgb

        # Early-stop on a purged tail of the training window, as the
        # regression gate does: last fifth, a purge before it.
        cut = int(train_x.shape[0] * 0.8)
        fit = purged_before(np.arange(0, cut), cut, times=times, embargo_minutes=embargo_minutes, gap_rows=DEFAULT_GAP_ROWS)
        fit_x, fit_y = train_x[fit], train_y[fit]
        tail_x, tail_y = train_x[cut:], train_y[cut:]
        estimator = lgb.LGBMClassifier(
            objective="multiclass", num_class=CLASS_COUNT, n_estimators=400,
            learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=0,
            verbose=-1,
        )
        estimator.fit(
            fit_x, fit_y, eval_set=[(tail_x, tail_y)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return full_proba(estimator, test_x)
    raise GateModelError(f"unknown classifier {model}")


__all__ = [
    "DEFAULT_GAP_ROWS", "GateModelError", "LOGISTIC_C_GRID", "Z_CLIP",
    "fit_predict_proba", "full_proba", "minutes_of", "purged_after",
    "purged_before", "select_logistic_c",
]
```

Note on equivalence with the old script: the old `fit_x = train_x[: cut - HORIZON]` is `rows < cut - 60`, which `purged_before(..., times=None, gap_rows=60)` reproduces; the old block-CV train `[0, lo-60) ∪ [hi+60, rows)` is `purged_before(np.arange(0, lo), lo, ...)` ∪ `purged_after(np.arange(hi, rows), hi-1, ...)` (rows `> hi-1+60`, i.e. `>= hi+60`). Same rows.

- [ ] **Step 4: Point the old gate script at the module**

In `brain/scripts/information_gain_gate.py`: delete `LOGISTIC_C_GRID`, `Z_CLIP`, `_full_proba`, `select_logistic_c`, `fit_predict_proba` (lines 65–157 region) and add after the `gate_family` import:

```python
from brain.research.gate_models import (  # noqa: E402
    Z_CLIP,
    fit_predict_proba,
    select_logistic_c,
)
```

Keep `MINIMUM_CLASS_SHARE`, `GateRunError`, `continuous_targets` in the script. The `row_losses` body is unchanged (it calls `select_logistic_c(train_x, y[fold.train])` and `fit_predict_proba(model, train_x, y[fold.train], test_x, c=c)`).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_models.py brain/tests/test_information_gain_gate.py -q`
Expected: all PASS (the old gate's two tests still pass on the moved helpers).

- [ ] **Step 6: Commit**

```bash
git add brain/research/gate_models.py brain/scripts/information_gain_gate.py brain/tests/test_gate_models.py
git commit -m "refactor(brain): share the gates' classifier fitting, purged by clock when rows are not minutes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The path log

**Files:**
- Create: `brain/research/path_log.py`
- Test: `brain/tests/test_path_log.py`

**Interfaces:**
- Produces:
  - `PATH_COLUMNS: tuple[str, ...]` = `IDENTITY_COLUMNS + ZONE_COLUMNS + POOL_COLUMNS + TAPE_COLUMNS + LEVEL_COLUMNS + STRUCTURE_COLUMNS + FEATURE_NAMES`
  - `path_rows(observation, *, close: float, high: float, low: float, atr: float, history: Sequence[float], features: Sequence[float]) -> list[dict]`
  - `empty_path_log() -> pd.DataFrame`
  - `direction_sign(value) -> float` (+1 long, −1 short, 0 otherwise; accepts `Direction` or its string value)
  - `minutes_since_open(asof: pd.Timestamp) -> float`
  - `realized_volatility(closes: Sequence[float], minutes: int, atr: float) -> float`
- Consumed by Task 3 (`build_dataset`) and Task 4/5 (column names).

- [ ] **Step 1: Write the failing tests**

```python
# brain/tests/test_path_log.py
"""One row per new Group-5 step, with the geometry the Setup gate labels from,
read at the bar the step is published."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.path_log import (
    LEVEL_COLUMNS,
    PATH_COLUMNS,
    STRUCTURE_COLUMNS,
    direction_sign,
    empty_path_log,
    minutes_since_open,
    path_rows,
    realized_volatility,
)
from contract.market import Direction, Timeframe

AT = pd.Timestamp("2022-01-04T15:30", tz="UTC")


def _step(step_id: str, kind: str, reason: str, strength: float) -> SimpleNamespace:
    return SimpleNamespace(step_id=step_id, kind=kind, reason=reason, strength=strength, observed_at=AT)


def _path(sequence_id: str, context_kind: str, context_id: str, direction, steps) -> SimpleNamespace:
    return SimpleNamespace(
        sequence_id=sequence_id, context_kind=context_kind, context_id=context_id,
        direction=direction, formed_at=AT - pd.Timedelta(minutes=30), lifecycle="active", steps=tuple(steps),
    )


def _timeframe_state(bsl, ssl, ext=Direction.LONG):
    return SimpleNamespace(
        liquidity=SimpleNamespace(unswept_bsl=bsl, unswept_ssl=ssl),
        structure=SimpleNamespace(external_direction=ext, internal_direction=Direction.SHORT, last_bos_direction=None),
    )


def _observation(update, manipulations=()):
    snapshot = SimpleNamespace(
        asof=AT,
        timeframe_states={
            Timeframe.M5: _timeframe_state([101.0, 103.0], [98.0, 99.5]),
            Timeframe.M15: _timeframe_state([104.0], [97.0]),
            Timeframe.H1: _timeframe_state([], [90.0], ext=Direction.SHORT),
        },
    )
    return SimpleNamespace(market_snapshot=snapshot, asof=AT, interaction_update=update, manipulations=tuple(manipulations))


def _rows(update, manipulations=()):
    return path_rows(
        _observation(update, manipulations), close=100.0, high=100.5, low=99.5, atr=2.0,
        history=[100.0 + 0.1 * i for i in range(90)], features=tuple(float(i) for i in range(len(FEATURE_NAMES))),
    )


def test_zone_return_row_carries_zone_geometry_levels_structure_and_state() -> None:
    first = _step("s:0", "zone_visible", "typed_entry_zone_registered", 0.6)
    second = _step("s:1", "reacceptance_held", "held", 0.8)
    path = _path("s", "zone_return", "loc-1", Direction.LONG, [first, second])
    location = SimpleNamespace(
        location_id="loc-1", lower_bound=99.0, upper_bound=99.8, near_edge=99.8, far_edge=99.0,
        failure_boundary=98.7, source_zone_kind="fvg", entry_mode="touch", first_penetration_fraction=0.25,
        nearest_visible_draw_distance_points=3.0,
    )
    update = SimpleNamespace(
        milestone_transitions=(("s", second),), interaction_paths=(path,), interaction_path_transitions=(),
        zone_interactions=(location,), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert set(row) == set(PATH_COLUMNS)
    assert row["known_at"] == AT and row["sequence_id"] == "s" and row["context_kind"] == "zone_return"
    assert row["direction"] == 1.0 and row["context_found"] is True
    assert row["step_kind"] == "reacceptance_held" and row["step_ordinal"] == 1
    assert json.loads(row["steps_so_far"]) == [
        ["zone_visible", "typed_entry_zone_registered", 0.6], ["reacceptance_held", "held", 0.8]
    ]
    assert row["failure_boundary"] == 98.7 and row["source_zone_kind"] == "fvg" and row["eye_draw_distance_points"] == 3.0
    assert math.isnan(row["sweep_extreme"])
    assert row["close"] == 100.0 and row["atr_1m"] == 2.0
    assert (row["bsl_5m"], row["ssl_5m"], row["bsl_15m"], row["ssl_15m"]) == (101.0, 99.5, 104.0, 97.0)
    assert math.isnan(row["bsl_1h"]) and row["ssl_1h"] == 90.0
    assert (row["ext_dir_5m"], row["int_dir_5m"], row["last_bos_dir_5m"], row["ext_dir_1h"]) == (1.0, -1.0, 0.0, -1.0)
    assert row[FEATURE_NAMES[0]] == 0.0 and row[FEATURE_NAMES[-1]] == float(len(FEATURE_NAMES) - 1)


def test_pool_reversal_row_takes_failure_from_reacceptance_else_sweep_extreme() -> None:
    step = _step("p:0", "pool_swept", "typed_pool_manipulation_swept", 0.4)
    path = _path("p", "pool_reversal", "man-1", Direction.SHORT, [step])
    manipulation = SimpleNamespace(
        manipulation_id="man-1", source_lower_bound=104.0, source_upper_bound=104.5, sweep_extreme=105.2,
        penetration_atr=0.35, timeframe=Timeframe.M5,
    )
    update = SimpleNamespace(
        milestone_transitions=(("p", step),), interaction_paths=(), interaction_path_transitions=(path,),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update, [manipulation])
    assert row["direction"] == -1.0 and row["failure_boundary"] == 105.2 and row["source_timeframe"] == "5m"
    assert math.isnan(row["reference_price"]) and math.isnan(row["lower_bound"])
    reacceptance = SimpleNamespace(
        context_id="man-1", reference_price=104.5, failure_boundary=105.6, reclaim_margin_atr=0.2, hold_margin_atr=0.1,
    )
    update = SimpleNamespace(
        milestone_transitions=(("p", step),), interaction_paths=(path,), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(reacceptance,),
    )
    (row,) = _rows(update, [manipulation])
    assert row["failure_boundary"] == 105.6 and row["reference_price"] == 104.5 and row["hold_margin_atr"] == 0.1


def test_missing_context_is_logged_with_nan_geometry_and_flagged() -> None:
    step = _step("s:0", "zone_visible", "typed_entry_zone_registered", 0.6)
    path = _path("s", "zone_return", "loc-missing", Direction.LONG, [step])
    update = SimpleNamespace(
        milestone_transitions=(("s", step),), interaction_paths=(path,), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert row["context_found"] is False and math.isnan(row["failure_boundary"])
    # a step whose path is not on the update at all cannot be typed
    update = SimpleNamespace(
        milestone_transitions=(("ghost", step),), interaction_paths=(), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert row["context_kind"] is None and row["context_found"] is False


def test_no_update_or_no_transitions_yields_nothing() -> None:
    assert _rows(None) == []
    empty = SimpleNamespace(
        milestone_transitions=(), interaction_paths=(), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    assert _rows(empty) == []


def test_helpers() -> None:
    assert direction_sign(Direction.LONG) == 1.0 and direction_sign("short") == -1.0 and direction_sign(None) == 0.0
    # 15:30 UTC on 2022-01-04 is 10:30 New York, 16.5 h after the 18:00 open
    assert minutes_since_open(AT) == 990.0
    assert minutes_since_open(pd.Timestamp("2022-01-04T23:00", tz="UTC")) == 0.0  # 18:00 New York
    closes = [100.0, 101.0, 100.0, 102.0]
    assert realized_volatility(closes, 3, 2.0) == np.sqrt(1 + 1 + 4) / 2.0
    assert math.isnan(realized_volatility([100.0], 3, 2.0))
    assert list(empty_path_log().columns) == list(PATH_COLUMNS)
    assert len(LEVEL_COLUMNS) == 6 and len(STRUCTURE_COLUMNS) == 9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest brain/tests/test_path_log.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain.research.path_log'`

- [ ] **Step 3: Create the module**

```python
# brain/research/path_log.py
"""One row per new Group-5 path step, with the geometry the Setup gate labels from.

Rows are read from ``InteractionUpdate.milestone_transitions`` at the bar the
step is published, so ``known_at`` is the completed bar's ``asof`` and nothing
is known earlier than the Eye knew it. The Eye's own DTOs are read by
attribute and never rebuilt; a step whose context is not on the same
observation is logged with NaN geometry and ``context_found = False`` so the
receipt can count it (spec §5.2).
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from contract.market import Direction, Timeframe

EXCHANGE_TZ = "America/New_York"
SESSION_OPEN_HOUR = 18
LEVEL_SCALES: tuple[tuple[Timeframe, str], ...] = (
    (Timeframe.M5, "5m"), (Timeframe.M15, "15m"), (Timeframe.H1, "1h"),
)
IDENTITY_COLUMNS: tuple[str, ...] = (
    "known_at", "sequence_id", "context_kind", "context_id", "direction", "path_formed_at",
    "path_lifecycle", "step_id", "step_kind", "step_reason", "step_strength", "step_observed_at",
    "step_ordinal", "steps_so_far", "context_found",
)
ZONE_COLUMNS: tuple[str, ...] = (
    "lower_bound", "upper_bound", "near_edge", "far_edge", "failure_boundary", "source_zone_kind",
    "entry_mode", "first_penetration_fraction", "eye_draw_distance_points",
)
POOL_COLUMNS: tuple[str, ...] = (
    "source_lower_bound", "source_upper_bound", "sweep_extreme", "penetration_atr", "source_timeframe",
    "reference_price", "reclaim_margin_atr", "hold_margin_atr",
)
TAPE_COLUMNS: tuple[str, ...] = ("close", "high", "low", "atr_1m", "rv_30", "rv_60", "minutes_since_open")
LEVEL_COLUMNS: tuple[str, ...] = tuple(f"{side}_{name}" for _, name in LEVEL_SCALES for side in ("bsl", "ssl"))
STRUCTURE_COLUMNS: tuple[str, ...] = tuple(
    f"{field}_{name}" for _, name in LEVEL_SCALES for field in ("ext_dir", "int_dir", "last_bos_dir")
)
PATH_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ZONE_COLUMNS + POOL_COLUMNS + TAPE_COLUMNS + LEVEL_COLUMNS + STRUCTURE_COLUMNS + FEATURE_NAMES
)
_NAN = float("nan")


def direction_sign(value: object) -> float:
    text = value.value if isinstance(value, Direction) else value
    if text == Direction.LONG.value:
        return 1.0
    if text == Direction.SHORT.value:
        return -1.0
    return 0.0


def _text(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _number(obj: object, name: str) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def minutes_since_open(asof: pd.Timestamp) -> float:
    local = pd.Timestamp(asof).tz_convert(EXCHANGE_TZ)
    session_day = (local + pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)).normalize()
    open_at = session_day - pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)
    return float((local - open_at) / pd.Timedelta(minutes=1))


def realized_volatility(closes: Sequence[float], minutes: int, atr: float) -> float:
    """``sqrt(sum of squared one-minute close changes)`` over the last
    ``minutes`` changes, in ATR units; NaN when the history is too short."""

    values = np.asarray(closes[-(minutes + 1):], dtype=float)
    if values.size < minutes + 1 or not atr or not math.isfinite(atr):
        return _NAN
    return float(np.sqrt(np.sum(np.diff(values) ** 2)) / atr)


def _nearest_levels(snapshot: Any, close: float) -> dict[str, float]:
    out: dict[str, float] = {}
    states = getattr(snapshot, "timeframe_states", None) or {}
    for timeframe, name in LEVEL_SCALES:
        liquidity = getattr(states.get(timeframe), "liquidity", None)
        above = [float(level) for level in (getattr(liquidity, "unswept_bsl", None) or ()) if float(level) > close]
        below = [float(level) for level in (getattr(liquidity, "unswept_ssl", None) or ()) if float(level) < close]
        out[f"bsl_{name}"] = min(above) if above else _NAN
        out[f"ssl_{name}"] = max(below) if below else _NAN
    return out


def _structure(snapshot: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    states = getattr(snapshot, "timeframe_states", None) or {}
    for timeframe, name in LEVEL_SCALES:
        structure = getattr(states.get(timeframe), "structure", None)
        out[f"ext_dir_{name}"] = direction_sign(getattr(structure, "external_direction", None))
        out[f"int_dir_{name}"] = direction_sign(getattr(structure, "internal_direction", None))
        out[f"last_bos_dir_{name}"] = direction_sign(getattr(structure, "last_bos_direction", None))
    return out


def _zone(location: Any) -> dict[str, Any]:
    return {
        "lower_bound": _number(location, "lower_bound"),
        "upper_bound": _number(location, "upper_bound"),
        "near_edge": _number(location, "near_edge"),
        "far_edge": _number(location, "far_edge"),
        "failure_boundary": _number(location, "failure_boundary"),
        "source_zone_kind": _text(getattr(location, "source_zone_kind", None)),
        "entry_mode": _text(getattr(location, "entry_mode", None)),
        "first_penetration_fraction": _number(location, "first_penetration_fraction"),
        "eye_draw_distance_points": _number(location, "nearest_visible_draw_distance_points"),
    }


def _pool(manipulation: Any, reacceptance: Any) -> dict[str, Any]:
    failure = _number(reacceptance, "failure_boundary") if reacceptance is not None else _NAN
    if not math.isfinite(failure):
        failure = _number(manipulation, "sweep_extreme")
    return {
        "source_lower_bound": _number(manipulation, "source_lower_bound"),
        "source_upper_bound": _number(manipulation, "source_upper_bound"),
        "sweep_extreme": _number(manipulation, "sweep_extreme"),
        "penetration_atr": _number(manipulation, "penetration_atr"),
        "source_timeframe": _text(getattr(manipulation, "timeframe", None)),
        "reference_price": _number(reacceptance, "reference_price"),
        "reclaim_margin_atr": _number(reacceptance, "reclaim_margin_atr"),
        "hold_margin_atr": _number(reacceptance, "hold_margin_atr"),
        "failure_boundary": failure,
    }


def _blank() -> dict[str, Any]:
    row: dict[str, Any] = {name: _NAN for name in PATH_COLUMNS}
    for name in ("known_at", "sequence_id", "context_kind", "context_id", "path_formed_at", "path_lifecycle",
                 "step_id", "step_kind", "step_reason", "step_observed_at", "steps_so_far",
                 "source_zone_kind", "entry_mode", "source_timeframe"):
        row[name] = None
    row["context_found"] = False
    return row


def path_rows(
    observation: Any, *, close: float, high: float, low: float, atr: float,
    history: Sequence[float], features: Sequence[float],
) -> list[dict[str, Any]]:
    update = getattr(observation, "interaction_update", None)
    if update is None:
        return []
    transitions = tuple(getattr(update, "milestone_transitions", ()) or ())
    if not transitions:
        return []
    snapshot = getattr(observation, "market_snapshot", None)
    asof = pd.Timestamp(getattr(snapshot, "asof", None) or getattr(observation, "asof"))
    paths = {path.sequence_id: path for path in getattr(update, "interaction_paths", ())}
    paths.update({path.sequence_id: path for path in getattr(update, "interaction_path_transitions", ())})
    locations = {loc.location_id: loc for loc in getattr(update, "zone_interactions", ())}
    reacceptances = {item.context_id: item for item in getattr(update, "reacceptance_interactions", ())}
    manipulations = {item.manipulation_id: item for item in getattr(observation, "manipulations", ())}
    common: dict[str, Any] = {
        "known_at": asof, "close": float(close), "high": float(high), "low": float(low), "atr_1m": float(atr),
        "rv_30": realized_volatility(history, 30, atr), "rv_60": realized_volatility(history, 60, atr),
        "minutes_since_open": minutes_since_open(asof),
        **_nearest_levels(snapshot, float(close)), **_structure(snapshot),
        **dict(zip(FEATURE_NAMES, (float(v) for v in features))),
    }
    rows: list[dict[str, Any]] = []
    for sequence_id, step in transitions:
        row = _blank()
        row.update(common)
        row.update({
            "sequence_id": sequence_id, "step_id": getattr(step, "step_id", None),
            "step_kind": _text(getattr(step, "kind", None)), "step_reason": _text(getattr(step, "reason", None)),
            "step_strength": _number(step, "strength"),
            "step_observed_at": pd.Timestamp(step.observed_at) if getattr(step, "observed_at", None) is not None else None,
        })
        path = paths.get(sequence_id)
        if path is None:
            rows.append(row)
            continue
        steps = tuple(getattr(path, "steps", ()))
        ordinal = next((i for i, item in enumerate(steps) if item.step_id == row["step_id"]), len(steps) - 1)
        row.update({
            "context_kind": _text(path.context_kind), "context_id": path.context_id,
            "direction": direction_sign(path.direction),
            "path_formed_at": pd.Timestamp(path.formed_at) if getattr(path, "formed_at", None) is not None else None,
            "path_lifecycle": _text(getattr(path, "lifecycle", None)), "step_ordinal": int(max(ordinal, 0)),
            "steps_so_far": json.dumps(
                [[_text(item.kind), _text(item.reason), float(item.strength)] for item in steps[: ordinal + 1]]
            ),
        })
        if row["context_kind"] == "zone_return":
            location = locations.get(path.context_id)
            if location is not None:
                row.update(_zone(location))
                row["context_found"] = True
        elif row["context_kind"] == "pool_reversal":
            manipulation = manipulations.get(path.context_id)
            if manipulation is not None:
                row.update(_pool(manipulation, reacceptances.get(path.context_id)))
                row["context_found"] = True
        rows.append(row)
    return rows


def empty_path_log() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in PATH_COLUMNS})


__all__ = [
    "IDENTITY_COLUMNS", "LEVEL_COLUMNS", "LEVEL_SCALES", "PATH_COLUMNS", "POOL_COLUMNS",
    "STRUCTURE_COLUMNS", "TAPE_COLUMNS", "ZONE_COLUMNS", "direction_sign", "empty_path_log",
    "minutes_since_open", "path_rows", "realized_volatility",
]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_path_log.py -q`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add brain/research/path_log.py brain/tests/test_path_log.py
git commit -m "feat(brain): log every new Group-5 path step with the geometry a Setup is labelled from

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Record paths in the Eye pass and cache them per block

**Files:**
- Modify: `brain/research/trajectory_dataset.py:96-236` (`TrajectoryDataset`, `build_dataset`)
- Modify: `brain/research/event_log.py:56-113` (`save_block`, `load_blocks`)
- Test: `brain/tests/test_path_log.py` (append), `brain/tests/test_event_log.py` (append one test)

**Interfaces:**
- Consumes: `path_rows`, `empty_path_log`, `PATH_COLUMNS` (Task 2).
- Produces:
  - `TrajectoryDataset.paths: pd.DataFrame` (new field, default `empty_path_log()`)
  - `build_dataset(..., record_paths: bool = False)`
  - `save_block` writes `paths.parquet` beside `dataset.npz` (always; empty when nothing was recorded), cut at `emit_end`
  - `load_blocks(...)["paths"]: pd.DataFrame` sorted by `known_at, sequence_id, step_id`

- [ ] **Step 1: Write the failing tests**

Append to `brain/tests/test_event_log.py`:

```python
def test_blocks_carry_a_path_log_cut_at_emit_end(tmp_path) -> None:
    from brain.research.path_log import PATH_COLUMNS

    source = write_synthetic_ohlcv(session_bars(3), tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT, record_paths=True,
    )
    assert list(dataset.paths.columns) == list(PATH_COLUMNS)
    save_block(dataset, tmp_path / "block", emit_end="2025-01-08T12:00")
    assert (tmp_path / "block" / "paths.parquet").exists()
    data = load_blocks(tmp_path)
    assert list(data["paths"].columns) == list(PATH_COLUMNS)
    limit = pd.Timestamp("2025-01-08T12:00", tz="America/New_York")
    if len(data["paths"]):
        assert (data["paths"]["known_at"] < limit).all()
        assert data["paths"]["known_at"].dt.tz is not None


def test_a_block_without_a_path_log_still_loads(tmp_path) -> None:
    source = write_synthetic_ohlcv(session_bars(3), tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT,
    )
    save_block(dataset, tmp_path / "block", emit_end=None)
    (tmp_path / "block" / "paths.parquet").unlink()
    data = load_blocks(tmp_path)
    assert len(data["paths"]) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest brain/tests/test_event_log.py -q -k "path_log"`
Expected: FAIL with `TypeError: build_dataset() got an unexpected keyword argument 'record_paths'` and `AttributeError: ... 'paths'`

- [ ] **Step 3: Extend the dataset and the builder**

In `brain/research/trajectory_dataset.py`:

```python
from brain.research.path_log import PATH_COLUMNS, empty_path_log, path_rows
```

Add to `TrajectoryDataset` after `events`:

```python
    # One row per new Group-5 path step published inside the emit window;
    # the Setup gate's unit. Empty for every consumer that predates it.
    paths: pd.DataFrame = field(default_factory=empty_path_log)
```

Change `build_dataset`'s signature to add `record_paths: bool = False` after `progress_every`, add `path_row_list: list[dict] = []` beside `event_rows`, and replace the block from `snapshot = observation.market_snapshot` to the `feature_rows.append(...)` call with:

```python
        snapshot = observation.market_snapshot
        if snapshot is None or snapshot.asof < emit_from:
            continue
        if len(history) <= CONTEXT_LOOKBACK_MINUTES:
            continue
        state = snapshot.timeframe_states.get(Timeframe.M1)
        atr = getattr(getattr(state, "quality", None), "atr", None)
        if atr is None or float(atr) <= 0.0:
            continue
        atr = float(atr)
        features = observation_features(
            snapshot,
            closes=history[-(CONTEXT_LOOKBACK_MINUTES + 1):],
            bar_high_low=(float(bar.high), float(bar.low)),
        )
        if record_paths:
            # Paths are recorded on every warm emit-window bar: a step
            # published in the last hour before a break is still a Setup,
            # even though no sixty-minute future exists for the state row.
            path_row_list.extend(
                path_rows(
                    observation, close=float(bar.close), high=float(bar.high), low=float(bar.low),
                    atr=atr, history=history, features=features,
                )
            )
        index = position.get(snapshot.asof)
        if index is None or index + FUTURE_HORIZON_MINUTES >= len(frame):
            continue
        # A future is sixty consecutive traded minutes, not sixty rows: a
        # window that spans the maintenance break or a weekend would splice
        # the next session onto this one and label it as one path.
        if (
            frame.index[index + FUTURE_HORIZON_MINUTES] - frame.index[index]
            != pd.Timedelta(minutes=FUTURE_HORIZON_MINUTES)
        ):
            continue
        window = slice(index + 1, index + 1 + FUTURE_HORIZON_MINUTES)
        future_close_rows.append(closes[window].copy())
        future_high_rows.append(highs[window].copy())
        future_low_rows.append(lows[window].copy())
        feature_rows.append(features)
```

(The kept-row conditions are the same set as before; only their order changed so the feature vector exists before the path recorder runs.) In the final `return TrajectoryDataset(...)` add:

```python
        paths=(
            pd.DataFrame(path_row_list, columns=list(PATH_COLUMNS))
            if path_row_list
            else empty_path_log()
        ),
```

In `brain/research/event_log.py`, `save_block`, after the events write:

```python
    paths = getattr(dataset, "paths", None)
    paths = paths.copy() if paths is not None else empty_path_log()
    if len(paths):
        paths["known_at"] = pd.to_datetime(paths["known_at"], utc=True)
        if emit_end is not None:
            paths = paths[paths["known_at"] < pd.Timestamp(emit_end, tz="America/New_York")]
    paths.to_parquet(out_dir / "paths.parquet", index=False)
```

with `from brain.research.path_log import PATH_COLUMNS, empty_path_log` at the top (import inside the module is fine: `path_log` does not import `event_log`). In `load_blocks`, before `merged["blocks"] = ...`:

```python
    path_logs = [pd.read_parquet(path / "paths.parquet") for path in parts if (path / "paths.parquet").exists()]
    paths = pd.concat(path_logs, ignore_index=True) if path_logs else empty_path_log()
    if len(paths):
        paths["known_at"] = pd.to_datetime(paths["known_at"], utc=True)
        paths = paths.sort_values(["known_at", "sequence_id", "step_id"], kind="stable").reset_index(drop=True)
    merged["paths"] = paths[list(PATH_COLUMNS)] if len(paths) else paths
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_event_log.py brain/tests/test_path_log.py brain/tests/test_information_gain_gate.py -q`
Expected: all PASS (the synthetic tape may produce zero paths; the test tolerates that).

- [ ] **Step 5: Commit**

```bash
git add brain/research/trajectory_dataset.py brain/research/event_log.py brain/tests/test_event_log.py
git commit -m "feat(brain): record Group-5 path steps during the Eye pass and cache them per block

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Labels on the raw tape

**Files:**
- Create: `brain/research/setup_labels.py`
- Test: `brain/tests/test_setup_labels.py`

**Interfaces:**
- Consumes: path-log column names (Task 2): `known_at`, `context_kind`, `direction`, `close`, `atr_1m`, `failure_boundary`, `bsl_5m` … `ssl_1h`.
- Produces:
  - `HORIZON_MINUTES = 240`; `TARGET_FIRST, FAILURE_FIRST, CENSORED = 0, 1, 2`; `LABEL_NAMES = ("target", "failure", "censored")`
  - `LABEL_COLUMNS = ("label", "target_price", "d_target_points", "d_failure_points", "unit_atr60", "d_target_atr", "d_failure_atr", "time_to_resolve", "mae_atr", "mfe_atr", "minutes_to_session_end", "same_bar", "drop_reason")`
  - `DROP_REASONS = ("no_path", "no_atr", "no_failure", "no_target", "past_failure", "no_tape")` (a level on the wrong side of the close is simply not a target, so there is no `past_target`)
  - `session_end_positions(index: pd.DatetimeIndex) -> np.ndarray`
  - `nearest_target(row: Mapping, sign: float) -> float`
  - `label_instances(paths: pd.DataFrame, tape: pd.DataFrame, *, horizon_minutes: int = HORIZON_MINUTES) -> pd.DataFrame` — the input columns plus `LABEL_COLUMNS`; kept rows have `drop_reason == ""`.
- `tape` is `load_ohlcv(...).frame`: a tz-aware `DatetimeIndex` named `ts` (bar open) with `high` / `low` columns. The bar whose `ts == known_at` is the first bar after the completed one and is the first scanned.

- [ ] **Step 1: Write the failing tests**

```python
# brain/tests/test_setup_labels.py
"""Which of a Setup's two levels the tape reaches first, scanned from the bar
after the one that published the step, censored at the horizon or the
session's end."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.setup_labels import (
    CENSORED,
    FAILURE_FIRST,
    HORIZON_MINUTES,
    LABEL_COLUMNS,
    TARGET_FIRST,
    label_instances,
    nearest_target,
    session_end_positions,
)

START = pd.Timestamp("2022-01-04T15:00", tz="UTC")


def _tape(highs, lows, *, start=START, gap_after: int | None = None) -> pd.DataFrame:
    stamps = [start + pd.Timedelta(minutes=i) for i in range(len(highs))]
    if gap_after is not None:  # a maintenance break after position gap_after
        stamps = stamps[: gap_after + 1] + [t + pd.Timedelta(hours=1) for t in stamps[gap_after + 1 :]]
    return pd.DataFrame({"high": highs, "low": lows}, index=pd.DatetimeIndex(stamps, name="ts"))


def _path(**overrides) -> pd.DataFrame:
    row = {
        "known_at": START, "sequence_id": "s", "context_kind": "zone_return", "direction": 1.0,
        "close": 100.0, "atr_1m": 1.0, "failure_boundary": 98.0,
        "bsl_5m": 103.0, "ssl_5m": 97.0, "bsl_15m": 102.0, "ssl_15m": np.nan, "bsl_1h": np.nan, "ssl_1h": 90.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _label(paths, tape, **kw):
    out = label_instances(paths, tape, **kw)
    assert set(LABEL_COLUMNS) <= set(out.columns)
    return out.iloc[0]


def test_nearest_target_takes_the_closest_level_in_direction_across_scales() -> None:
    row = _path().iloc[0]
    assert nearest_target(row, 1.0) == 102.0
    assert nearest_target(row, -1.0) == 97.0
    assert np.isnan(nearest_target(_path(bsl_5m=np.nan, bsl_15m=np.nan).iloc[0], 1.0))


def test_target_first_with_time_and_excursions() -> None:
    tape = _tape([100.5, 101.0, 102.2, 100.0], [99.7, 99.0, 101.0, 99.5])
    row = _label(_path(), tape)
    assert row["label"] == TARGET_FIRST and row["drop_reason"] == ""
    assert row["target_price"] == 102.0 and row["d_target_points"] == 2.0 and row["d_failure_points"] == 2.0
    assert row["unit_atr60"] == np.sqrt(60) and np.isclose(row["d_target_atr"], 2.0 / np.sqrt(60))
    assert row["time_to_resolve"] == 3  # third scanned bar
    assert np.isclose(row["mae_atr"], (100.0 - 99.0) / np.sqrt(60))
    assert np.isclose(row["mfe_atr"], (102.2 - 100.0) / np.sqrt(60))
    assert row["same_bar"] == False  # noqa: E712


def test_failure_first_and_same_bar_is_failure() -> None:
    tape = _tape([100.5, 100.8], [99.7, 97.9])
    assert _label(_path(), tape)["label"] == FAILURE_FIRST
    tape = _tape([102.5], [97.5])
    row = _label(_path(), tape)
    assert row["label"] == FAILURE_FIRST and row["same_bar"] == True  # noqa: E712


def test_short_direction_is_mirrored() -> None:
    tape = _tape([100.3, 100.4, 100.2], [99.8, 99.5, 96.9])
    row = _label(_path(direction=-1.0, failure_boundary=101.0), tape)
    assert row["label"] == TARGET_FIRST and row["target_price"] == 97.0
    assert row["d_target_points"] == 3.0 and row["d_failure_points"] == 1.0


def test_censored_at_the_horizon_and_at_the_session_end() -> None:
    n = HORIZON_MINUTES + 20
    quiet = _tape([100.5] * n, [99.5] * n)
    row = _label(_path(), quiet)
    assert row["label"] == CENSORED and row["time_to_resolve"] == HORIZON_MINUTES
    assert row["minutes_to_session_end"] == n
    # the target is reached only after the horizon: still censored
    late = quiet.copy(); late.iloc[HORIZON_MINUTES, 0] = 105.0
    assert _label(_path(), late)["label"] == CENSORED
    # a break after ten bars ends the session: the hit on bar twelve is unseen
    broken = _tape([100.5] * 30, [99.5] * 30, gap_after=9)
    broken.iloc[12, 0] = 105.0
    row = _label(_path(), broken)
    assert row["label"] == CENSORED and row["time_to_resolve"] == 10 and row["minutes_to_session_end"] == 10


def test_session_end_positions_follow_gaps() -> None:
    tape = _tape([1.0] * 6, [1.0] * 6, gap_after=2)
    assert session_end_positions(tape.index).tolist() == [2, 2, 2, 5, 5, 5]


def test_drop_reasons() -> None:
    tape = _tape([100.5] * 5, [99.5] * 5)
    assert _label(_path(context_kind=None), tape)["drop_reason"] == "no_path"
    assert _label(_path(atr_1m=np.nan), tape)["drop_reason"] == "no_atr"
    assert _label(_path(failure_boundary=np.nan), tape)["drop_reason"] == "no_failure"
    assert _label(_path(bsl_5m=np.nan, bsl_15m=np.nan), tape)["drop_reason"] == "no_target"
    assert _label(_path(failure_boundary=100.5), tape)["drop_reason"] == "past_failure"
    assert _label(_path(bsl_5m=99.0, bsl_15m=99.0), tape)["drop_reason"] == "no_target"  # levels below close are not targets for a long
    assert _label(_path(known_at=START - pd.Timedelta(days=1)), tape)["drop_reason"] == "no_tape"
    dropped = label_instances(_path(atr_1m=np.nan), tape).iloc[0]
    assert np.isnan(dropped["label"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest brain/tests/test_setup_labels.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the module**

```python
# brain/research/setup_labels.py
"""Which of a Setup's two levels the tape reaches first.

The scan starts at the bar whose ``ts`` equals ``known_at``: the completed bar
that published the step closed at ``known_at``, so that is the first bar not
yet seen. It stops at ``min(horizon, session end)``, the session end being the
last bar before the next gap longer than a minute in the tape. Both levels on
one bar read as ``failure``, the conservative reading for the claim
(spec §5.3). Distances are also expressed in ATR₆₀ = ATR₁ₘ·√60.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from brain.research.first_passage import HORIZON_ATR_SCALE
from brain.research.path_log import LEVEL_SCALES

HORIZON_MINUTES = 240
TARGET_FIRST, FAILURE_FIRST, CENSORED = 0, 1, 2
LABEL_NAMES: tuple[str, ...] = ("target", "failure", "censored")
LABEL_COLUMNS: tuple[str, ...] = (
    "label", "target_price", "d_target_points", "d_failure_points", "unit_atr60", "d_target_atr",
    "d_failure_atr", "time_to_resolve", "mae_atr", "mfe_atr", "minutes_to_session_end", "same_bar", "drop_reason",
)
DROP_REASONS: tuple[str, ...] = ("no_path", "no_atr", "no_failure", "no_target", "past_failure", "no_tape")
_MINUTE_NS = 60_000_000_000


def session_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """For every row, the position of the last bar of its contiguous run."""

    stamps = pd.DatetimeIndex(index).asi8
    if stamps.size == 0:
        return np.zeros(0, dtype=int)
    gap_after = np.flatnonzero(np.diff(stamps) != _MINUTE_NS)
    ends = np.append(gap_after, stamps.size - 1)
    return ends[np.searchsorted(ends, np.arange(stamps.size))]


def nearest_target(row: Mapping[str, object], sign: float) -> float:
    close = float(row["close"])
    side = "bsl" if sign > 0 else "ssl"
    levels = []
    for _, name in LEVEL_SCALES:
        value = row.get(f"{side}_{name}")
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and sign * (value - close) > 0:
            levels.append(value)
    if not levels:
        return float("nan")
    return min(levels) if sign > 0 else max(levels)


def _blank(reason: str) -> dict[str, object]:
    out: dict[str, object] = {name: float("nan") for name in LABEL_COLUMNS}
    out["same_bar"] = False
    out["drop_reason"] = reason
    return out


def label_instances(
    paths: pd.DataFrame, tape: pd.DataFrame, *, horizon_minutes: int = HORIZON_MINUTES
) -> pd.DataFrame:
    index = pd.DatetimeIndex(tape.index)
    highs = tape["high"].to_numpy(dtype=float)
    lows = tape["low"].to_numpy(dtype=float)
    ends = session_end_positions(index)
    known = pd.to_datetime(paths["known_at"], utc=True)
    if index.tz is not None:
        known = known.dt.tz_convert(index.tz)
    positions = index.get_indexer(pd.DatetimeIndex(known))
    labelled: list[dict[str, object]] = []
    for row, start in zip(paths.to_dict("records"), positions):
        kind = row.get("context_kind")
        if kind is None or (isinstance(kind, float) and math.isnan(kind)):
            labelled.append(_blank("no_path")); continue
        atr = float(row.get("atr_1m", float("nan")))
        if not math.isfinite(atr) or atr <= 0.0:
            labelled.append(_blank("no_atr")); continue
        sign = float(row["direction"])
        failure = float(row.get("failure_boundary", float("nan")))
        if not math.isfinite(failure) or sign == 0.0:
            labelled.append(_blank("no_failure")); continue
        target = nearest_target(row, sign)
        if not math.isfinite(target):
            labelled.append(_blank("no_target")); continue
        close = float(row["close"])
        d_failure = sign * (close - failure)
        if d_failure <= 0.0:
            labelled.append(_blank("past_failure")); continue
        d_target = sign * (target - close)
        if start < 0:
            labelled.append(_blank("no_tape")); continue
        stop = min(start + horizon_minutes - 1, int(ends[start]))
        h = highs[start : stop + 1]
        l = lows[start : stop + 1]
        if sign > 0:
            hit_t, hit_f = h >= target, l <= failure
        else:
            hit_t, hit_f = l <= target, h >= failure
        first_t = int(hit_t.argmax()) if hit_t.any() else len(h)
        first_f = int(hit_f.argmax()) if hit_f.any() else len(h)
        if first_f <= first_t and first_f < len(h):
            label, k, same_bar = FAILURE_FIRST, first_f, bool(first_f == first_t)
        elif first_t < len(h):
            label, k, same_bar = TARGET_FIRST, first_t, False
        else:
            label, k, same_bar = CENSORED, len(h) - 1, False
        unit = atr * HORIZON_ATR_SCALE
        seen_h, seen_l = h[: k + 1], l[: k + 1]
        if sign > 0:
            mae, mfe = (close - seen_l.min()) / unit, (seen_h.max() - close) / unit
        else:
            mae, mfe = (seen_h.max() - close) / unit, (close - seen_l.min()) / unit
        labelled.append({
            "label": label, "target_price": target, "d_target_points": d_target, "d_failure_points": d_failure,
            "unit_atr60": unit / atr, "d_target_atr": d_target / unit, "d_failure_atr": d_failure / unit,
            "time_to_resolve": int(k + 1), "mae_atr": max(float(mae), 0.0), "mfe_atr": max(float(mfe), 0.0),
            "minutes_to_session_end": int(ends[start] - start + 1), "same_bar": same_bar, "drop_reason": "",
        })
    out = paths.reset_index(drop=True).copy()
    for name in LABEL_COLUMNS:
        out[name] = [item[name] for item in labelled]
    return out


__all__ = [
    "CENSORED", "DROP_REASONS", "FAILURE_FIRST", "HORIZON_MINUTES", "LABEL_COLUMNS", "LABEL_NAMES",
    "TARGET_FIRST", "label_instances", "nearest_target", "session_end_positions",
]
```

(`unit_atr60` is stored as the multiplier √60 so the test's `== np.sqrt(60)` holds with `atr_1m = 1`; the points-to-ATR₆₀ conversion uses `unit = atr · √60`.)

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_setup_labels.py -q`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add brain/research/setup_labels.py brain/tests/test_setup_labels.py
git commit -m "feat(brain): label each Setup by which of its two levels the tape reaches first

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Geometry and Setup feature matrices

**Files:**
- Create: `brain/research/setup_features.py`
- Test: `brain/tests/test_setup_features.py`

**Interfaces:**
- Consumes: label columns (Task 4), path-log columns (Task 2), `FEATURE_NAMES`.
- Produces:
  - `GEOMETRY_COLUMNS = ("log_ratio", "span_atr", "d_target_atr", "d_failure_atr", "minutes_to_session_end", "rv_30", "rv_60", "tod_sin", "tod_cos")`
  - `geometry_matrix(instances: pd.DataFrame) -> np.ndarray` (float64, `len × 9`)
  - `analytic_target_probability(instances) -> np.ndarray` = `d_f / (d_t + d_f)`
  - `setup_matrix(instances) -> tuple[np.ndarray, tuple[str, ...]]` — the M₁ increment: `context_kind=*`, `zone_kind=*`, `entry_mode=*`, `source_tf=*` one-hots over the sorted categories present, `step_strength:<kind>` for every `INTERACTION_PHYSICAL_PATH_STEP_KINDS` kind (max strength among reached steps of that kind, 0 when none), `reason=*` one-hots over reasons present in `steps_so_far`, `path_age`, `zone_width_atr`, `first_penetration_fraction`, `penetration_atr`, `reclaim_margin_atr`, `hold_margin_atr`, `align_<ext|int|bos>_<5m|15m|1h>` = `direction × dir`
  - `eye_state_matrix(instances) -> np.ndarray` (the 150 `FEATURE_NAMES` columns)
- NaN stays NaN in the matrices; `standardize_pair` imputes to 0 after z-scoring (spec §5.4).

- [ ] **Step 1: Write the failing tests**

```python
# brain/tests/test_setup_features.py
"""M₀ is geometry only; M₁ adds the Setup; the analytic driftless probability
is the zero-parameter reference."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.setup_features import (
    GEOMETRY_COLUMNS,
    analytic_target_probability,
    eye_state_matrix,
    geometry_matrix,
    setup_matrix,
)
from contract.eye.vocabulary import INTERACTION_PHYSICAL_PATH_STEP_KINDS


def _instances() -> pd.DataFrame:
    rows = []
    for i, (kind, zone, direction, ext) in enumerate(
        [("zone_return", "fvg", 1.0, 1.0), ("pool_reversal", None, -1.0, 1.0), ("zone_return", "ob", 1.0, -1.0)]
    ):
        row = {
            "known_at": pd.Timestamp("2022-01-04T15:00", tz="UTC") + pd.Timedelta(minutes=i),
            "path_formed_at": pd.Timestamp("2022-01-04T14:30", tz="UTC"),
            "context_kind": kind, "source_zone_kind": zone, "entry_mode": "touch" if zone else None,
            "source_timeframe": None if zone else "5m", "direction": direction,
            "d_target_atr": 2.0, "d_failure_atr": 1.0, "minutes_to_session_end": 300, "rv_30": 1.1, "rv_60": 1.5,
            "minutes_since_open": 690.0,
            "steps_so_far": json.dumps([["zone_visible", "typed_entry_zone_registered", 0.6], ["reacceptance_held", "held", 0.8]]),
            "lower_bound": 99.0, "upper_bound": 99.8, "atr_1m": 1.0, "first_penetration_fraction": 0.2,
            "penetration_atr": np.nan, "reclaim_margin_atr": np.nan, "hold_margin_atr": np.nan,
            "ext_dir_5m": ext, "int_dir_5m": 0.0, "last_bos_dir_5m": 1.0,
            "ext_dir_15m": 1.0, "int_dir_15m": 1.0, "last_bos_dir_15m": 1.0,
            "ext_dir_1h": -1.0, "int_dir_1h": -1.0, "last_bos_dir_1h": -1.0,
        }
        row.update({name: float(j) for j, name in enumerate(FEATURE_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_geometry_matrix_is_the_nine_registered_columns() -> None:
    x = geometry_matrix(_instances())
    assert x.shape == (3, len(GEOMETRY_COLUMNS)) and len(GEOMETRY_COLUMNS) == 9
    assert np.isclose(x[0, 0], np.log(2.0)) and x[0, 1] == 3.0 and x[0, 4] == 300
    # 690 minutes into a 1380-minute session is half a turn: sin ≈ 0, cos ≈ -1
    assert np.isclose(x[0, 7], 0.0, atol=1e-9) and np.isclose(x[0, 8], -1.0)


def test_analytic_probability_is_the_driftless_ratio() -> None:
    assert np.allclose(analytic_target_probability(_instances()), 1.0 / 3.0)


def test_setup_matrix_columns_and_values() -> None:
    x, names = setup_matrix(_instances())
    assert x.shape == (3, len(names))
    col = {name: i for i, name in enumerate(names)}
    assert x[0, col["context_kind=zone_return"]] == 1.0 and x[1, col["context_kind=pool_reversal"]] == 1.0
    assert x[0, col["zone_kind=fvg"]] == 1.0 and x[2, col["zone_kind=ob"]] == 1.0 and x[1, col["zone_kind=fvg"]] == 0.0
    assert x[1, col["source_tf=5m"]] == 1.0
    assert x[0, col["step_strength:reacceptance_held"]] == 0.8 and x[0, col["step_strength:micro_break_observed"]] == 0.0
    assert all(f"step_strength:{kind}" in col for kind in INTERACTION_PHYSICAL_PATH_STEP_KINDS)
    assert x[0, col["reason=held"]] == 1.0
    assert x[0, col["path_age"]] == 30.0 and x[1, col["path_age"]] == 31.0
    assert np.isclose(x[0, col["zone_width_atr"]], 0.8 / np.sqrt(60))
    assert np.isnan(x[1, col["zone_width_atr"]])  # no zone on a pool path: NaN, imputed later
    # alignment flips with direction: ext 5m is +1 on row 0 (long) and +1 on row 1 (short)
    assert x[0, col["align_ext_5m"]] == 1.0 and x[1, col["align_ext_5m"]] == -1.0 and x[2, col["align_ext_5m"]] == -1.0
    assert names == tuple(sorted(names, key=names.index))  # order is fixed by construction, not sorted later


def test_eye_state_matrix_is_the_150_components_in_order() -> None:
    x = eye_state_matrix(_instances())
    assert x.shape == (3, len(FEATURE_NAMES)) and x[0, 5] == 5.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest brain/tests/test_setup_features.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the module**

```python
# brain/research/setup_features.py
"""The two feature sets of the Setup gate and the zero-parameter reference.

M₀ is geometry only: under a driftless walk the target-first probability is
``d_f / (d_t + d_f)``, so a model that sees the two distances, the remaining
session and the volatility already holds everything a Setup-free forecast can
hold. M₁ adds what the Eye knows about the Setup. Columns are fixed here
(spec §5.4); nothing is chosen on the data.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.first_passage import HORIZON_ATR_SCALE
from contract.eye.vocabulary import INTERACTION_PHYSICAL_PATH_STEP_KINDS

SESSION_MINUTES = 1380
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "log_ratio", "span_atr", "d_target_atr", "d_failure_atr", "minutes_to_session_end",
    "rv_30", "rv_60", "tod_sin", "tod_cos",
)
STEP_KINDS: tuple[str, ...] = tuple(sorted(INTERACTION_PHYSICAL_PATH_STEP_KINDS))
CATEGORICALS: tuple[tuple[str, str], ...] = (
    ("context_kind", "context_kind"), ("source_zone_kind", "zone_kind"),
    ("entry_mode", "entry_mode"), ("source_timeframe", "source_tf"),
)
ALIGNMENT: tuple[tuple[str, str], ...] = (("ext_dir", "ext"), ("int_dir", "int"), ("last_bos_dir", "bos"))
SCALES: tuple[str, ...] = ("5m", "15m", "1h")


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def geometry_matrix(instances: pd.DataFrame) -> np.ndarray:
    d_t, d_f = _column(instances, "d_target_atr"), _column(instances, "d_failure_atr")
    phase = _column(instances, "minutes_since_open") / SESSION_MINUTES * 2.0 * np.pi
    columns = [
        np.log(d_t / d_f), d_t + d_f, d_t, d_f, _column(instances, "minutes_to_session_end"),
        _column(instances, "rv_30"), _column(instances, "rv_60"), np.sin(phase), np.cos(phase),
    ]
    return np.column_stack(columns).astype(float)


def analytic_target_probability(instances: pd.DataFrame) -> np.ndarray:
    d_t, d_f = _column(instances, "d_target_atr"), _column(instances, "d_failure_atr")
    return d_f / (d_t + d_f)


def _categories(frame: pd.DataFrame, name: str) -> tuple[str, ...]:
    if name not in frame:
        return ()
    values = {str(v) for v in frame[name].tolist() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return tuple(sorted(values))


def _steps(frame: pd.DataFrame) -> list[list[list]]:
    out = []
    for value in frame.get("steps_so_far", pd.Series([None] * len(frame))).tolist():
        try:
            out.append(json.loads(value) if isinstance(value, str) else [])
        except json.JSONDecodeError:
            out.append([])
    return out


def setup_matrix(instances: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    rows = len(instances)
    names: list[str] = []
    columns: list[np.ndarray] = []
    for source, prefix in CATEGORICALS:
        for category in _categories(instances, source):
            names.append(f"{prefix}={category}")
            columns.append((instances[source].astype(object).map(lambda v, c=category: 1.0 if str(v) == c else 0.0)).to_numpy(dtype=float))
    steps = _steps(instances)
    for kind in STEP_KINDS:
        names.append(f"step_strength:{kind}")
        columns.append(np.array([max((float(s[2]) for s in items if s[0] == kind), default=0.0) for items in steps]))
    reasons = sorted({str(s[1]) for items in steps for s in items if s[1] is not None})
    for reason in reasons:
        names.append(f"reason={reason}")
        columns.append(np.array([1.0 if any(str(s[1]) == reason for s in items) else 0.0 for items in steps]))
    known = pd.to_datetime(instances["known_at"], utc=True)
    formed = pd.to_datetime(instances["path_formed_at"], utc=True, errors="coerce")
    names.append("path_age")
    columns.append(((known - formed) / pd.Timedelta(minutes=1)).to_numpy(dtype=float))
    unit = _column(instances, "atr_1m") * HORIZON_ATR_SCALE
    names.append("zone_width_atr")
    columns.append((_column(instances, "upper_bound") - _column(instances, "lower_bound")) / unit)
    for name in ("first_penetration_fraction", "penetration_atr", "reclaim_margin_atr", "hold_margin_atr"):
        names.append(name)
        columns.append(_column(instances, name))
    sign = _column(instances, "direction")
    for field, short in ALIGNMENT:
        for scale in SCALES:
            names.append(f"align_{short}_{scale}")
            columns.append(sign * _column(instances, f"{field}_{scale}"))
    matrix = np.column_stack(columns).astype(float) if columns else np.zeros((rows, 0))
    return matrix, tuple(names)


def eye_state_matrix(instances: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_column(instances, name) for name in FEATURE_NAMES]).astype(float)


__all__ = [
    "GEOMETRY_COLUMNS", "STEP_KINDS", "analytic_target_probability", "eye_state_matrix",
    "geometry_matrix", "setup_matrix",
]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_setup_features.py -q`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add brain/research/setup_features.py brain/tests/test_setup_features.py
git commit -m "feat(brain): geometry-only and geometry-plus-Setup feature sets for the Setup gate

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Block builder records paths under its own run id

**Files:**
- Modify: `brain/scripts/build_gate_blocks.py:70-159`
- Test: `brain/tests/test_gate_blocks.py` (append)

**Interfaces:**
- Consumes: `build_dataset(..., record_paths=...)` (Task 3).
- Produces: `run_id(*, source, model, first_session, last_session, recorder: str | None = None) -> str`; CLI flags `--record-paths` (sets `recorder = "paths_v1"`) and `--output-root` (default unchanged `outputs/information_gain_gate`; the Setup run passes `outputs/setup_gate`). `_build_one` job tuple gains `record_paths: bool`; its cache check also requires `paths.parquet` when recording.

- [ ] **Step 1: Write the failing test**

Append to `brain/tests/test_gate_blocks.py`:

```python
def test_recording_paths_changes_the_run_id_and_nothing_else_does() -> None:
    from pathlib import Path

    from brain.scripts.build_gate_blocks import run_id

    root = Path(__file__).resolve().parents[2]
    source = root / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
    model = root / "configs/model.json"
    plain = run_id(source=source, model=model, first_session="2022-01-03", last_session="2022-06-06")
    again = run_id(source=source, model=model, first_session="2022-01-03", last_session="2022-06-06", recorder=None)
    paths = run_id(source=source, model=model, first_session="2022-01-03", last_session="2022-06-06", recorder="paths_v1")
    assert plain == again and plain != paths and len(paths) == 16
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_blocks.py -q`
Expected: FAIL with `TypeError: run_id() got an unexpected keyword argument 'recorder'`

- [ ] **Step 3: Implement**

In `run_id`, add the parameter `recorder: str | None = None` and, after `"data_splits_sha256"`, the line `if recorder is not None: digest_input["recorder"] = recorder` (so the plain digest is byte-identical to before). Change `_build_one`:

```python
def _build_one(job: tuple[Block, str, str, str, bool]) -> str:
    block, source, model, out_root, record_paths = job
    out = Path(out_root) / block.week
    cached = (out / "dataset.npz").exists() and (out / "events.parquet").exists()
    if record_paths:
        cached = cached and (out / "paths.parquet").exists()
    if cached:
        return f"{block.week}: cached"
    started = time.monotonic()
    dataset = build_dataset(
        source=Path(source), warmup_start=block.warmup_start, emit_start=block.emit_start,
        end=block.end, model_path=Path(model), root=ROOT, record_paths=record_paths,
    )
    save_block(dataset, out, emit_end=block.emit_end)
    return (
        f"{block.week}: {len(dataset.index)} clocks, {len(dataset.events)} events, "
        f"{len(dataset.paths)} path steps, {(time.monotonic() - started) / 60:.1f} min"
    )
```

In `main`: add `parser.add_argument("--record-paths", action="store_true")`; compute `recorder = "paths_v1" if args.record_paths else None`; pass `recorder=recorder` to `run_id`; write `"recorder": recorder` into `run.json`; build `jobs = [(block, str(source), str(model), str(out_root), args.record_paths) for block in blocks]`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_blocks.py -q`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add brain/scripts/build_gate_blocks.py brain/tests/test_gate_blocks.py
git commit -m "feat(brain): build gate blocks with the path recorder under a distinct run id

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The Setup gate script

**Files:**
- Create: `brain/scripts/setup_gate.py`
- Modify: `brain/scripts/information_gain_gate.py:159-185` (`folds_for` gains `embargo_minutes: int = HORIZON`)
- Test: `brain/tests/test_setup_gate.py`

**Interfaces:**
- Consumes: `label_instances`, `LABEL_COLUMNS`, `TARGET_FIRST/FAILURE_FIRST/CENSORED`, `HORIZON_MINUTES` (Task 4); `geometry_matrix`, `setup_matrix`, `eye_state_matrix`, `analytic_target_probability` (Task 5); `fit_predict_proba`, `select_logistic_c`, `minutes_of`, `Z_CLIP` (Task 1); `load_blocks(...)["paths"]` (Task 3); `family_verdict`, `log_loss_rows`, `session_block_bootstrap` (`gate_family`); `build_folds`, `session_labels`, `standardize_pair`, `Fold` (`predictability_gate`); `folds_for` (`information_gain_gate`, with the new `embargo_minutes`).
- Produces:
  - `CLOCKS = {"K0": ("zone_visible", "pool_swept"), "K1": ("reacceptance_held",), "K2": ("micro_break_observed",)}`, `CONTEXT_KINDS = ("zone_return", "pool_reversal")`, `PRIMARY = (90, 20)`, `ROLLING = (60, 10, 10)`, `MODELS = ("logistic", "lightgbm")`, `MINIMUM_CLASS_SHARE = 0.05`, `MINIMUM_OOS_ROWS = 200`, `TARGET_NAME = "target_first"`
  - `clock_instances(instances: pd.DataFrame, step_kinds: tuple[str, ...]) -> pd.DataFrame` — kept rows with `step_kind` in `step_kinds`, first per `sequence_id` by `known_at`, sorted by `known_at`
  - `run_setup_gate(instances, *, out_dir, clocks=CLOCKS, context_kinds=CONTEXT_KINDS, primary=PRIMARY, rolling=ROLLING, models=MODELS, minimum_oos_rows=MINIMUM_OOS_ROWS, log=print) -> pd.DataFrame` (the verdict; also writes `results.csv`, `verdict.csv`, `descriptives.csv`)
  - `main()`: `--run-id`, `--output-root outputs/setup_gate`, `--models`; loads blocks, the tape, labels the paths, writes `instances.parquet`, runs the gate, tees `gate.log`.
- Result rows: `clock` = `"K1:zone_return"`, `target` = `"target_first"`, `model` ∈ models, plus `m2_loss`, `m0_accuracy`, `m1_accuracy` and `analytic_loss` columns; verdict rows add `judged`, `oos_rows`, `min_class_share`, `refusal`.

- [ ] **Step 1: Write the failing tests**

```python
# brain/tests/test_setup_gate.py
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
            "time_to_resolve_p50_target", "mae_p50_failure"} <= set(descriptives.columns)


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
    assert set(verdict.loc[verdict["clock"].str.endswith("pool_reversal"), "refusal"]) == {"class_share"}
    assert set(verdict.loc[verdict["clock"].str.endswith("zone_return"), "refusal"]) == {"oos_rows"}
    assert not verdict["PASS"].any()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest brain/tests/test_setup_gate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain.scripts.setup_gate'`

- [ ] **Step 3: Generalise `folds_for`**

In `brain/scripts/information_gain_gate.py`, change the signature to
`def folds_for(index, mask, *, primary, rolling, embargo_minutes: int = HORIZON)` and pass `embargo_minutes=embargo_minutes` to both `build_folds` calls instead of `HORIZON`. (The old gate keeps its 60.)

- [ ] **Step 4: Create the script**

```python
# brain/scripts/setup_gate.py
"""The Setup first-passage gate: does a Group-5 Setup change where price goes first?

Pre-registered in brain/docs/specs/2026-09-13-setup-first-passage-gate-design.md.
M₀ = geometry (``setup_features.GEOMETRY_COLUMNS``); M₁ = M₀ + the Setup
(``setup_matrix``); M₂ = M₁ + the Eye state, reported only. Cells are
{K0, K1, K2} × {zone_return, pool_reversal}; a cell is judged only with every
class ≥ 5 % on its primary training rows and ≥ 200 primary OOS rows.

Usage::

    python -m brain.scripts.setup_gate --run-id <id>

where ``<id>`` is the directory ``build_gate_blocks.py --record-paths`` wrote
under ``outputs/setup_gate/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.research.event_log import load_blocks  # noqa: E402
from brain.research.first_passage import CLASS_COUNT  # noqa: E402
from brain.research.gate_family import family_verdict, log_loss_rows, session_block_bootstrap  # noqa: E402
from brain.research.gate_models import Z_CLIP, fit_predict_proba, minutes_of, select_logistic_c  # noqa: E402
from brain.research.setup_features import (  # noqa: E402
    analytic_target_probability,
    eye_state_matrix,
    geometry_matrix,
    setup_matrix,
)
from brain.research.setup_labels import (  # noqa: E402
    CENSORED,
    FAILURE_FIRST,
    HORIZON_MINUTES,
    LABEL_NAMES,
    TARGET_FIRST,
    label_instances,
)
from brain.scripts.information_gain_gate import folds_for  # noqa: E402
from brain.scripts.predictability_gate import Fold, session_labels, standardize_pair  # noqa: E402

CLOCKS: dict[str, tuple[str, ...]] = {
    "K0": ("zone_visible", "pool_swept"),
    "K1": ("reacceptance_held",),
    "K2": ("micro_break_observed",),
}
CONTEXT_KINDS: tuple[str, ...] = ("zone_return", "pool_reversal")
TARGET_NAME = "target_first"
PRIMARY: tuple[int, int] = (90, 20)
ROLLING: tuple[int, int, int] = (60, 10, 10)
MODELS: tuple[str, ...] = ("logistic", "lightgbm")
MINIMUM_CLASS_SHARE = 0.05
MINIMUM_OOS_ROWS = 200
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


class SetupGateError(RuntimeError):
    pass


def clock_instances(instances: pd.DataFrame, step_kinds: tuple[str, ...]) -> pd.DataFrame:
    kept = instances[(instances["drop_reason"] == "") & instances["step_kind"].isin(step_kinds)]
    kept = kept.assign(known_at=pd.to_datetime(kept["known_at"], utc=True))
    kept = kept.sort_values(["known_at", "sequence_id"], kind="stable")
    return kept.drop_duplicates("sequence_id", keep="first").reset_index(drop=True)


def _analytic_loss(instances: pd.DataFrame, rows: np.ndarray) -> float:
    """Two-class log-loss of the driftless ratio on resolved rows."""

    labels = instances["label"].to_numpy()[rows]
    resolved = labels != CENSORED
    if not resolved.any():
        return float("nan")
    p = np.clip(analytic_target_probability(instances.iloc[rows])[resolved], 1e-6, 1 - 1e-6)
    y = labels[resolved] == TARGET_FIRST
    return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))


def _descriptive(cell: str, sub: pd.DataFrame, fold: Fold | None, dropped: pd.Series) -> dict:
    row: dict = {"clock": cell, "rows": int(len(sub))}
    labels = sub["label"].to_numpy()
    for value, name in zip((TARGET_FIRST, FAILURE_FIRST, CENSORED), LABEL_NAMES):
        row[f"share_{name}"] = float(np.mean(labels == value)) if len(sub) else float("nan")
    row["share_same_bar"] = float(sub["same_bar"].astype(bool).mean()) if len(sub) else float("nan")
    for reason, count in dropped.items():
        row[f"dropped_{reason}"] = int(count)
    row["analytic_loss"] = _analytic_loss(sub, fold.holdout) if fold is not None else float("nan")
    for value, name in zip((TARGET_FIRST, FAILURE_FIRST, CENSORED), LABEL_NAMES):
        part = sub[labels == value]
        for column in ("time_to_resolve", "mae_atr"):
            for q in QUANTILES:
                row[f"{column}_p{int(q * 100)}_{name}"] = float(part[column].quantile(q)) if len(part) else float("nan")
    return row


def run_setup_gate(
    instances: pd.DataFrame,
    *,
    out_dir: Path,
    clocks: dict[str, tuple[str, ...]] = CLOCKS,
    context_kinds: tuple[str, ...] = CONTEXT_KINDS,
    primary: tuple[int, int] = PRIMARY,
    rolling: tuple[int, int, int] = ROLLING,
    models: tuple[str, ...] = MODELS,
    minimum_oos_rows: int = MINIMUM_OOS_ROWS,
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    descriptives: list[dict] = []
    refused: list[dict] = []
    judged_cells: dict[str, dict] = {}

    for clock, step_kinds in clocks.items():
        for kind in context_kinds:
            cell = f"{clock}:{kind}"
            raw = instances[(instances["context_kind"] == kind) & instances["step_kind"].isin(step_kinds)]
            dropped = raw.loc[raw["drop_reason"] != "", "drop_reason"].value_counts()
            sub = clock_instances(raw, step_kinds)
            if len(sub) == 0:
                log(f"{cell}: no instances")
                descriptives.append(_descriptive(cell, sub, None, dropped))
                refused.append({"clock": cell, "refusal": "no_rows", "oos_rows": 0, "min_class_share": float("nan")})
                continue
            index = pd.DatetimeIndex(sub["known_at"])
            y = sub["label"].to_numpy(dtype=int)
            sessions = session_labels(index)
            try:
                folds = folds_for(index, np.ones(len(sub), dtype=bool), primary=primary, rolling=rolling,
                                  embargo_minutes=HORIZON_MINUTES)
            except Exception as error:  # too few sessions for one fold
                log(f"{cell}: {error}")
                descriptives.append(_descriptive(cell, sub, None, dropped))
                refused.append({"clock": cell, "refusal": "folds", "oos_rows": 0, "min_class_share": float("nan")})
                continue
            first, _ = folds[0]
            shares = np.bincount(y[first.train], minlength=CLASS_COUNT) / max(first.train.size, 1)
            oos_rows = int(first.holdout.size)
            descriptives.append(_descriptive(cell, sub, first, dropped))
            log(f"{cell}: {len(sub)} instances, primary train {first.train.size} / OOS {oos_rows}, "
                f"class shares (target/failure/censored) {np.round(shares, 3).tolist()}")
            refusal = None
            if shares.min() < MINIMUM_CLASS_SHARE:
                refusal = "class_share"
            elif oos_rows < minimum_oos_rows:
                refusal = "oos_rows"
            if refusal is not None:
                log(f"{cell}: refused ({refusal}); reported, not judged")
                refused.append({"clock": cell, "refusal": refusal, "oos_rows": oos_rows, "min_class_share": float(shares.min())})
                continue
            judged_cells[cell] = {"oos_rows": oos_rows, "min_class_share": float(shares.min())}

            base = geometry_matrix(sub)
            increment, _ = setup_matrix(sub)
            full = np.hstack([base, increment])
            wide = np.hstack([full, eye_state_matrix(sub)])
            times = minutes_of(index)
            penalties: dict[str, float] = {}

            def losses(model: str, fold: Fold, x: np.ndarray, *, key: str) -> tuple[np.ndarray, float]:
                train_x, test_x = standardize_pair(x[fold.train], x[fold.holdout])
                np.clip(train_x, -Z_CLIP, Z_CLIP, out=train_x)
                np.clip(test_x, -Z_CLIP, Z_CLIP, out=test_x)
                c = None
                if model == "logistic":
                    c = penalties.get(key)
                    if c is None:
                        c = select_logistic_c(train_x, y[fold.train], times=times[fold.train], embargo_minutes=HORIZON_MINUTES)
                        penalties[key] = c
                proba = fit_predict_proba(model, train_x, y[fold.train], test_x, c=c,
                                          times=times[fold.train], embargo_minutes=HORIZON_MINUTES)
                accuracy = float(np.mean(proba.argmax(axis=1) == y[fold.holdout]))
                return log_loss_rows(proba, y[fold.holdout]), accuracy

            for fold, is_primary in folds:
                for model in models:
                    tick = time.monotonic()
                    m0, m0_acc = losses(model, fold, base, key=f"{model}:m0")
                    m1, m1_acc = losses(model, fold, full, key=f"{model}:m1")
                    m2, _ = losses(model, fold, wide, key=f"{model}:m2")
                    diff = m1 - m0
                    hold_sessions = sessions[fold.holdout]
                    mean, low, high, p = session_block_bootstrap(diff, hold_sessions)
                    results.append({
                        "clock": cell, "target": TARGET_NAME, "model": model, "fold": fold.index,
                        "primary": is_primary, "holdout_from": fold.holdout_sessions[0],
                        "holdout_to": fold.holdout_sessions[1], "rows": int(diff.size),
                        "m0_loss": float(m0.mean()), "m1_loss": float(m1.mean()), "m2_loss": float(m2.mean()),
                        "m0_accuracy": m0_acc, "m1_accuracy": m1_acc,
                        "analytic_loss": _analytic_loss(sub, fold.holdout),
                        "delta_logloss": float(mean), "ci_low": low, "ci_high": high, "p_raw": p,
                    })
                    log(f"  {cell} fold {fold.index} {model:9s} M0 {m0.mean():.4f} M1 {m1.mean():.4f} "
                        f"M2 {m2.mean():.4f} delta {mean:+.4f} [{low:+.4f}, {high:+.4f}] ({time.monotonic() - tick:.0f}s)")
                    if is_primary:
                        pooled.setdefault((cell, TARGET_NAME, model), []).append((diff, hold_sessions))

    frame = pd.DataFrame(results)
    frame.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(descriptives).to_csv(out_dir / "descriptives.csv", index=False)

    verdicts = []
    for model in models:
        cells = frame[(frame["model"] == model) & frame["clock"].isin(judged_cells)] if len(frame) else frame
        if len(cells) == 0:
            continue
        family = {key: value for key, value in pooled.items() if key[2] == model}
        verdicts.append(family_verdict(cells, family, alpha=0.10))
    verdict = pd.concat(verdicts, ignore_index=True) if verdicts else pd.DataFrame(
        columns=["clock", "target", "model", "primary_delta", "primary_ci_low", "primary_ci_high", "p_raw",
                 "folds", "folds_beating", "consistent", "robust", "p_holm_reject", "PASS"]
    )
    verdict["judged"] = True
    verdict["refusal"] = ""
    verdict["oos_rows"] = [judged_cells[c]["oos_rows"] for c in verdict["clock"]]
    verdict["min_class_share"] = [judged_cells[c]["min_class_share"] for c in verdict["clock"]]
    for item in refused:
        for model in models:
            verdict.loc[len(verdict)] = {
                "clock": item["clock"], "target": TARGET_NAME, "model": model, "primary_delta": float("nan"),
                "primary_ci_low": float("nan"), "primary_ci_high": float("nan"), "p_raw": float("nan"), "folds": 0,
                "folds_beating": float("nan"), "consistent": False, "robust": False, "p_holm_reject": False,
                "PASS": False, "judged": False, "refusal": item["refusal"], "oos_rows": item["oos_rows"],
                "min_class_share": item["min_class_share"],
            }
    for column in ("consistent", "robust", "p_holm_reject", "PASS", "judged"):
        verdict[column] = verdict[column].astype(bool)
    verdict = verdict.sort_values(["model", "clock"], kind="stable").reset_index(drop=True)
    verdict.to_csv(out_dir / "verdict.csv", index=False)
    log(verdict.to_string(index=False))
    log(f"VERDICT: {'PASS' if bool(verdict['PASS'].any()) else 'FAIL'}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/setup_gate")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    run_root = ROOT / args.output_root / args.run_id
    log_path = run_root / "gate.log"
    handle = log_path.open("a", encoding="utf-8")

    def log(line: str) -> None:
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    data = load_blocks(run_root / "blocks")
    meta = json.loads((run_root / "run.json").read_text())
    from shares.core.io import load_ohlcv

    tape = load_ohlcv(meta["source"], start=meta["blocks"][0]["warmup_start"], end=meta["blocks"][-1]["end"]).frame
    paths = data["paths"]
    log(f"{len(paths)} path steps over {len(data['blocks'])} blocks; "
        f"context found on {float(paths['context_found'].astype(bool).mean()):.1%}")
    instances = label_instances(paths, tape)
    instances.to_parquet(run_root / "instances.parquet", index=False)
    kept = instances["drop_reason"] == ""
    log(f"{int(kept.sum())} labelled instances of {len(instances)}; drops: "
        f"{instances.loc[~kept, 'drop_reason'].value_counts().to_dict()}")
    run_setup_gate(instances, out_dir=run_root, models=tuple(args.models.split(",")), log=log)
    handle.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest brain/tests/test_setup_gate.py brain/tests/test_information_gain_gate.py -q`
Expected: all PASS. If `test_planted_setup_effect_passes_and_writes_every_table` fails on `judged["PASS"].any()`, raise `effect` to 3.5 in that test only (the planted effect must be large enough for ≥ 80 % of the rolling folds); do not touch the gate.

- [ ] **Step 6: Commit**

```bash
git add brain/scripts/setup_gate.py brain/scripts/information_gain_gate.py brain/tests/test_setup_gate.py
git commit -m "feat(brain): the Setup first-passage gate over K0/K1/K2 x context kind with a geometry baseline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Run on the real tape and write the receipt

**Files:**
- Create: `brain/docs/evidence/2026-09-1X_setup_gate.md` (date of the run)
- Outputs (ignored): `outputs/setup_gate/<run_id>/`

- [ ] **Step 1: Full suite green before any run**

Run: `.venv/bin/python -m pytest brain/tests -q`
Expected: all PASS (the eleven pre-existing broken modules live outside `brain/tests`).

- [ ] **Step 2: Smoke one short block end to end (about twenty minutes)**

Write to the scratchpad (not the repo) `smoke_paths.py`:

```python
import sys; sys.path.insert(0, "~/Desktop/quant/smc_trader")
from pathlib import Path
import pandas as pd
from brain.research.trajectory_dataset import build_dataset
from brain.research.setup_labels import label_instances
from shares.core.io import load_ohlcv
ROOT = Path("~/Desktop/quant/smc_trader")
SRC = ROOT / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
ds = build_dataset(source=SRC, warmup_start="2021-12-27T18:00", emit_start="2022-01-03T18:00",
                   end="2022-01-05T20:00", model_path=ROOT / "configs/model.json", root=ROOT, record_paths=True)
p = ds.paths
print(len(p), "steps;", p["sequence_id"].nunique(), "paths;", p["context_kind"].value_counts().to_dict())
print("context_found", p["context_found"].mean(), "| step kinds", p["step_kind"].value_counts().head(8).to_dict())
tape = load_ohlcv(SRC, start="2021-12-27T18:00", end="2022-01-05T20:00").frame
inst = label_instances(p, tape)
print(inst["drop_reason"].value_counts().to_dict())
kept = inst[inst["drop_reason"] == ""]
print(kept.groupby(["context_kind", "step_kind"])["label"].value_counts().unstack(fill_value=0))
print(kept[["d_target_atr", "d_failure_atr", "time_to_resolve", "mae_atr"]].describe().round(3))
```

Run: `.venv/bin/python <scratchpad>/smoke_paths.py`
Check by hand: `context_found` near 1.0; both context kinds present; `reacceptance_held` and `micro_break_observed` appear; drop reasons are dominated by `no_target` or `past_failure` only if that is explainable (record the shares); distances positive; no label class absent. Fix the recorder or labeler before the full run if any of this is off; every fix gets a test.

- [ ] **Step 3: Build the 23 blocks in the background (about four hours)**

Run:
```bash
nohup .venv/bin/python brain/scripts/build_gate_blocks.py --record-paths --output-root outputs/setup_gate --workers 6 > outputs/setup_gate/blocks.log 2>&1 &
```
Poll `outputs/setup_gate/blocks.log` every 30–40 minutes; the run id is its first line. A block failing is a bug to fix and rebuild (cached blocks are skipped on rerun).

- [ ] **Step 4: Run the gate**

Run: `.venv/bin/python -m brain.scripts.setup_gate --run-id <id>` (tee'd to `gate.log` by the script). Expect under an hour: instances number in the low thousands per clock.

- [ ] **Step 5: Write the receipt**

`brain/docs/evidence/<date>_setup_gate.md`, in the shape of `2026-09-12_information_gain_gate.md`: verdict line; the run table (run id, source, atomic identity, blocks, wall time, path steps, `context_found` share, instances per cell, drop shares by reason, same-bar share, class shares); the verdict table per model (Δ log-loss M₁−M₀ with intervals, folds beating, judged / refusal); the reported-not-judged tables (M₂ − M₁, analytic loss vs M₀, time-to-resolve and MAE quantiles by label and cell); deviations from the spec, if any, each disclosed; "what the result says, and does not say"; the sha256 of `results.csv`, `verdict.csv`, `descriptives.csv`, `instances.parquet` (`shasum -a 256`); the commit list.

- [ ] **Step 6: Commit the receipt**

```bash
git add brain/docs/evidence/<date>_setup_gate.md
git commit -m "docs(brain): Setup first-passage gate receipt — <PASS|FAIL>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Close the record

**Files:**
- Modify: `brain/docs/evidence/2026-09-12_information_gain_gate.md` (append the run-2 section)
- Modify: `brain/docs/README.md` ("What actually predicts what" — one paragraph on the Setup gate pointing at its receipt; the scripts table gains `build_gate_blocks.py --record-paths` and `setup_gate.py`; the research table gains `path_log.py`, `setup_labels.py`, `setup_features.py`, `gate_models.py`)

- [ ] **Step 1: Append run 2 to the information-gain receipt**

Under a new heading `## Run 2 (ATR₆₀ barriers, 2026-09-12 14:00): FAIL on all eighteen cells`, the primary-fold table read from `outputs/information_gain_gate/aa4be1c91244d0c4/verdict.csv` (the eighteen `primary_delta` / CI / `folds_beating` values), the M₀ log-losses against the class-prior entropies (fp_1.0_1.0: M₀ 0.815–0.830 vs prior 0.864; fp_1.0_0.5: 0.952–0.991 vs 0.998; fp_0.5_1.0: 0.998–1.023 vs 1.021), and one sentence: the 4–5 % M₀ gains on the symmetric target are magnitude skill (reaching any barrier), not side. Keep the run-1 text and mark it void as the spec correction already does.

- [ ] **Step 2: README paragraph and tables**

Under "What actually predicts what", after the Eye-state subsection, add a subsection `#### The Setup gate: Group-5 paths against their own geometry` with: one paragraph stating the question (does knowing the Setup change P(target before failure) beyond the barrier geometry, the remaining session and the volatility), the unit (one Group-5 path at K0/K1/K2), the baseline (M₀ = geometry, analytic driftless reference `d_f/(d_t+d_f)` reported), and the verdict with the judged cells' Δ log-loss and intervals copied from the receipt, ending with a link to `evidence/<date>_setup_gate.md`. In the "Research — `brain/research/`" table add rows for `gate_models.py` (classifier fitting shared by the gates, purged by clock), `path_log.py` (one row per new Group-5 step with its geometry), `setup_labels.py` (target-before-failure labels on the raw tape) and `setup_features.py` (geometry and Setup feature sets). In the "Scripts" table add `build_gate_blocks.py --record-paths` and `setup_gate.py`.

- [ ] **Step 3: Full suite, then commit**

Run: `.venv/bin/python -m pytest brain/tests -q`
Expected: all PASS

```bash
git add brain/docs/evidence/2026-09-12_information_gain_gate.md brain/docs/README.md
git commit -m "docs(brain): record run 2 of the information-gain gate and point the README at the Setup gate

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Cleanup and self-review**

Delete the scratchpad smoke script; `git status --short` must show nothing but ignored `* 2.py` files; `git log --oneline main..brain` lists the plan's commits; nothing under `outputs/` is tracked. Then report per the CLAUDE.md rules: files changed, commands run with results, the verdict, remaining risks, next steps.
