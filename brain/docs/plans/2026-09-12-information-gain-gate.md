# Information-Gain Gate (Part 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on out-of-sample event clocks of 2022, whether the Eye's event sequence Δₜ improves sixty-minute first-passage forecasts over the raw tape plus the Eye's state Sₜ, under the pre-registered family verdict of the spec.

**Architecture:** One Eye pass in warmed Globex-week blocks writes, per block, the existing state dataset plus a transition-event log. A new research module encodes Δₜ (recency/count/direction per kind × scale, one K=4 ordered track per scale, trigger one-hot) causally from the log; a new script fits M₀ = RAW+S and M₁ = RAW+S+Δ on three clocks with the existing purged folds, scores multiclass log-loss with a session-block bootstrap, Holm-corrects nine tests and writes the verdict.

**Tech Stack:** Python 3.12, numpy, pandas, pyarrow, scikit-learn ≥ 1.3, lightgbm ≥ 4.3 (`uv sync --all-extras`), the existing `brain/scripts/predictability_gate.py` helpers.

**Spec:** `brain/docs/specs/2026-09-11-information-gain-gate-design.md` §5–§7.

## Global Constraints

- Data source: `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet` only. Sessions: warm-up from 2021-12-27; training 2022-01-03 → 2022-05-09 (90 sessions); OOS 2022-05-10 → 2022-06-06 (20 sessions). Nothing from `rolling_oof` (≥ 2024-02-01) or `sealed_holdout` (≥ 2026-04-01) is read.
- Blocks: Globex weeks, each with the seven calendar days before the week's open as unsampled warm-up.
- Transition event = `EventKind` value not ending in `_state`, not `bar_completed`, not `market_epoch_reset` (49 kinds).
- Horizon 60 minutes; ATR = the snapshot's one-minute ATR (`prices[:, 3]`).
- Δₜ: L = 120 minutes, K = 4 per scale; fixed, never tuned.
- Verdict family: {`fp_1.0_1.0`, `fp_1.0_0.5`, `fp_0.5_1.0`} × {C1, C5, C15}; Holm at α = 0.10; PASS needs the primary-fold interval, ≥ 80 % rolling folds, and drop-best robustness for one cell and one model class.
- Nothing is selected on the OOS sessions. Standardisation and hyper-parameters come from training rows only.
- Every new module has tests on synthetic sessions (`shares/tests/helpers.session_bars`) that run in seconds; the real-data run is Task 7 and is not a test.
- All outputs under `outputs/information_gain_gate/` (ignored). Commit per task; do not commit outputs.

---

### Task 1: Event log beside the state dataset

**Files:**
- Create: `brain/research/event_log.py`
- Modify: `brain/research/trajectory_dataset.py:96-215` (`TrajectoryDataset`, `build_dataset`)
- Test: `brain/tests/test_event_log.py`

**Interfaces:**
- Produces:
  - `TRANSITION_KINDS: tuple[str, ...]`, `is_transition(kind: EventKind) -> bool`
  - `EVENT_COLUMNS = ("known_at", "kind", "timeframe", "direction", "side", "strength", "price", "entity_id", "lifecycle", "event_id")`
  - `event_row(event: MarketEvent) -> dict[str, object]`
  - `empty_event_log() -> pd.DataFrame`
  - `TrajectoryDataset.events: pd.DataFrame` (new field, default `empty_event_log()`)
  - `save_block(dataset: TrajectoryDataset, out_dir: Path, *, emit_end: str | None) -> None` writing `dataset.npz` (keys `index, features, prices, future_closes, future_highs, future_lows`) and `events.parquet`
  - `load_blocks(blocks_root: Path) -> dict` returning the `load_dataset` dict with an extra `"events"` DataFrame, blocks concatenated in directory-name order.

- [ ] **Step 1: Write the failing test**

```python
# brain/tests/test_event_log.py
"""The event log records every transition event, and nothing else, at the
clock it was published."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain.research.event_log import (
    EVENT_COLUMNS,
    TRANSITION_KINDS,
    is_transition,
    load_blocks,
    save_block,
)
from brain.research.trajectory_dataset import build_dataset
from contract.eye import EventKind

ROOT = Path(__file__).resolve().parents[2]


def test_transition_kinds_exclude_state_republication_and_heartbeats() -> None:
    assert len(TRANSITION_KINDS) == 49
    assert "bar_completed" not in TRANSITION_KINDS
    assert "market_epoch_reset" not in TRANSITION_KINDS
    assert not any(kind.endswith("_state") for kind in TRANSITION_KINDS)
    assert is_transition(EventKind.SWEEP_CONFIRMED)
    assert not is_transition(EventKind.BAR_COMPLETED)


@pytest.fixture(scope="module")
def synthetic_block(tmp_path_factory) -> Path:
    from shares.tests.helpers import session_bars, write_synthetic_ohlcv

    source = tmp_path_factory.mktemp("tape") / "synthetic.parquet"
    write_synthetic_ohlcv(session_bars(3), source)
    dataset = build_dataset(
        source=source,
        warmup_start="2025-01-05",
        emit_start="2025-01-07",
        end="2025-01-09",
        model_path=ROOT / "configs" / "model.json",
        root=ROOT,
    )
    out = tmp_path_factory.mktemp("blocks") / "2025-01-06"
    save_block(dataset, out, emit_end=None)
    return out.parent


def test_the_log_has_one_row_per_transition_event_at_its_known_at(synthetic_block) -> None:
    data = load_blocks(synthetic_block)
    events = data["events"]
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert len(events) > 0
    assert set(events["kind"]).issubset(set(TRANSITION_KINDS))
    assert events["known_at"].is_monotonic_increasing
    # every event clock is a completed-bar clock of the emit window
    clocks = pd.DatetimeIndex(data["index"]).tz_localize("UTC")
    assert events["known_at"].dt.tz_convert("UTC").isin(clocks).all()


def test_blocks_concatenate_in_order(synthetic_block) -> None:
    data = load_blocks(synthetic_block)
    assert data["features"].shape[0] == data["index"].shape[0] == data["prices"].shape[0]
    assert np.all(np.diff(data["index"].astype("int64")) > 0)
```

`shares/tests/helpers.py` must provide `write_synthetic_ohlcv(bars, path)` writing a parquet with columns `open, high, low, close, volume, symbol, instrument_id` indexed by `ts` in the shape `load_ohlcv` reads (see `shares/core/io.py:268-340` for the accepted columns). If it does not exist, add it to `shares/tests/helpers.py`:

```python
def write_synthetic_ohlcv(bars: list[Bar], path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "open": [b.open for b in bars], "high": [b.high for b in bars],
            "low": [b.low for b in bars], "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "symbol": [b.symbol for b in bars],
            "instrument_id": [int(b.instrument_id) for b in bars],
        },
        index=pd.DatetimeIndex([b.start for b in bars], name="ts"),
    )
    frame.to_parquet(path)
    return path
```

and confirm `load_ohlcv(path)` on it returns a frame with the same length (`require_materialized` may want a sibling manifest; if so, read what it checks at `shares/core/io.py` and write that too, or pass the frame through `iter_completed_bars` directly in `build_dataset` tests — `build_dataset` takes a `source` path, so the file must load).

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_event_log.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain.research.event_log'`

- [ ] **Step 3: Write `event_log.py`**

```python
# brain/research/event_log.py
"""The Eye's transition events, one row each, as the Brain's Δₜ source.

A transition is an ``EventKind`` that reports a change: not the ``*_state``
re-publication of a retained entity, not the ``bar_completed`` heartbeat, not
the epoch reset. On the real tape the excluded kinds fire on 67-100 % of
bars; the 49 that remain are what an event clock is built from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract.eye import EventKind, MarketEvent

HEARTBEAT_KINDS: frozenset[str] = frozenset({"bar_completed", "market_epoch_reset"})
TRANSITION_KINDS: tuple[str, ...] = tuple(
    kind.value
    for kind in EventKind
    if not kind.value.endswith("_state") and kind.value not in HEARTBEAT_KINDS
)
EVENT_COLUMNS: tuple[str, ...] = (
    "known_at", "kind", "timeframe", "direction", "side", "strength",
    "price", "entity_id", "lifecycle", "event_id",
)


def is_transition(kind: EventKind) -> bool:
    return kind.value in TRANSITION_KINDS


def event_row(event: MarketEvent) -> dict[str, Any]:
    return {
        "known_at": pd.Timestamp(event.known_at if event.known_at is not None else event.observed_at),
        "kind": event.kind.value,
        "timeframe": event.timeframe.value,
        "direction": None if event.direction is None else event.direction.value,
        "side": event.side,
        "strength": float(event.strength),
        "price": None if event.price is None else float(event.price),
        "entity_id": event.entity_id,
        "lifecycle": event.lifecycle,
        "event_id": event.event_id,
    }


def empty_event_log() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in EVENT_COLUMNS})


def save_block(dataset, out_dir: Path, *, emit_end: str | None) -> None:
    """Write one block the way ``build_forecast_index`` caches a dataset, plus the log."""

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = np.ones(len(dataset.index), dtype=bool)
    if emit_end is not None:
        local = dataset.index.tz_convert("America/New_York")
        keep = (local < pd.Timestamp(emit_end, tz="America/New_York")).to_numpy()
    np.savez_compressed(
        out_dir / "dataset.npz",
        index=dataset.index[keep].tz_convert("UTC").tz_localize(None).to_numpy(),
        features=dataset.features[keep],
        prices=dataset.prices.to_numpy(dtype=float)[keep],
        future_closes=dataset.future_closes[keep],
        future_highs=dataset.future_highs[keep],
        future_lows=dataset.future_lows[keep],
    )
    events = dataset.events.copy()
    if emit_end is not None and len(events):
        limit = pd.Timestamp(emit_end, tz="America/New_York")
        events = events[events["known_at"] < limit]
    events.to_parquet(out_dir / "events.parquet", index=False)


def load_blocks(blocks_root: Path) -> dict:
    """Concatenate every block directory under ``blocks_root`` in name order."""

    from brain.scripts._windows import load_dataset

    parts = sorted(p for p in Path(blocks_root).iterdir() if (p / "dataset.npz").exists())
    if not parts:
        raise FileNotFoundError(f"no block under {blocks_root}")
    loaded = [load_dataset(p / "dataset.npz") for p in parts]
    logs = [pd.read_parquet(p / "events.parquet") for p in parts]
    merged = {
        "index": pd.DatetimeIndex(np.concatenate([d["index"].asi8 for d in loaded])).tz_localize("UTC")
        if hasattr(loaded[0]["index"], "asi8") else np.concatenate([d["index"] for d in loaded]),
    }
    for name in ("features", "prices", "future_closes", "future_highs", "future_lows"):
        merged[name] = np.concatenate([d[name] for d in loaded], axis=0)
    events = pd.concat(logs, ignore_index=True) if logs else empty_event_log()
    events["known_at"] = pd.to_datetime(events["known_at"], utc=True)
    merged["events"] = events.sort_values(["known_at", "event_id"], kind="stable").reset_index(drop=True)
    merged["blocks"] = tuple(p.name for p in parts)
    return merged
```

Check how `brain/scripts/_windows.load_dataset` returns `index` (it is a `DatetimeIndex` built from the stored naive UTC values — read lines 52-70 of `_windows.py`) and make `load_blocks` produce the same type so `session_labels` and `slice_window` accept it; the `hasattr(..., "asi8")` branch above is the `DatetimeIndex` case.

- [ ] **Step 4: Extend `TrajectoryDataset` and `build_dataset`**

In `brain/research/trajectory_dataset.py`:

```python
from brain.research.event_log import empty_event_log, event_row, is_transition
```

Add to `TrajectoryDataset` after `feature_names`:

```python
    events: pd.DataFrame = field(default_factory=empty_event_log)
```

(`from dataclasses import dataclass, field`). In `build_dataset`, beside `stamps`:

```python
    event_rows: list[dict] = []
```

and inside the bar loop, immediately after `observation = observer.observe(reader.on_bar(bar))`:

```python
        for event in observation.semantic_events_this_update:
            if not is_transition(event.kind):
                continue
            row = event_row(event)
            if row["known_at"] < emit_from:
                continue
            event_rows.append(row)
```

and in the return:

```python
        events=(
            pd.DataFrame(event_rows, columns=list(EVENT_COLUMNS))
            if event_rows else empty_event_log()
        ),
```

with `EVENT_COLUMNS` imported. Add `"events"` handling nowhere else: `brain/scripts/build_forecast_index.py` keeps ignoring it.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest brain/tests/test_event_log.py brain/tests/test_hypothesis_forecast.py brain/tests/test_eye_to_brain_link.py -q -o addopts=""`
Expected: all pass (the existing Brain tests must not notice the new field).

- [ ] **Step 6: Commit**

```bash
git add brain/research/event_log.py brain/research/trajectory_dataset.py brain/tests/test_event_log.py shares/tests/helpers.py
git commit -m "feat(brain): record the Eye's transition events beside the state dataset"
```

---

### Task 2: Weekly block builder

**Files:**
- Create: `brain/scripts/build_gate_blocks.py`
- Test: `brain/tests/test_gate_blocks.py`

**Interfaces:**
- Produces: `globex_weeks(first_session: str, last_session: str, *, warmup_days: int = 7) -> list[Block]` with `Block(week: str, warmup_start: str, emit_start: str, emit_end: str, end: str)`; `run_id(*, source: Path, model: Path, first_session: str, last_session: str) -> str`; CLI `python -m brain.scripts.build_gate_blocks --first-session 2022-01-03 --last-session 2022-06-06 --workers 8`.

- [ ] **Step 1: Write the failing test**

```python
# brain/tests/test_gate_blocks.py
"""Weekly blocks tile the sessions once, each warmed by the seven days before its open."""
from __future__ import annotations

import pandas as pd

from brain.scripts.build_gate_blocks import Block, globex_weeks


def test_weeks_tile_the_window_without_gaps_or_overlap() -> None:
    blocks = globex_weeks("2022-01-03", "2022-06-06")
    assert blocks[0].week == "2022-01-03"
    assert blocks[-1].week == "2022-06-06"
    # a Globex week opens Sunday 18:00 New York; emit_start is that open.
    assert blocks[0].emit_start == "2022-01-02T18:00"
    assert blocks[0].warmup_start == "2021-12-26T18:00"
    for previous, current in zip(blocks, blocks[1:]):
        assert previous.emit_end == current.emit_start
    # 22 Globex weeks between the first and last session inclusive
    assert len(blocks) == 23 or len(blocks) == 22


def test_the_last_block_ends_after_the_last_session_plus_horizon() -> None:
    blocks = globex_weeks("2022-01-03", "2022-01-07")
    assert len(blocks) == 1
    block = blocks[0]
    assert pd.Timestamp(block.end) > pd.Timestamp("2022-01-07T17:00")
```

Compute the expected block count precisely before finalising the first assertion: Monday 2022-01-03 is in the week opening Sunday 2022-01-02; Monday 2022-06-06 is in the week opening Sunday 2022-06-05; that is `(2022-06-05 − 2022-01-02) / 7 + 1 = 23` weeks. Fix the assertion to `== 23`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_blocks.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the builder**

```python
# brain/scripts/build_gate_blocks.py
"""Drive the Eye over Globex-week blocks and cache each block for the gate.

Each block is a fresh Eye warmed for the seven calendar days before the
week's Sunday 18:00 New York open and sampled through the week; the last
block runs one extra day past its final session so every sampled clock has
its full sixty-minute future. Blocks are independent, so ``--workers``
builds them in parallel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.research.event_log import save_block  # noqa: E402
from brain.research.trajectory_dataset import build_dataset  # noqa: E402

EXCHANGE_TZ = "America/New_York"
DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
DEFAULT_MODEL = "configs/model.json"


@dataclass(frozen=True)
class Block:
    week: str          # the Monday session date, used as the directory name
    warmup_start: str  # ISO local time, fed to the Eye, never sampled
    emit_start: str    # the week's Sunday 18:00 open
    emit_end: str      # the next week's open; rows at or after it are dropped
    end: str           # where the tape read stops (emit_end + one day)


def globex_weeks(first_session: str, last_session: str, *, warmup_days: int = 7) -> list[Block]:
    first = pd.Timestamp(first_session)
    last = pd.Timestamp(last_session)
    # The Sunday on or before each session date.
    first_open = (first - pd.Timedelta(days=(first.weekday() + 1) % 7)).normalize() + pd.Timedelta(hours=18)
    last_open = (last - pd.Timedelta(days=(last.weekday() + 1) % 7)).normalize() + pd.Timedelta(hours=18)
    blocks: list[Block] = []
    open_at = first_open
    while open_at <= last_open:
        next_open = open_at + pd.Timedelta(days=7)
        blocks.append(
            Block(
                week=(open_at + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                warmup_start=(open_at - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%dT%H:%M"),
                emit_start=open_at.strftime("%Y-%m-%dT%H:%M"),
                emit_end=next_open.strftime("%Y-%m-%dT%H:%M"),
                end=(next_open + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M"),
            )
        )
        open_at = next_open
    return blocks


def run_id(*, source: Path, model: Path, first_session: str, last_session: str) -> str:
    from eyes.core.semantics import load_semantic_selection

    selection = load_semantic_selection(json.loads(model.read_text())["semantic_selection"])
    payload = {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "model": str(model.relative_to(ROOT)) if model.is_relative_to(ROOT) else str(model),
        "first_session": first_session,
        "last_session": last_session,
        "block_rule": "globex_week_warmup_7d",
        "atomic_definition_identity": selection.atomic_definition_identity,
        "data_splits_sha256": hashlib.sha256((ROOT / "configs" / "data_splits.json").read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _build_one(args: tuple[Block, str, str, str]) -> str:
    block, source, model, out_root = args
    out = Path(out_root) / block.week
    if (out / "dataset.npz").exists() and (out / "events.parquet").exists():
        return f"{block.week}: cached"
    started = time.monotonic()
    dataset = build_dataset(
        source=Path(source), warmup_start=block.warmup_start, emit_start=block.emit_start,
        end=block.end, model_path=Path(model), root=ROOT,
    )
    save_block(dataset, out, emit_end=block.emit_end)
    return f"{block.week}: {len(dataset.index)} clocks, {len(dataset.events)} events, {(time.monotonic() - started) / 60:.1f} min"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--first-session", default="2022-01-03")
    parser.add_argument("--last-session", default="2022-06-06")
    parser.add_argument("--output-root", default="outputs/information_gain_gate")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    source, model = ROOT / args.source, ROOT / args.model
    identity = run_id(source=source, model=model, first_session=args.first_session, last_session=args.last_session)
    out_root = ROOT / args.output_root / identity / "blocks"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root.parent / "run.json").write_text(json.dumps({
        "run_id": identity, "first_session": args.first_session, "last_session": args.last_session,
        "source": str(source), "model": str(model), "block_rule": "globex_week_warmup_7d",
    }, indent=2))
    blocks = globex_weeks(args.first_session, args.last_session)
    print(f"run {identity}: {len(blocks)} blocks -> {out_root}", flush=True)
    jobs = [(block, str(source), str(model), str(out_root)) for block in blocks]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for line in pool.map(_build_one, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(_build_one(job), flush=True)


if __name__ == "__main__":
    main()
```

`load_semantic_selection` returns the validated selection; confirm the attribute name `atomic_definition_identity` on its return value (it printed `f92b24c8…` in the earlier probe via that attribute).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_blocks.py -q -o addopts=""`
Expected: `2 passed`

- [ ] **Step 5: Smoke one tiny block on the real tape**

Run: `.venv/bin/python -m brain.scripts.build_gate_blocks --first-session 2022-01-03 --last-session 2022-01-03 --output-root outputs/information_gain_gate_smoke`
Expected: one line `2022-01-03: N clocks, M events, T min` with N ≈ 6,800 and T well under 15 minutes (Part 1 done) — this is the per-block cost the 23-block run will multiply.

- [ ] **Step 6: Commit**

```bash
git add brain/scripts/build_gate_blocks.py brain/tests/test_gate_blocks.py
git commit -m "feat(brain): build the gate's Eye pass in warmed Globex-week blocks"
```

---

### Task 3: Clocks and the Δₜ encoding

**Files:**
- Create: `brain/research/event_sequence.py`
- Test: `brain/tests/test_event_sequence.py`

**Interfaces:**
- Produces:
  - `SCALE_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "4H": 240}`
  - `clock_mask(index: pd.DatetimeIndex, events: pd.DataFrame, *, min_timeframe_minutes: int) -> np.ndarray` (bool, len(index))
  - `sequence_features(index: pd.DatetimeIndex, events: pd.DataFrame, *, window_minutes: int = 120, slots: int = 4) -> tuple[np.ndarray, tuple[str, ...]]`
  - `without_kind(events: pd.DataFrame, kind: str) -> pd.DataFrame` (the ablation's event stream)

- [ ] **Step 1: Write the failing test**

```python
# brain/tests/test_event_sequence.py
"""Δₜ reads only the past, keeps order, and keeps one track per scale."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.event_log import EVENT_COLUMNS
from brain.research.event_sequence import (
    SCALE_MINUTES,
    clock_mask,
    sequence_features,
    without_kind,
)

T0 = pd.Timestamp("2022-01-03T10:00", tz="America/New_York")


def _clocks(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([T0 + pd.Timedelta(minutes=i) for i in range(n)])


def _events(rows: list[tuple[int, str, str, str | None]]) -> pd.DataFrame:
    """rows: (minute offset, kind, timeframe, direction)."""
    return pd.DataFrame(
        [
            {
                "known_at": T0 + pd.Timedelta(minutes=m), "kind": k, "timeframe": tf,
                "direction": d, "side": None, "strength": 1.0, "price": None,
                "entity_id": f"e{i}", "lifecycle": None, "event_id": f"ev{i}",
            }
            for i, (m, k, tf, d) in enumerate(rows)
        ],
        columns=list(EVENT_COLUMNS),
    )


def test_clock_membership_follows_the_timeframe_threshold() -> None:
    index = _clocks(5)
    events = _events([(1, "sweep_confirmed", "1m", "up"), (3, "qualified_bos", "15m", "down")])
    assert clock_mask(index, events, min_timeframe_minutes=1).tolist() == [False, True, False, True, False]
    assert clock_mask(index, events, min_timeframe_minutes=5).tolist() == [False, False, False, True, False]
    assert clock_mask(index, events, min_timeframe_minutes=15).tolist() == [False, False, False, True, False]


def test_a_future_event_leaves_the_vector_unchanged() -> None:
    index = _clocks(3)
    past = _events([(0, "sweep_confirmed", "5m", "up")])
    with_future = _events([(0, "sweep_confirmed", "5m", "up"), (2, "qualified_bos", "5m", "down")])
    a, names = sequence_features(index, past)
    b, _ = sequence_features(index, with_future)
    assert np.array_equal(a[1], b[1])  # at minute 1 the minute-2 event is not visible


def test_order_of_the_same_three_events_is_distinguishable() -> None:
    index = _clocks(4)
    forward = _events([(0, "sweep_confirmed", "5m", "up"), (1, "acceptance_confirmed", "5m", "up"), (2, "displacement_observed", "5m", "up")])
    reverse = _events([(0, "displacement_observed", "5m", "up"), (1, "sweep_confirmed", "5m", "up"), (2, "acceptance_confirmed", "5m", "up")])
    a, names = sequence_features(index, forward)
    b, _ = sequence_features(index, reverse)
    track = [i for i, n in enumerate(names) if n.startswith("track_5m_")]
    assert not np.array_equal(a[3, track], b[3, track])


def test_each_scale_keeps_its_own_track() -> None:
    index = _clocks(60)
    rows = [(0, "qualified_bos", "1H", "up")] + [(m, "level_touched", "1m", None) for m in range(1, 41)]
    x, names = sequence_features(index, _events(rows))
    kind_col = names.index("track_1H_0_kind")
    empty_col = names.index("track_1H_0_empty")
    assert x[59, empty_col] == 0.0
    assert x[59, kind_col] == float(sorted(set(k for _, k, _, _ in rows)).index("qualified_bos")) or x[59, kind_col] >= 0


def test_recency_count_and_direction_block() -> None:
    index = _clocks(10)
    x, names = sequence_features(index, _events([(2, "sweep_confirmed", "5m", "down"), (5, "sweep_confirmed", "5m", "up")]), window_minutes=120)
    since = names.index("sweep_confirmed@5m_since")
    count = names.index("sweep_confirmed@5m_count")
    direction = names.index("sweep_confirmed@5m_dir")
    assert x[1, since] == 120.0 and x[1, count] == 0.0 and x[1, direction] == 0.0
    assert x[2, since] == 0.0 and x[2, count] == 1.0 and x[2, direction] == -1.0
    assert x[9, since] == 4.0 and x[9, count] == 2.0 and x[9, direction] == 1.0


def test_trigger_one_hot_marks_the_kinds_fired_at_t() -> None:
    index = _clocks(3)
    x, names = sequence_features(index, _events([(1, "fvg_created", "5m", None)]))
    col = names.index("trigger_fvg_created")
    assert x[:, col].tolist() == [0.0, 1.0, 0.0]


def test_without_kind_drops_only_that_kind() -> None:
    events = _events([(0, "sweep_confirmed", "5m", "up"), (1, "qualified_bos", "5m", "up")])
    left = without_kind(events, "sweep_confirmed")
    assert list(left["kind"]) == ["qualified_bos"]
```

Kind ids in the track block are the position of the kind in `TRANSITION_KINDS` (fixed vocabulary order), so the fourth test's kind assertion should be `x[59, kind_col] == float(TRANSITION_KINDS.index("qualified_bos"))`; import `TRANSITION_KINDS` and write it that way.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_event_sequence.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the encoder**

```python
# brain/research/event_sequence.py
"""Δₜ: the Eye's recent transition events as a fixed-width, causal vector.

Three blocks, all read from events with ``known_at <= t``:

1. per (kind × scale): minutes since the last occurrence (capped at the
   window, the window when none), the count inside the window, and the
   direction of the last occurrence (+1 / -1 / 0);
2. one ordered track per scale holding the last ``slots`` events published
   on that scale — kind id, minutes ago, direction, strength, empty flag —
   because one-minute transitions arrive every ~1.3 minutes on the real tape
   and a shared track would never show a 15-minute or 1-hour sequence;
3. a one-hot over the kinds fired at ``t`` itself.

Nothing here carries a weight. Every coefficient that reads these columns is
fitted inside the training window of the gate.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from brain.research.event_log import TRANSITION_KINDS

SCALE_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "4H": 240}
SCALES: tuple[str, ...] = tuple(SCALE_MINUTES)
TRACK_FIELDS: tuple[str, ...] = ("kind", "since", "dir", "strength", "empty")
_KIND_ID = {kind: float(i) for i, kind in enumerate(TRANSITION_KINDS)}
_DIRECTION = {"up": 1.0, "down": -1.0, "bullish": 1.0, "bearish": -1.0, "long": 1.0, "short": -1.0}


def _direction(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return _DIRECTION.get(str(value), 0.0)


def _prepared(events: pd.DataFrame) -> pd.DataFrame:
    frame = events[events["kind"].isin(TRANSITION_KINDS) & events["timeframe"].isin(SCALES)]
    return frame.sort_values(["known_at", "event_id"], kind="stable").reset_index(drop=True)


def feature_names(*, slots: int = 4) -> tuple[str, ...]:
    names: list[str] = []
    for kind in TRANSITION_KINDS:
        for scale in SCALES:
            names += [f"{kind}@{scale}_since", f"{kind}@{scale}_count", f"{kind}@{scale}_dir"]
    for scale in SCALES:
        for slot in range(slots):
            names += [f"track_{scale}_{slot}_{field}" for field in TRACK_FIELDS]
    names += [f"trigger_{kind}" for kind in TRANSITION_KINDS]
    return tuple(names)


def clock_mask(index: pd.DatetimeIndex, events: pd.DataFrame, *, min_timeframe_minutes: int) -> np.ndarray:
    frame = _prepared(events)
    minutes = frame["timeframe"].map(SCALE_MINUTES)
    hits = pd.DatetimeIndex(frame.loc[minutes >= min_timeframe_minutes, "known_at"]).unique()
    return index.isin(hits)


def sequence_features(
    index: pd.DatetimeIndex, events: pd.DataFrame, *, window_minutes: int = 120, slots: int = 4
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = feature_names(slots=slots)
    frame = _prepared(events)
    known = frame["known_at"].to_numpy()
    kinds = frame["kind"].to_numpy()
    scales = frame["timeframe"].to_numpy()
    directions = np.array([_direction(v) for v in frame["direction"].to_numpy()])
    strengths = frame["strength"].to_numpy(dtype=float)

    out = np.zeros((len(index), len(names)), dtype=float)
    window = np.timedelta64(window_minutes, "m")
    column = {name: i for i, name in enumerate(names)}
    # Recency block state: last time and direction per (kind, scale), and a
    # deque of times inside the window for the count.
    last_at: dict[tuple[str, str], np.datetime64] = {}
    last_dir: dict[tuple[str, str], float] = {}
    recent: dict[tuple[str, str], deque] = {}
    tracks: dict[str, deque] = {scale: deque(maxlen=slots) for scale in SCALES}
    since_default = float(window_minutes)

    pointer = 0
    stamps = index.tz_convert("UTC").tz_localize(None).to_numpy() if index.tz is not None else index.to_numpy()
    known_naive = pd.DatetimeIndex(known).tz_convert("UTC").tz_localize(None).to_numpy() if pd.DatetimeIndex(known).tz is not None else known
    for row, t in enumerate(stamps):
        fired: list[str] = []
        while pointer < len(known_naive) and known_naive[pointer] <= t:
            key = (kinds[pointer], scales[pointer])
            last_at[key] = known_naive[pointer]
            last_dir[key] = directions[pointer]
            recent.setdefault(key, deque()).append(known_naive[pointer])
            tracks[scales[pointer]].append((known_naive[pointer], kinds[pointer], directions[pointer], strengths[pointer]))
            if known_naive[pointer] == t:
                fired.append(kinds[pointer])
            pointer += 1
        for key, at in last_at.items():
            kind, scale = key
            base = column[f"{kind}@{scale}_since"]
            queue = recent[key]
            while queue and queue[0] < t - window:
                queue.popleft()
            out[row, base] = min(float((t - at) / np.timedelta64(1, "m")), since_default)
            out[row, base + 1] = float(len(queue))
            out[row, base + 2] = last_dir[key]
        for scale in SCALES:
            slot_events = list(tracks[scale])[::-1]  # newest first
            for slot in range(slots):
                base = column[f"track_{scale}_{slot}_kind"]
                if slot < len(slot_events):
                    at, kind, direction, strength = slot_events[slot]
                    out[row, base] = _KIND_ID[kind]
                    out[row, base + 1] = float((t - at) / np.timedelta64(1, "m"))
                    out[row, base + 2] = direction
                    out[row, base + 3] = strength
                    out[row, base + 4] = 0.0
                else:
                    out[row, base : base + 5] = (-1.0, since_default, 0.0, 0.0, 1.0)
        for kind in fired:
            out[row, column[f"trigger_{kind}"]] = 1.0
    # Kinds never seen read "no occurrence": since = window, count 0, dir 0.
    never = np.zeros(len(names), dtype=bool)
    for kind in TRANSITION_KINDS:
        for scale in SCALES:
            if (kind, scale) not in last_at:
                never[column[f"{kind}@{scale}_since"]] = True
    out[:, never] = since_default
    return out, names


def without_kind(events: pd.DataFrame, kind: str) -> pd.DataFrame:
    return events[events["kind"] != kind].reset_index(drop=True)
```

Two details to get right while implementing: (a) the "since" default for a (kind, scale) that has occurred *later* but not yet at row `t` is also the window — the loop only writes keys in `last_at`, so rows before a key's first occurrence keep the `never` fill only if the key never occurs at all; fix by initialising `out[:, since_columns] = since_default` before the loop instead of the `never` pass, and let the loop overwrite; (b) the direction vocabulary — check `Direction` in `contract/market/primitives.py` for its actual values and put those in `_DIRECTION`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest brain/tests/test_event_sequence.py -q -o addopts=""`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add brain/research/event_sequence.py brain/tests/test_event_sequence.py
git commit -m "feat(brain): encode the Eye's event sequence causally, one track per scale"
```

---

### Task 4: First-passage targets

**Files:**
- Create: `brain/research/first_passage.py`
- Test: `brain/tests/test_first_passage.py`

**Interfaces:**
- Produces: `FIRST_PASSAGE_TARGETS: tuple[tuple[str, float, float], ...] = (("fp_1.0_1.0", 1.0, 1.0), ("fp_1.0_0.5", 1.0, 0.5), ("fp_0.5_1.0", 0.5, 1.0))`; `NEITHER, UPPER_FIRST, LOWER_FIRST = 0, 1, 2`; `first_passage_labels(*, prices: np.ndarray, future_highs: np.ndarray, future_lows: np.ndarray, up_atr: float, down_atr: float, horizon: int = 60) -> np.ndarray` (int64, shape rows).

- [ ] **Step 1: Write the failing test**

```python
# brain/tests/test_first_passage.py
from __future__ import annotations

import numpy as np

from brain.research.first_passage import (
    LOWER_FIRST, NEITHER, UPPER_FIRST, FIRST_PASSAGE_TARGETS, first_passage_labels,
)


def _future(highs: list[float], lows: list[float], horizon: int = 60):
    h = np.full((1, horizon), 100.0); l = np.full((1, horizon), 100.0)
    h[0, : len(highs)] = highs; l[0, : len(lows)] = lows
    return h, l


def test_upper_first() -> None:
    h, l = _future([100.5, 101.2], [99.8, 99.9])
    prices = np.array([[100.0, 100.0, 100.0, 1.0]])
    assert first_passage_labels(prices=prices, future_highs=h, future_lows=l, up_atr=1.0, down_atr=0.5)[0] == UPPER_FIRST


def test_lower_first() -> None:
    h, l = _future([100.2, 100.3], [99.4, 99.9])
    prices = np.array([[100.0, 100.0, 100.0, 1.0]])
    assert first_passage_labels(prices=prices, future_highs=h, future_lows=l, up_atr=1.0, down_atr=0.5)[0] == LOWER_FIRST


def test_neither_within_horizon() -> None:
    h, l = _future([100.4] * 60, [99.7] * 60)
    prices = np.array([[100.0, 100.0, 100.0, 1.0]])
    assert first_passage_labels(prices=prices, future_highs=h, future_lows=l, up_atr=1.0, down_atr=0.5)[0] == NEITHER


def test_same_bar_tie_is_lower_first() -> None:
    h, l = _future([101.0], [99.5])
    prices = np.array([[100.0, 100.0, 100.0, 1.0]])
    assert first_passage_labels(prices=prices, future_highs=h, future_lows=l, up_atr=1.0, down_atr=0.5)[0] == LOWER_FIRST


def test_barriers_scale_with_the_anchor_atr() -> None:
    h, l = _future([102.0], [99.0])
    prices = np.array([[100.0, 100.0, 100.0, 4.0]])  # ATR 4: +2 is only half an ATR
    assert first_passage_labels(prices=prices, future_highs=h, future_lows=l, up_atr=1.0, down_atr=1.0)[0] == NEITHER


def test_three_registered_targets() -> None:
    assert [t[0] for t in FIRST_PASSAGE_TARGETS] == ["fp_1.0_1.0", "fp_1.0_0.5", "fp_0.5_1.0"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_first_passage.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
# brain/research/first_passage.py
"""Which barrier the next sixty minutes reach first, in ATR units.

Three classes: neither barrier inside the horizon, the upper first, the lower
first. A bar that touches both is read as the lower first: the conservative
reading for a long claim, the same convention ``configs/model.json`` names
``same_bar_resolution: conservative`` for the risk engine.
"""
from __future__ import annotations

import numpy as np

NEITHER, UPPER_FIRST, LOWER_FIRST = 0, 1, 2
FIRST_PASSAGE_TARGETS: tuple[tuple[str, float, float], ...] = (
    ("fp_1.0_1.0", 1.0, 1.0),
    ("fp_1.0_0.5", 1.0, 0.5),
    ("fp_0.5_1.0", 0.5, 1.0),
)


def first_passage_labels(
    *, prices: np.ndarray, future_highs: np.ndarray, future_lows: np.ndarray,
    up_atr: float, down_atr: float, horizon: int = 60,
) -> np.ndarray:
    anchor = prices[:, 0].reshape(-1, 1)
    scale = prices[:, 3].reshape(-1, 1)
    up_hit = (future_highs[:, :horizon] - anchor) / scale >= up_atr
    down_hit = (anchor - future_lows[:, :horizon]) / scale >= down_atr
    first_up = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), horizon)
    first_down = np.where(down_hit.any(axis=1), down_hit.argmax(axis=1), horizon)
    labels = np.full(prices.shape[0], NEITHER, dtype=np.int64)
    labels[first_up < first_down] = UPPER_FIRST
    labels[(first_down <= first_up) & (first_down < horizon)] = LOWER_FIRST
    return labels
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest brain/tests/test_first_passage.py -q -o addopts=""`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add brain/research/first_passage.py brain/tests/test_first_passage.py
git commit -m "feat(brain): first-passage labels in ATR units for the gate"
```

---

### Task 5: Classification scoring, session-block bootstrap, Holm and the family verdict

**Files:**
- Create: `brain/research/gate_family.py`
- Test: `brain/tests/test_gate_family.py`

**Interfaces:**
- Produces:
  - `log_loss_rows(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray` (per-row −log p[y], probabilities clipped to [1e-15, 1])
  - `session_block_bootstrap(differences: np.ndarray, sessions: np.ndarray, *, draws: int = 2000, seed: int = 29) -> tuple[float, float, float, float]` → (mean, low95, high95, p_one_sided) where p = fraction of draws with mean ≥ 0
  - `pooled_session_bootstrap(parts: list[tuple[np.ndarray, np.ndarray]], *, draws=2000, seed=31) -> tuple[float, float, float, float]` resampling each fold's sessions independently
  - `holm(pvalues: Sequence[float], *, alpha: float) -> list[bool]`
  - `family_verdict(results: pd.DataFrame, pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]], *, alpha: float = 0.10) -> pd.DataFrame` with one row per (clock, target, model) and columns `primary_delta, primary_ci_low, primary_ci_high, p_raw, p_holm, folds, folds_beating, consistent, robust, PASS`.
  - `results` columns consumed: `clock, target, model, fold, primary (bool), delta_logloss` (mean per-row M₁ − M₀ on that fold), `holdout_from`.

- [ ] **Step 1: Write the failing test**

```python
# brain/tests/test_gate_family.py
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.gate_family import (
    family_verdict, holm, log_loss_rows, pooled_session_bootstrap, session_block_bootstrap,
)


def test_log_loss_rows_is_negative_log_probability_of_the_true_class() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    y = np.array([0, 2])
    assert np.allclose(log_loss_rows(p, y), -np.log([0.7, 0.8]))


def test_holm_rejects_in_step_down_order() -> None:
    assert holm([0.001, 0.02, 0.04, 0.5], alpha=0.10) == [True, True, True, False]
    assert holm([0.05, 0.05, 0.05], alpha=0.10) == [False, False, False]


def test_session_bootstrap_flags_a_clear_improvement() -> None:
    rng = np.random.default_rng(0)
    sessions = np.repeat(np.arange(20).astype(str), 50)
    differences = rng.normal(-0.05, 0.1, size=sessions.size)
    mean, low, high, p = session_block_bootstrap(differences, sessions)
    assert mean < 0 and high < 0 and p < 0.01


def test_session_bootstrap_does_not_flag_noise() -> None:
    rng = np.random.default_rng(1)
    sessions = np.repeat(np.arange(20).astype(str), 50)
    differences = rng.normal(0.0, 0.1, size=sessions.size)
    _, low, high, p = session_block_bootstrap(differences, sessions)
    assert low < 0 < high and p > 0.05


def _results(delta_primary: float, rolling: list[float]) -> tuple[pd.DataFrame, dict]:
    rows = [{"clock": "C5", "target": "fp_1.0_1.0", "model": "lightgbm", "fold": 0, "primary": True,
             "delta_logloss": delta_primary, "holdout_from": "2022-05-10"}]
    rows += [{"clock": "C5", "target": "fp_1.0_1.0", "model": "lightgbm", "fold": i + 1, "primary": False,
              "delta_logloss": d, "holdout_from": f"2022-0{2 + i}-01"} for i, d in enumerate(rolling)]
    rng = np.random.default_rng(3)
    sessions = np.repeat(np.arange(20).astype(str), 40)
    pooled = {("C5", "fp_1.0_1.0", "lightgbm"): [(rng.normal(delta_primary, 0.05, sessions.size), sessions)]}
    return pd.DataFrame(rows), pooled


def test_family_verdict_passes_a_consistent_robust_significant_cell() -> None:
    results, pooled = _results(-0.03, [-0.02, -0.01, -0.03, -0.02, -0.01])
    verdict = family_verdict(results, pooled, alpha=0.10)
    assert verdict.loc[0, "PASS"]


def test_family_verdict_fails_when_the_best_fold_carries_it() -> None:
    results, pooled = _results(-0.03, [-0.20, 0.01, 0.02, 0.01, 0.02])
    verdict = family_verdict(results, pooled, alpha=0.10)
    assert not verdict.loc[0, "PASS"]
    assert not verdict.loc[0, "consistent"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_family.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
# brain/research/gate_family.py
"""Scoring and the pre-registered family verdict of the information-gain gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def log_loss_rows(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -np.log(clipped[np.arange(labels.size), labels])


def _session_means(differences: np.ndarray, sessions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(sessions, kind="stable")
    names, starts = np.unique(sessions[order], return_index=True)
    sums = np.add.reduceat(differences[order], starts)
    counts = np.diff(np.append(starts, sessions.size))
    return sums, counts


def session_block_bootstrap(
    differences: np.ndarray, sessions: np.ndarray, *, draws: int = 2000, seed: int = 29
) -> tuple[float, float, float, float]:
    """Resample whole sessions: rows inside a session share their regime."""

    sums, counts = _session_means(differences, sessions)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, sums.size, size=(draws, sums.size))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p_one_sided = float((means >= 0.0).mean())
    return float(differences.mean()), float(low), float(high), p_one_sided


def pooled_session_bootstrap(
    parts: list[tuple[np.ndarray, np.ndarray]], *, draws: int = 2000, seed: int = 31
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws); weights = np.zeros(draws)
    for differences, sessions in parts:
        sums, counts = _session_means(differences, sessions)
        picks = rng.integers(0, sums.size, size=(draws, sums.size))
        totals += sums[picks].sum(axis=1); weights += counts[picks].sum(axis=1)
    means = totals / weights
    low, high = np.percentile(means, [2.5, 97.5])
    everything = np.concatenate([d for d, _ in parts])
    return float(everything.mean()), float(low), float(high), float((means >= 0.0).mean())


def holm(pvalues: Sequence[float], *, alpha: float) -> list[bool]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            rejected[i] = True
        else:
            break
    return rejected


def family_verdict(
    results: pd.DataFrame,
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]],
    *, alpha: float = 0.10,
) -> pd.DataFrame:
    rows = []
    for (clock, target, model), block in results.groupby(["clock", "target", "model"], sort=False):
        primary = block[block["primary"]]
        rolling = block[~block["primary"]]
        parts = pooled[(clock, target, model)]
        mean, low, high, p_raw = pooled_session_bootstrap(parts)
        primary_delta = float(primary["delta_logloss"].iloc[0]) if len(primary) else float("nan")
        beating = (rolling["delta_logloss"] < 0).mean() if len(rolling) else 0.0
        without_best = rolling.sort_values("delta_logloss").iloc[1:]["delta_logloss"].mean() if len(rolling) > 1 else float("nan")
        rows.append({
            "clock": clock, "target": target, "model": model,
            "primary_delta": primary_delta, "primary_ci_low": low, "primary_ci_high": high,
            "p_raw": p_raw, "folds": int(len(rolling)), "folds_beating": float(beating),
            "consistent": bool(beating >= 0.8),
            "robust": bool(np.isfinite(without_best) and without_best < 0.0),
        })
    frame = pd.DataFrame(rows)
    frame["p_holm_reject"] = holm(frame["p_raw"].tolist(), alpha=alpha)
    frame["PASS"] = (
        (frame["primary_delta"] < 0.0) & (frame["primary_ci_high"] < 0.0)
        & frame["p_holm_reject"] & frame["consistent"] & frame["robust"]
    )
    return frame
```

The Holm family in the real run must be exactly the nine verdict cells (three first-passage targets × three clocks) per model class; `family_verdict` applies Holm across whatever rows it is given, so the script (Task 6) calls it once per model class with only those nine rows, and reports every other cell without Holm.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest brain/tests/test_gate_family.py -q -o addopts=""`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add brain/research/gate_family.py brain/tests/test_gate_family.py
git commit -m "feat(brain): session-block bootstrap, Holm and the family verdict for the gate"
```

---

### Task 6: The gate script

**Files:**
- Create: `brain/scripts/information_gain_gate.py`
- Test: `brain/tests/test_information_gain_gate.py`

**Interfaces:**
- Consumes: `load_blocks` (Task 1), `clock_mask`, `sequence_features`, `without_kind` (Task 3), `FIRST_PASSAGE_TARGETS`, `first_passage_labels` (Task 4), `gate_family.*` (Task 5), and from `brain/scripts/predictability_gate.py`: `raw_features`, `session_labels`, `build_folds`, `standardize_pair`, `select_ridge_alpha`, `fit_predict`, `TAPE_FEATURE_NAMES`, `HORIZON`, `EXCHANGE_TZ`.
- Produces: `run_gate(data: dict, *, out_dir: Path, clocks: dict[str, int], primary: tuple[int, int], rolling: tuple[int, int, int], models: tuple[str, ...], ablation: bool) -> pd.DataFrame` (the verdict), and CLI `python -m brain.scripts.information_gain_gate --run-id <id>`.

- [ ] **Step 1: Write the failing test**

```python
# brain/tests/test_information_gain_gate.py
"""The gate runs end to end on a synthetic tape and writes every table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain.research.event_log import EVENT_COLUMNS
from brain.scripts.information_gain_gate import run_gate


def _synthetic_data(sessions: int = 12, per_session: int = 300, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    rows = sessions * per_session
    start = pd.Timestamp("2022-01-03T09:30", tz="America/New_York")
    stamps = []
    for s in range(sessions):
        day = start + pd.Timedelta(days=s if s < 5 else s + 2 * (s // 5))
        stamps += [day + pd.Timedelta(minutes=i) for i in range(per_session)]
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    closes = 100 + np.cumsum(rng.normal(0, 0.3, rows))
    prices = np.column_stack([closes, closes + 0.2, closes - 0.2, np.full(rows, 1.0)])
    steps = rng.normal(0, 0.3, (rows, 60)).cumsum(axis=1)
    future_closes = closes[:, None] + steps
    events = pd.DataFrame(
        [{"known_at": index[i], "kind": "sweep_confirmed", "timeframe": "5m", "direction": "up",
          "side": None, "strength": 1.0, "price": None, "entity_id": f"e{i}", "lifecycle": None, "event_id": f"ev{i}"}
         for i in range(0, rows, 7)],
        columns=list(EVENT_COLUMNS),
    )
    return {
        "index": index, "features": rng.normal(size=(rows, 150)), "prices": prices,
        "future_closes": future_closes, "future_highs": future_closes + 0.3, "future_lows": future_closes - 0.3,
        "volumes": np.ones(rows), "events": events,
    }


def test_run_gate_writes_results_ablation_and_verdict(tmp_path: Path) -> None:
    data = _synthetic_data()
    verdict = run_gate(
        data, out_dir=tmp_path, clocks={"C1": 1, "C5": 5},
        primary=(8, 2), rolling=(6, 2, 2), models=("logistic",), ablation=False, descriptive={},
    )
    assert (tmp_path / "results.csv").exists() and (tmp_path / "verdict.csv").exists()
    assert set(verdict["clock"]) == {"C1", "C5"}
    assert {"fp_1.0_1.0", "fp_1.0_0.5", "fp_0.5_1.0"} <= set(verdict["target"])
    assert verdict["PASS"].dtype == bool
```

The `features` in the synthetic dict has 150 columns to match `FEATURE_NAMES`, from which the script drops the 8 tape names to form the `eye` group.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest brain/tests/test_information_gain_gate.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the script**

```python
# brain/scripts/information_gain_gate.py
"""The information-gain gate: does Δₜ improve sixty-minute forecasts on event clocks?

Pre-registered in brain/docs/specs/2026-09-11-information-gain-gate-design.md.
M₀ = RAW + S (the existing raw and eye groups); M₁ = M₀ + Δ. Both are fitted
and scored on the same clock set. Verdict cells are the three first-passage
targets on C1, C5 and C15, Holm-corrected per model class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.core.hypothesis_proposer import FEATURE_NAMES  # noqa: E402
from brain.research.event_log import TRANSITION_KINDS, load_blocks  # noqa: E402
from brain.research.event_sequence import clock_mask, sequence_features, without_kind  # noqa: E402
from brain.research.first_passage import FIRST_PASSAGE_TARGETS, first_passage_labels  # noqa: E402
from brain.research.gate_family import family_verdict, log_loss_rows, session_block_bootstrap  # noqa: E402
from brain.scripts.predictability_gate import (  # noqa: E402
    EXCHANGE_TZ, HORIZON, TAPE_FEATURE_NAMES, Fold, build_folds, fit_predict,
    raw_features, session_labels, standardize_pair,
)

VERDICT_TARGETS = tuple(name for name, _, _ in FIRST_PASSAGE_TARGETS)
CLOCKS = {"C1": 1, "C5": 5, "C15": 15}
# Reported, never judged: too thin (C60) or the dilution reference (ALL).
DESCRIPTIVE_CLOCKS = {"C60": 60, "ALL": 0}
PRIMARY = (90, 20)          # train sessions, holdout sessions
ROLLING = (60, 10, 10)      # train, holdout, step


def continuous_targets(prices: np.ndarray, future_highs: np.ndarray, future_lows: np.ndarray) -> dict[str, np.ndarray]:
    anchor = prices[:, 0].reshape(-1, 1); scale = prices[:, 3].reshape(-1, 1)
    up = np.maximum((future_highs[:, :HORIZON].max(axis=1).reshape(-1, 1) - anchor) / scale, 0.0)[:, 0]
    down = np.maximum((anchor - future_lows[:, :HORIZON].min(axis=1).reshape(-1, 1)) / scale, 0.0)[:, 0]
    span = up + down
    return {
        "asymmetry_60": np.divide(up - down, span, out=np.zeros_like(span), where=span > 1e-9),
        "range_60": span,
    }


def fit_predict_proba(model: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    classes = 3
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression

        best_c, best = 1.0, -np.inf
        rows = train_x.shape[0]
        edges = np.linspace(0, rows, 6, dtype=int)
        for c in (0.001, 0.01, 0.1, 1.0):
            scores = []
            for b in range(5):
                lo, hi = edges[b], edges[b + 1]
                test = np.arange(lo, hi)
                train = np.concatenate([np.arange(0, max(0, lo - HORIZON)), np.arange(min(rows, hi + HORIZON), rows)])
                if train.size < 200 or test.size < 100 or len(np.unique(train_y[train])) < 2:
                    continue
                fitted = LogisticRegression(C=c, max_iter=2000).fit(train_x[train], train_y[train])
                proba = _full_proba(fitted, train_x[test], classes)
                scores.append(-log_loss_rows(proba, train_y[test]).mean())
            if scores and np.mean(scores) > best:
                best, best_c = float(np.mean(scores)), c
        fitted = LogisticRegression(C=best_c, max_iter=2000).fit(train_x, train_y)
        return _full_proba(fitted, test_x, classes)
    if model == "lightgbm":
        import lightgbm as lgb

        cut = int(train_x.shape[0] * 0.8)
        fit_x, fit_y = train_x[: cut - HORIZON], train_y[: cut - HORIZON]
        tail_x, tail_y = train_x[cut:], train_y[cut:]
        estimator = lgb.LGBMClassifier(
            objective="multiclass", num_class=classes, n_estimators=400, learning_rate=0.03,
            num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=10.0, random_state=0, verbose=-1,
        )
        estimator.fit(fit_x, fit_y, eval_set=[(tail_x, tail_y)], callbacks=[lgb.early_stopping(30, verbose=False)])
        return _full_proba(estimator, test_x, classes)
    raise ValueError(f"unknown classifier {model}")


def _full_proba(estimator, x: np.ndarray, classes: int) -> np.ndarray:
    proba = np.full((x.shape[0], classes), 1e-6)
    partial = estimator.predict_proba(x)
    for position, label in enumerate(estimator.classes_):
        proba[:, int(label)] = partial[:, position]
    return proba / proba.sum(axis=1, keepdims=True)


def folds_for(index: pd.DatetimeIndex, mask: np.ndarray) -> list[tuple[Fold, bool]]:
    """Primary fold then rolling folds, all on the clock's rows only."""

    rows = np.flatnonzero(mask)
    sub = index[rows]
    out: list[tuple[Fold, bool]] = []
    primary = build_folds(sub, train_sessions=PRIMARY[0], holdout_sessions=PRIMARY[1],
                          embargo_minutes=HORIZON, step_sessions=10_000, start_session=None)[0]
    out.append((Fold(primary.index, rows[primary.train], rows[primary.holdout], primary.train_sessions, primary.holdout_sessions), True))
    for fold in build_folds(sub, train_sessions=ROLLING[0], holdout_sessions=ROLLING[1],
                            embargo_minutes=HORIZON, step_sessions=ROLLING[2], start_session=None):
        out.append((Fold(fold.index + 1, rows[fold.train], rows[fold.holdout], fold.train_sessions, fold.holdout_sessions), False))
    return out


def run_gate(
    data: dict, *, out_dir: Path, clocks: dict[str, int] = CLOCKS,
    primary: tuple[int, int] = PRIMARY, rolling: tuple[int, int, int] = ROLLING,
    models: tuple[str, ...] = ("logistic", "lightgbm"), ablation: bool = True,
    descriptive: dict[str, int] = DESCRIPTIVE_CLOCKS,
) -> pd.DataFrame:
    global PRIMARY, ROLLING
    PRIMARY, ROLLING = primary, rolling
    out_dir.mkdir(parents=True, exist_ok=True)
    index: pd.DatetimeIndex = data["index"]
    if index.tz is None:
        index = index.tz_localize("UTC")
    prices = data["prices"]
    volumes = data.get("volumes", np.ones(len(index)))
    raw, raw_names = raw_features(closes=prices[:, 0], highs=prices[:, 1], lows=prices[:, 2],
                                  volumes=volumes, atrs=prices[:, 3], index=index)
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in TAPE_FEATURE_NAMES]
    eye = data["features"][:, keep]
    base = np.hstack([raw, eye])
    delta, delta_names = sequence_features(index, data["events"])
    labels = {name: first_passage_labels(prices=prices, future_highs=data["future_highs"],
                                         future_lows=data["future_lows"], up_atr=up, down_atr=down)
              for name, up, down in FIRST_PASSAGE_TARGETS}
    continuous = continuous_targets(prices, data["future_highs"], data["future_lows"])
    sessions = session_labels(index)

    results: list[dict] = []
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    ablation_rows: list[dict] = []

    def score_cell(clock: str, fold: Fold, is_primary: bool, target: str, model: str,
                   m0_x: np.ndarray, m1_x: np.ndarray, tag: str = "full") -> np.ndarray:
        if target in labels:
            y = labels[target]
            p0 = fit_predict_proba(model, m0_x[fold.train], y[fold.train], m0_x[fold.holdout])
            p1 = fit_predict_proba(model, m1_x[fold.train], y[fold.train], m1_x[fold.holdout])
            e0 = log_loss_rows(p0, y[fold.holdout]); e1 = log_loss_rows(p1, y[fold.holdout])
        else:
            y = continuous[target]
            reg = "ridge" if model == "logistic" else "lightgbm"
            p0 = fit_predict(reg, m0_x[fold.train], y[fold.train], m0_x[fold.holdout])
            p1 = fit_predict(reg, m1_x[fold.train], y[fold.train], m1_x[fold.holdout])
            e0 = (y[fold.holdout] - p0) ** 2; e1 = (y[fold.holdout] - p1) ** 2
        return e1 - e0

    for clock, minutes in {**clocks, **descriptive}.items():
        judged = clock in clocks
        mask = clock_mask(index, data["events"], min_timeframe_minutes=minutes)
        print(f"{clock}: {int(mask.sum())} clocks of {mask.size}", flush=True)
        for fold, is_primary in folds_for(index, mask):
            if not judged and not is_primary:
                continue  # descriptive clocks get the primary fold only
            m0_train, m0_test = standardize_pair(base[fold.train], base[fold.holdout])
            m1_train, m1_test = standardize_pair(np.hstack([base, delta])[fold.train], np.hstack([base, delta])[fold.holdout])
            m0_x = np.zeros_like(base); m0_x[fold.train] = m0_train; m0_x[fold.holdout] = m0_test
            m1_x = np.zeros((base.shape[0], base.shape[1] + delta.shape[1])); m1_x[fold.train] = m1_train; m1_x[fold.holdout] = m1_test
            for target in (*VERDICT_TARGETS, "asymmetry_60", "range_60"):
                for model in models:
                    diff = score_cell(clock, fold, is_primary, target, model, m0_x, m1_x)
                    hold_sessions = sessions[fold.holdout]
                    mean, low, high, p = session_block_bootstrap(diff, hold_sessions)
                    results.append({"clock": clock, "target": target, "model": model, "fold": fold.index,
                                    "primary": is_primary, "holdout_from": fold.holdout_sessions[0],
                                    "delta_logloss": float(diff.mean()), "ci_low": low, "ci_high": high, "p_raw": p,
                                    "rows": int(diff.size)})
                    if is_primary:
                        pooled.setdefault((clock, target, model), []).append((diff, hold_sessions))
                    if judged and is_primary and ablation and target in labels:
                        for kind in TRANSITION_KINDS:
                            reduced, _ = sequence_features(index, without_kind(data["events"], kind))
                            r_train, r_test = standardize_pair(np.hstack([base, reduced])[fold.train], np.hstack([base, reduced])[fold.holdout])
                            r_x = np.zeros_like(m1_x); r_x[fold.train] = r_train; r_x[fold.holdout] = r_test
                            ablated = score_cell(clock, fold, True, target, model, m1_x, r_x, tag=kind)
                            ablation_rows.append({"clock": clock, "target": target, "model": model, "kind": kind,
                                                  "delta_logloss_vs_full": float(ablated.mean())})
    frame = pd.DataFrame(results)
    frame.to_csv(out_dir / "results.csv", index=False)
    if ablation_rows:
        pd.DataFrame(ablation_rows).sort_values(["clock", "target", "model", "delta_logloss_vs_full"], ascending=[True, True, True, False]).to_csv(out_dir / "ablation.csv", index=False)
    verdicts = []
    for model in models:
        cells = frame[(frame["model"] == model) & frame["target"].isin(VERDICT_TARGETS) & frame["clock"].isin(clocks)]
        verdicts.append(family_verdict(cells, {k: v for k, v in pooled.items() if k[2] == model and k[1] in VERDICT_TARGETS}, alpha=0.10))
    verdict = pd.concat(verdicts, ignore_index=True)
    verdict.to_csv(out_dir / "verdict.csv", index=False)
    print(verdict.to_string(index=False))
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/information_gain_gate")
    parser.add_argument("--models", default="logistic,lightgbm")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()
    run_root = ROOT / args.output_root / args.run_id
    data = load_blocks(run_root / "blocks")
    source = json.loads((run_root / "run.json").read_text())["source"]
    from shares.core.io import load_ohlcv
    tape = load_ohlcv(source, start="2021-12-26", end="2022-06-15").frame
    volumes = tape["volume"].reindex(pd.DatetimeIndex(data["index"]).tz_convert(tape.index.tz)).to_numpy(dtype=float)
    data["volumes"] = np.where(np.isfinite(volumes), volumes, 1.0)
    run_gate(data, out_dir=run_root, models=tuple(args.models.split(",")), ablation=not args.no_ablation)


if __name__ == "__main__":
    main()
```

Notes for the implementer: the ablation computes `sequence_features` 49 times on the primary fold per (clock, target, model) — hoist it: compute `reduced` once per kind per clock (it does not depend on target or model) and cache in a dict before the target loop. `raw_features` shifts by up to 240 rows and leaves NaN at the top of each block; `standardize_pair` imputes NaN to the centre, which is the existing gate's behaviour. `volumes` for the real run come from the tape at the sampled clocks — note the dataset `index` is the snapshot `asof` (bar close), so align on `index − 1 min` if `reindex` returns NaN everywhere; check with `np.isfinite(volumes).mean()` and print it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest brain/tests/test_information_gain_gate.py -q -o addopts=""`
Expected: `1 passed` (allow ~1 minute)

- [ ] **Step 5: Commit**

```bash
git add brain/scripts/information_gain_gate.py brain/tests/test_information_gain_gate.py
git commit -m "feat(brain): the information-gain gate over three clocks with the family verdict"
```

---

### Task 7: Run on the real tape and write the receipt

**Files:**
- Create: `brain/docs/evidence/<run date>_information_gain_gate.md`
- Modify: `brain/docs/README.md` (one paragraph pointing at the receipt and the verdict)

- [ ] **Step 1: Build the 23 blocks**

Run: `.venv/bin/python -m brain.scripts.build_gate_blocks --first-session 2022-01-03 --last-session 2022-06-06 --workers 8 2>&1 | tee outputs/information_gain_gate/blocks.log`
Expected: 23 lines, each `YYYY-MM-DD: ~6,900 clocks, N events, T min`; note the run id printed on the first line.

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/python -m brain.scripts.information_gain_gate --run-id <id> 2>&1 | tee outputs/information_gain_gate/<id>/gate.log`
Expected: three `Cn: k clocks of N` lines with k/N near 75 %, 17 %, 3.6 %; then the verdict table. Any `GateError`/`ValueError` stops the run — read it, fix the cause in the module it names, re-run (blocks are cached).

- [ ] **Step 3: Receipt**

`brain/docs/evidence/<run date>_information_gain_gate.md`: run id and `run.json`, block table (clocks/events per week), clock coverage, the full `verdict.csv` for both model classes, the `asymmetry_60`/`range_60` and C60/every-minute reference rows from `results.csv`, the top ten kinds from `ablation.csv` per clock, the Holm-adjusted decision per cell, and one sentence stating the outcome: PASS (which cells) → C starts; FAIL → the redesign stops at the measurement. Add the paragraph to `brain/docs/README.md` under the forecast-skill section.

- [ ] **Step 4: Commit**

```bash
git add brain/docs/evidence/ brain/docs/README.md
git commit -m "docs(brain): information-gain gate receipt and verdict"
```
