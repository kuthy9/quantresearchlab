# Eye Per-Bar Cost (Part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Eye's per-bar cost proportional to what changed this bar, with its published output byte-identical before and after, so the gate's weekly blocks run in minutes.

**Architecture:** Three memoisation/index changes on the measured growth owners (liquidity candidate projection, `InteractionUpdate` DTO re-admission, structural-leg eligible-bar scan), each guarded by a per-bar hash-stream harness that records `content_hash` of every observation and event batch and must diff empty against a reference captured before any edit.

**Tech Stack:** Python 3.12, existing Eye (`eyes/core`, `contract/eye`), pytest, `.venv/bin/python`.

**Spec:** `brain/docs/specs/2026-09-11-information-gain-gate-design.md` §2, §4.

## Global Constraints

- No change to `configs/*`, `semantics/*`, `maximum_context_states`, `history_limit`, eviction order, retention semantics, `EventKind`, any `to_primitive` payload. The atomic identity stays `f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c`.
- No `eyes/core/` module may import `brain`, `execution` or `shares.core.engine` (enforced by `eyes/tests/test_eye_module_boundary.py`).
- Every task ends with the hash-stream diff empty against the reference (Task 1) and `eyes/tests` + `shares/tests` collecting and passing as before. Run tests with `--ignore-glob='* 2.py'` to skip the iCloud duplicates on disk.
- Commit only after the diff is empty; one commit per task. Do not commit `outputs/` (ignored).
- The repo's Python is `.venv/bin/python`; run from the repository root.

---

### Task 1: Hash-stream harness and reference capture

**Files:**
- Create: `eyes/scripts/replay_hash_stream.py`
- Test: `eyes/tests/test_replay_hash_stream.py`

**Interfaces:**
- Produces: `hash_stream(bars: Iterable[Bar], *, model_path: Path, root: Path) -> list[tuple[str, str, str, int]]` returning one `(asof_iso, observation_hash, events_hash, n_events)` per bar; CLI `python -m eyes.scripts.replay_hash_stream --start 2022-01-09 --end 2022-01-13 --output outputs/eye_hash_stream/reference.csv` that also prints wall time per 500 bars.

- [ ] **Step 1: Write the failing test**

```python
# eyes/tests/test_replay_hash_stream.py
"""The hash-stream harness is the acceptance test of every Part 1 change."""
from __future__ import annotations

from pathlib import Path

from eyes.scripts.replay_hash_stream import hash_stream
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "configs" / "model.json"


def test_two_runs_over_the_same_bars_produce_the_same_stream() -> None:
    bars = session_bars(1)[:300]
    first = hash_stream(bars, model_path=MODEL, root=ROOT)
    second = hash_stream(bars, model_path=MODEL, root=ROOT)
    assert len(first) == 300
    assert first == second


def test_the_stream_changes_when_a_bar_changes() -> None:
    bars = session_bars(1)[:300]
    altered = list(bars)
    altered[150] = altered[150].__class__(
        **{**altered[150].__dict__, "close": altered[150].close + 0.25}
    )
    reference = hash_stream(bars, model_path=MODEL, root=ROOT)
    changed = hash_stream(altered, model_path=MODEL, root=ROOT)
    assert reference[:150] == changed[:150]
    assert reference[150][1] != changed[150][1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest eyes/tests/test_replay_hash_stream.py -q -o addopts=""`
Expected: FAIL with `ModuleNotFoundError: No module named 'eyes.scripts.replay_hash_stream'`

- [ ] **Step 3: Write the harness**

```python
# eyes/scripts/replay_hash_stream.py
"""Per-bar hash stream of the Eye's published output.

One line per completed bar: the clock, ``content_hash`` of the whole
``MarketObservation``, ``content_hash`` of the events published on that bar,
and the event count. Two runs of the same code over the same bars produce the
same stream (the Eye is deterministic), so a change that leaves the stream
identical has not changed what the Eye publishes. This is the acceptance test
for every cost change in brain/docs/specs/2026-09-11-information-gain-gate-design.md §4.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract.market import Bar, content_hash  # noqa: E402
from eyes.core.causal import CausalMarketReader  # noqa: E402
from eyes.core.observation import CausalObserver  # noqa: E402
from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402

DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"


def build_eye(model_path: Path, *, root: Path) -> tuple[CausalMarketReader, CausalObserver]:
    """The same construction ``brain/research/trajectory_dataset.build_eye`` uses."""

    from brain.research.trajectory_dataset import build_eye as _build_eye

    return _build_eye(model_path, root=root)


def hash_stream(
    bars: Iterable[Bar], *, model_path: Path, root: Path
) -> list[tuple[str, str, str, int]]:
    reader, observer = build_eye(model_path, root=root)
    rows: list[tuple[str, str, str, int]] = []
    for bar in bars:
        observation = observer.observe(reader.on_bar(bar))
        events = observation.semantic_events_this_update
        rows.append(
            (
                observation.asof.isoformat(),
                content_hash(observation),
                content_hash(events),
                len(events),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default="configs/model.json")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many bars")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-every", type=int, default=500)
    args = parser.parse_args()

    frame = load_ohlcv(ROOT / args.source, start=args.start, end=args.end).frame
    bars = list(iter_completed_bars(frame))
    if args.limit:
        bars = bars[: args.limit]
    reader, observer = build_eye(ROOT / args.model, root=ROOT)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    started = block_started = time.monotonic()
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("asof", "observation_hash", "events_hash", "n_events"))
        for number, bar in enumerate(bars, start=1):
            observation = observer.observe(reader.on_bar(bar))
            events = observation.semantic_events_this_update
            writer.writerow(
                (
                    observation.asof.isoformat(),
                    content_hash(observation),
                    content_hash(events),
                    len(events),
                )
            )
            if args.timing_every and number % args.timing_every == 0:
                now = time.monotonic()
                block = now - block_started
                print(
                    f"bars {number - args.timing_every:6d}-{number:6d}: {block:6.1f}s "
                    f"({args.timing_every / block:5.1f} bars/s)",
                    flush=True,
                )
                block_started = now
    print(f"{len(bars)} bars -> {out}  ({(time.monotonic() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
```

Note: `Bar` is a frozen dataclass; if `altered[150].__class__(**{**__dict__, ...})` fails because `Bar` post-processes fields, use `dataclasses.replace(altered[150], close=altered[150].close + 0.25)` in the test instead.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest eyes/tests/test_replay_hash_stream.py -q -o addopts=""`
Expected: `2 passed`

- [ ] **Step 5: Capture the reference stream and the timing baseline on the real tape**

Run: `.venv/bin/python -m eyes.scripts.replay_hash_stream --start 2022-01-09 --end 2022-01-13 --limit 4500 --output outputs/eye_hash_stream/reference.csv`
Expected: nine `bars ... bars/s` lines rising from roughly 9 s to roughly 45 s per 500 bars (this is the §2 curve), then `4500 bars -> .../reference.csv`. Copy the nine timing lines into `outputs/eye_hash_stream/reference_timing.txt` for the receipt.

- [ ] **Step 6: Commit**

```bash
git add eyes/scripts/replay_hash_stream.py eyes/tests/test_replay_hash_stream.py
git commit -m "test(eye): per-bar hash-stream harness for output-preserving cost changes"
```

---

### Task 2: Memoise the liquidity candidate projection

**Files:**
- Modify: `eyes/core/market_state.py:2086-2140` (`_project_candidate_views`, `_settled_candidate_state`)
- Test: `eyes/tests/test_candidate_projection_memo.py`

**Interfaces:**
- Consumes: `_project_candidate_views(liquidity, hierarchy, range_state) -> TimeframeLiquidityState` (existing signature, unchanged).
- Produces: module-level `_PROJECTION_MEMO: dict[int, tuple[weakref.ref, tuple, DOLCandidateView]]` and helper `_projected_candidate(candidate, rank, range_key) -> DOLCandidateView`.

- [ ] **Step 1: Write the failing test**

```python
# eyes/tests/test_candidate_projection_memo.py
"""Projecting an unchanged candidate under an unchanged range costs nothing."""
from __future__ import annotations

from dataclasses import replace

from eyes.core import market_state
from eyes.core.market_state import (
    _project_candidate_views,
    _projected_candidate,
    DOLCandidateView,
    TimeframeLiquidityState,
    TimeframeRangeState,
)


def _candidate(candidate_id: str, price: float) -> DOLCandidateView:
    # Build through the same constructor the reducer uses; every required
    # field is given explicitly so the object is exactly what a reducer emits.
    return DOLCandidateView(
        candidate_id=candidate_id,
        price=price,
        side="above",
        lifecycle="armed",
        source_kind="registered_level",
    )


def _range(low: float | None, high: float | None) -> TimeframeRangeState:
    return TimeframeRangeState(
        range_id="r1" if low is not None else None,
        range_kind="active_dealing_range" if low is not None else None,
        low=low,
        high=high,
    )


def test_the_same_candidate_under_the_same_range_projects_to_the_same_object() -> None:
    candidate = _candidate("c1", 101.0)
    key = market_state._range_key(_range(100.0, 110.0))
    first = _projected_candidate(candidate, candidate.rank, key)
    second = _projected_candidate(candidate, candidate.rank, key)
    assert first is second


def test_a_new_range_reprojects() -> None:
    candidate = _candidate("c1", 101.0)
    inside = _projected_candidate(candidate, candidate.rank, market_state._range_key(_range(100.0, 110.0)))
    outside = _projected_candidate(candidate, candidate.rank, market_state._range_key(_range(102.0, 110.0)))
    assert inside is not outside
    assert inside.range_role != outside.range_role


def test_a_new_candidate_object_with_equal_fields_is_projected_afresh() -> None:
    a = _candidate("c1", 101.0)
    b = replace(a)
    key = market_state._range_key(_range(100.0, 110.0))
    assert _projected_candidate(a, a.rank, key) is not _projected_candidate(b, b.rank, key)


def test_project_candidate_views_output_is_unchanged() -> None:
    candidates = (_candidate("c1", 101.0), _candidate("c2", 99.0))
    liquidity = TimeframeLiquidityState(
        unswept_bsl=(), unswept_ssl=(), candidate_bsl_ids=(), candidate_ssl_ids=(),
        candidates=candidates,
    )
    projected_once = _project_candidate_views(liquidity, (), _range(100.0, 110.0))
    projected_twice = _project_candidate_views(liquidity, (), _range(100.0, 110.0))
    assert projected_once == projected_twice
    assert projected_once.candidates[0] is projected_twice.candidates[0]
```

If `DOLCandidateView`, `TimeframeLiquidityState` or `TimeframeRangeState` require more fields than shown, read their `__init__` signatures with `.venv/bin/python -c "import inspect, eyes.core.market_state as m; print(inspect.signature(m.DOLCandidateView))"` and supply the minimal valid values; the assertions do not depend on the extra fields.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest eyes/tests/test_candidate_projection_memo.py -q -o addopts=""`
Expected: FAIL with `ImportError: cannot import name '_projected_candidate'`

- [ ] **Step 3: Implement the memo**

In `eyes/core/market_state.py`, add `import weakref` to the imports, and above `_project_candidate_views`:

```python
# One projection per candidate object, rank and range key. Candidate views
# are frozen dataclasses and the reducer keeps the same object in the state
# tuple until an event replaces it, so an unchanged bar re-projects nothing.
# Keyed by ``id`` with a weak reference guard: a recycled id whose object is
# gone, or a different object at the same id, misses the cache.
_PROJECTION_MEMO: dict[int, tuple["weakref.ReferenceType[DOLCandidateView]", tuple[Any, ...], DOLCandidateView]] = {}
_PROJECTION_MEMO_LIMIT = 200_000


def _range_key(range_state: TimeframeRangeState) -> tuple[Any, ...]:
    """The five range fields ``_candidate_range_membership`` reads."""

    return (
        range_state.range_id,
        range_state.range_kind,
        range_state.lifecycle,
        range_state.low,
        range_state.high,
    )


def _projected_candidate(
    candidate: DOLCandidateView,
    rank: Any,
    range_key: tuple[Any, ...],
) -> DOLCandidateView:
    key = id(candidate)
    hit = _PROJECTION_MEMO.get(key)
    if hit is not None:
        reference, hit_inputs, projected = hit
        if reference() is candidate and hit_inputs == (rank, range_key):
            return projected
    if len(_PROJECTION_MEMO) >= _PROJECTION_MEMO_LIMIT:
        _PROJECTION_MEMO.clear()
    ranked = (
        candidate if rank == candidate.rank else replace(candidate, rank=rank)
    )
    projected = _candidate_range_membership(
        ranked, _RangeKeyView(*range_key)
    )
    _PROJECTION_MEMO[key] = (weakref.ref(candidate), (rank, range_key), projected)
    return projected
```

`_candidate_range_membership` reads `range_state.range_id`, `.range_kind`, `.lifecycle`, `.low`, `.high`. Rather than construct a `TimeframeRangeState`, pass the real one: change `_projected_candidate` to take `range_state: TimeframeRangeState` as its third parameter and compute `range_key = _range_key(range_state)` inside; the tests then call `_projected_candidate(candidate, rank, _range(...))`. Drop `_RangeKeyView`. (Adjust the test calls accordingly — the assertions stay.)

Then rewrite the body of `_project_candidate_views` to use it:

```python
    ranks = {item.swing_id: item.semantic_rank.value for item in hierarchy}
    return _liquidity_state(
        (
            _projected_candidate(
                candidate,
                ranks.get(
                    candidate.candidate_id.removeprefix("swing:"),
                    candidate.rank,
                )
                if ranks and candidate.source_kind == "confirmed_swing"
                else candidate.rank,
                range_state,
            )
            for candidate in liquidity.candidates
        ),
        liquidity.recently_swept_ids,
    )
```

This preserves the original semantics exactly: the original applied `replace(candidate, rank=...)` only for confirmed swings with a hierarchy rank, then `_candidate_range_membership`; here the rank argument equals `candidate.rank` in every other case, and `replace` is skipped when the rank is unchanged (a `replace` with identical fields yields an equal object, so the projected value is the same; only identity differs, and nothing downstream compares identity).

Also remove the now-unused `Any` import warning if any; keep `from dataclasses import replace` as is.

- [ ] **Step 4: Run the new test and the market-state tests**

Run: `.venv/bin/python -m pytest eyes/tests/test_candidate_projection_memo.py eyes/tests/test_hierarchical_market_state.py eyes/tests/test_liquidity_level_rearm.py eyes/tests/test_liquidity_projection_cost.py -q -o addopts=""`
Expected: all pass

- [ ] **Step 5: Hash-stream diff**

Run: `.venv/bin/python -m eyes.scripts.replay_hash_stream --start 2022-01-09 --end 2022-01-13 --limit 4500 --output outputs/eye_hash_stream/after_task2.csv && diff -q outputs/eye_hash_stream/reference.csv outputs/eye_hash_stream/after_task2.csv && echo IDENTICAL`
Expected: `IDENTICAL`, and the timing lines lower than the reference from the second block on. If the diff is not empty, find the first differing line with `diff outputs/eye_hash_stream/reference.csv outputs/eye_hash_stream/after_task2.csv | head -3`, revert the change, and re-examine: the projection must be a pure function of (candidate, rank, range fields) — a hit on stale inputs means a range field outside `_range_key` is being read.

- [ ] **Step 6: Full Eye and shares suites**

Run: `.venv/bin/python -m pytest eyes/tests shares/tests --ignore-glob='* 2.py' -q -p no:cacheprovider 2>&1 | tail -3`
Expected: the same 8 collection errors as before (the engine/validation imports, unrelated) and no new failures; pass count unchanged or higher.

- [ ] **Step 7: Commit**

```bash
git add eyes/core/market_state.py eyes/tests/test_candidate_projection_memo.py
git commit -m "perf(eye): project each liquidity candidate once per rank and range, not once per bar"
```

---

### Task 3: Memoise `InteractionUpdate` DTO re-admission

**Files:**
- Modify: `contract/eye/interaction.py:397-470` (`validate_canonical_bindings`, inner `exact_values`)
- Test: `eyes/tests/test_interaction_admission_memo.py`

**Interfaces:**
- Produces: module-level `_ADMITTED: dict[int, weakref.ReferenceType]` and `_is_admitted(value) -> bool`, `_remember_admitted(value) -> None` in `contract/eye/interaction.py`.

- [ ] **Step 1: Write the failing test**

```python
# eyes/tests/test_interaction_admission_memo.py
"""A DTO that passed canonical re-admission once is not reconstructed again."""
from __future__ import annotations

from dataclasses import replace

import contract.eye.interaction as interaction_contract
from contract.eye.interaction import InteractionUpdate, _is_admitted
from shares.tests.legacy_group5 import sample_path_sequence_state


def test_a_path_is_remembered_after_its_first_admission() -> None:
    path = sample_path_sequence_state()
    assert not _is_admitted(path)
    InteractionUpdate(
        zone_interactions=(), reacceptance_interactions=(), micro_break_facts=(),
        interaction_paths=(path,), interaction_path_transitions=(),
        reacceptance_interaction_transitions=(), milestone_transitions=(),
        cold_source_ids=(),
    )
    assert _is_admitted(path)
    for step in path.steps:
        assert _is_admitted(step)


def test_an_equal_but_distinct_object_is_admitted_on_its_own(monkeypatch) -> None:
    path = sample_path_sequence_state()
    twin = replace(path)
    calls: list[str] = []
    original = interaction_contract.PathSequenceState

    class Counting(original):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            calls.append("reconstruct")
            super().__init__(*args, **kwargs)

    # exact_values rebuilds through ``kind(**state)``; count those rebuilds.
    monkeypatch.setattr(interaction_contract, "PathSequenceState", Counting)
    InteractionUpdate(
        zone_interactions=(), reacceptance_interactions=(), micro_break_facts=(),
        interaction_paths=(path,), interaction_path_transitions=(),
        reacceptance_interaction_transitions=(), milestone_transitions=(),
        cold_source_ids=(),
    )
    InteractionUpdate(
        zone_interactions=(), reacceptance_interactions=(), micro_break_facts=(),
        interaction_paths=(path,), interaction_path_transitions=(),
        reacceptance_interaction_transitions=(), milestone_transitions=(),
        cold_source_ids=(),
    )
    assert calls.count("reconstruct") == 0  # ``type(value) is kind`` fails for Counting; see note
```

Note on the second test: `exact_values` requires `type(value) is kind`, so a monkeypatched subclass makes the check fail rather than count. Replace the second test with a direct count on the memo instead:

```python
def test_an_equal_but_distinct_object_is_admitted_on_its_own() -> None:
    path = sample_path_sequence_state()
    twin = replace(path)
    update = dict(
        zone_interactions=(), reacceptance_interactions=(), micro_break_facts=(),
        interaction_path_transitions=(), reacceptance_interaction_transitions=(),
        milestone_transitions=(), cold_source_ids=(),
    )
    InteractionUpdate(interaction_paths=(path,), **update)
    assert _is_admitted(path) and not _is_admitted(twin)
    InteractionUpdate(interaction_paths=(twin,), **update)
    assert _is_admitted(twin)


def test_a_freed_object_does_not_leave_a_stale_entry() -> None:
    import gc
    path = sample_path_sequence_state()
    key = id(path)
    InteractionUpdate(
        zone_interactions=(), reacceptance_interactions=(), micro_break_facts=(),
        interaction_paths=(path,), interaction_path_transitions=(),
        reacceptance_interaction_transitions=(), milestone_transitions=(),
        cold_source_ids=(),
    )
    assert key in interaction_contract._ADMITTED
    del path
    gc.collect()
    assert key not in interaction_contract._ADMITTED
```

`shares/tests/legacy_group5.py` must expose a `sample_path_sequence_state()` returning a valid `PathSequenceState` with at least one step. If it does not, build one in the test from the fixtures `eyes/tests/test_v3_group5_primitives.py` already uses (search that file for `PathSequenceState(`), copying the exact keyword arguments.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest eyes/tests/test_interaction_admission_memo.py -q -o addopts=""`
Expected: FAIL with `ImportError: cannot import name '_is_admitted'`

- [ ] **Step 3: Implement the memo**

In `contract/eye/interaction.py` add `import weakref` and, at module level above `class InteractionUpdate`:

```python
# DTO instances that have passed ``exact_values`` once. Every nested DTO is a
# frozen dataclass, so an instance that was canonical stays canonical; the
# validator re-admits only objects it has not seen. Keyed by ``id`` with a
# weak reference so a freed object cannot leave an entry a recycled id could
# hit, and the callback removes the entry when the object is collected.
_ADMITTED: dict[int, "weakref.ReferenceType[Any]"] = {}


def _is_admitted(value: Any) -> bool:
    reference = _ADMITTED.get(id(value))
    return reference is not None and reference() is value


def _remember_admitted(value: Any) -> None:
    key = id(value)

    def _forget(_reference: Any, key: int = key) -> None:
        _ADMITTED.pop(key, None)

    _ADMITTED[key] = weakref.ref(value, _forget)
```

Then in `exact_values`, wrap the per-value body:

```python
            for value in values:
                if _is_admitted(value):
                    continue
                if (
                    type(value) is not kind
                    or set(getattr(value, "__dict__", ())) != expected
                ):
                    raise ValueError(f"interaction {label} shape changed")
                state = {name: getattr(value, name) for name in names}
                try:
                    canonical = kind(**state)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"interaction {label} canonical state changed"
                    ) from error
                if any(
                    type(state[name]) is not type(getattr(canonical, name))
                    or state[name] != getattr(canonical, name)
                    for name in names
                ):
                    raise ValueError(
                        f"interaction {label} canonical state changed"
                    )
                _remember_admitted(value)
```

Everything after `exact_values` in `validate_canonical_bindings` (the `index_by` uniqueness checks, boundary checks, lineage checks) stays exactly as it is: those are cross-record checks on the update, not per-DTO admission, and they are cheap.

- [ ] **Step 4: Run the new test and the interaction tests**

Run: `.venv/bin/python -m pytest eyes/tests/test_interaction_admission_memo.py eyes/tests/test_v3_group5_primitives.py eyes/tests/test_interaction_eye_brain_boundary.py -q -o addopts=""`
Expected: all pass

- [ ] **Step 5: Hash-stream diff**

Run: `.venv/bin/python -m eyes.scripts.replay_hash_stream --start 2022-01-09 --end 2022-01-13 --limit 4500 --output outputs/eye_hash_stream/after_task3.csv && diff -q outputs/eye_hash_stream/reference.csv outputs/eye_hash_stream/after_task3.csv && echo IDENTICAL`
Expected: `IDENTICAL`

- [ ] **Step 6: Full Eye and shares suites**

Run: `.venv/bin/python -m pytest eyes/tests shares/tests --ignore-glob='* 2.py' -q -p no:cacheprovider 2>&1 | tail -3`
Expected: same collection errors as before, no new failures.

- [ ] **Step 7: Commit**

```bash
git add contract/eye/interaction.py eyes/tests/test_interaction_admission_memo.py
git commit -m "perf(eye): re-admit an interaction DTO once, not on every bar it is republished"
```

---

### Task 4: Index the eligible bars the structural-leg contract reads

**Files:**
- Modify: `eyes/core/event_store.py:495-500` (fields), `:679-800` (`append_batch` staging and commit), `:1567-1830` (`_validate_structural_leg_contract`)
- Test: `eyes/tests/test_event_store_eligible_bar_index.py`

**Interfaces:**
- Produces: `EventStore._eligible_bar_index: dict[tuple[Timeframe, str, int], list[MarketEvent]]` maintained on commit; `EventStore._eligible_bars(event, *, available_events, staged_bars) -> tuple[MarketEvent, ...]` returning the same tuple the scan at line 1771 returns; `_validate_structural_leg_contract` gains a keyword `eligible_bars: tuple[MarketEvent, ...] | None = None` and computes the scan only when it is `None`.

- [ ] **Step 1: Write the failing test**

```python
# eyes/tests/test_event_store_eligible_bar_index.py
"""The indexed eligible-bar sequence equals the scanned one, bar for bar."""
from __future__ import annotations

from pathlib import Path

from contract.eye import EventKind, EventOrigin
from contract.market import bar_evidence_coverage
from eyes.core.event_store import EventStore
from eyes.scripts.replay_hash_stream import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]


def _scanned(store: EventStore, event) -> tuple:
    symbol = event.evidence.get("symbol")
    instrument_id = event.evidence.get("instrument_id")
    return tuple(
        sorted(
            (
                c for c in store._by_id.values()
                if c.kind is EventKind.BAR_COMPLETED
                and c.origin is EventOrigin.NORMALIZED_DATA
                and c.semantic_version == event.semantic_version
                and c.timeframe is event.timeframe
                and c.event_time == c.known_at
                and bar_evidence_coverage(c.evidence).admits_definitional_path
                and c.evidence.get("symbol") == symbol
                and c.evidence.get("instrument_id") == instrument_id
            ),
            key=lambda c: (c.known_at, c.event_id),
        )
    )


def test_index_matches_scan_for_every_structural_leg_event() -> None:
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT)
    for bar in session_bars(1)[:600]:
        observer.observe(reader.on_bar(bar))
    store = observer.memory._audit_store if hasattr(observer.memory, "_audit_store") else observer.audit_store
    legs = [e for e in store._by_id.values() if e.kind is EventKind.STRUCTURAL_LEG_CREATED]
    assert legs, "the synthetic session produced no structural leg"
    for leg in legs:
        indexed = store._eligible_bars(leg, available_events=store._by_id, staged_bars=())
        assert indexed == _scanned(store, leg)
```

Locate the `EventStore` instance the observer writes to with `.venv/bin/python -c "..."` if the attribute name above is wrong (`grep -n "EventStore(" eyes/core/observation.py` shows where it is created and stored).

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest eyes/tests/test_event_store_eligible_bar_index.py -q -o addopts=""`
Expected: FAIL with `AttributeError: 'EventStore' object has no attribute '_eligible_bars'`

- [ ] **Step 3: Implement the index**

In `EventStore.__init__` (near `self._by_id`):

```python
        # Committed BAR_COMPLETED events with NORMALIZED_DATA origin, keyed by
        # (timeframe, symbol, instrument_id) and kept in (known_at, event_id)
        # order. ``_validate_structural_leg_contract`` reads its eligible
        # sequence from here instead of scanning every committed event.
        self._eligible_bar_index: dict[
            tuple[Timeframe, Any, Any], list[MarketEvent]
        ] = {}
```

A helper that says whether an event belongs in the index (the predicates that do not depend on the leg event being validated):

```python
    @staticmethod
    def _eligible_bar_key(event: MarketEvent) -> tuple[Timeframe, Any, Any] | None:
        if (
            event.kind is not EventKind.BAR_COMPLETED
            or event.origin is not EventOrigin.NORMALIZED_DATA
            or event.event_time != event.known_at
        ):
            return None
        return (
            event.timeframe,
            event.evidence.get("symbol"),
            event.evidence.get("instrument_id"),
        )
```

The reader, applying the leg-dependent predicates and merging the batch's staged bars:

```python
    def _eligible_bars(
        self,
        event: MarketEvent,
        *,
        available_events: Mapping[str, MarketEvent],
        staged_bars: tuple[MarketEvent, ...],
    ) -> tuple[MarketEvent, ...]:
        key = (
            event.timeframe,
            event.evidence.get("symbol"),
            event.evidence.get("instrument_id"),
        )
        committed = self._eligible_bar_index.get(key, ())
        candidates = [
            bar
            for bar in (*committed, *staged_bars)
            if self._eligible_bar_key(bar) == key
            and bar.semantic_version == event.semantic_version
            and bar_evidence_coverage(bar.evidence).admits_definitional_path
        ]
        candidates.sort(key=lambda bar: (bar.known_at, bar.event_id))
        return tuple(candidates)
```

In `append_batch`, keep a `staged_bars: list[MarketEvent] = []` beside `staged_events`; after `staged_events[event.event_id] = event`, add `if self._eligible_bar_key(event) is not None: staged_bars.append(event)`. Pass `eligible_bars=self._eligible_bars(event, available_events=available_events, staged_bars=tuple(staged_bars))` into the call chain that reaches `_validate_structural_leg_contract` — that is `_validate_canonical_provenance` → `_validate_authoritative_parent_contract` → `_validate_structural_leg_contract`; thread a keyword `eligible_bars` through the two intermediate signatures with default `None`, and only compute it when `event.kind is EventKind.STRUCTURAL_LEG_CREATED` (cheap guard: `eligible_bars = self._eligible_bars(...) if event.kind is EventKind.STRUCTURAL_LEG_CREATED else None`). Confirm the exact kind name with `grep -n "_validate_structural_leg_contract" eyes/core/event_store.py` and read the dispatch that calls it.

In `_validate_structural_leg_contract` replace the `eligible_bars = tuple(sorted(...))` block with:

```python
        if eligible_bars is None:
            eligible_bars = tuple(
                sorted(
                    (
                        candidate
                        for candidate in available_events.values()
                        if candidate.kind is EventKind.BAR_COMPLETED
                        and candidate.origin is EventOrigin.NORMALIZED_DATA
                        and candidate.semantic_version == event.semantic_version
                        and candidate.timeframe is event.timeframe
                        and candidate.event_time == candidate.known_at
                        and bar_evidence_coverage(
                            candidate.evidence
                        ).admits_definitional_path
                        and candidate.evidence.get("symbol") == symbol
                        and candidate.evidence.get("instrument_id") == instrument_id
                    ),
                    key=lambda candidate: (
                        candidate.known_at,
                        candidate.event_id,
                    ),
                )
            )
```

(the scan survives as the fallback for callers that pass nothing, so cold-replay and checkpoint paths that call the validator directly are unchanged).

At the commit point of `append_batch` (where `self._events.append(event)` / `self._by_id[event.event_id] = event` happen for each staged event), add:

```python
            key = self._eligible_bar_key(event)
            if key is not None:
                bucket = self._eligible_bar_index.setdefault(key, [])
                position = bisect.bisect_right(
                    [(bar.known_at, bar.event_id) for bar in bucket],
                    (event.known_at, event.event_id),
                )
                bucket.insert(position, event)
```

with `import bisect`. If the store has a restore/`__setstate__` or a cold-replay constructor that fills `_by_id` without going through `append_batch`, rebuild the index there with one pass over `self._events` using `_eligible_bar_key`; `grep -n "_by_id\[" eyes/core/event_store.py` lists every writer.

- [ ] **Step 4: Run the new test and the event-store tests**

Run: `.venv/bin/python -m pytest eyes/tests/test_event_store_eligible_bar_index.py eyes/tests/test_event_provenance_contract.py eyes/tests/test_phase1_semantic_identity_journal.py -q -o addopts=""`
Expected: all pass

- [ ] **Step 5: Hash-stream diff**

Run: `.venv/bin/python -m eyes.scripts.replay_hash_stream --start 2022-01-09 --end 2022-01-13 --limit 4500 --output outputs/eye_hash_stream/after_task4.csv && diff -q outputs/eye_hash_stream/reference.csv outputs/eye_hash_stream/after_task4.csv && echo IDENTICAL`
Expected: `IDENTICAL`

- [ ] **Step 6: Full Eye and shares suites**

Run: `.venv/bin/python -m pytest eyes/tests shares/tests --ignore-glob='* 2.py' -q -p no:cacheprovider 2>&1 | tail -3`
Expected: same collection errors as before, no new failures.

- [ ] **Step 7: Commit**

```bash
git add eyes/core/event_store.py eyes/tests/test_event_store_eligible_bar_index.py
git commit -m "perf(eye): read a structural leg's eligible bars from an index kept on commit"
```

---

### Task 5: Re-measure and record

**Files:**
- Create: `eyes/docs/evidence/2026-09-12_per_bar_cost.md`
- Modify: `AGENTS.md` (one paragraph under "Runtime Architecture" naming the harness)

- [ ] **Step 1: Re-measure the §2 curve over 8,000 bars**

Run: `.venv/bin/python -m eyes.scripts.replay_hash_stream --start 2022-01-09 --end 2022-01-19 --limit 8000 --output outputs/eye_hash_stream/after_all.csv 2>&1 | tee outputs/eye_hash_stream/after_all_timing.txt`
Expected: sixteen timing lines. Acceptance from the spec: the eighth 500-bar block costs at most 1.5× the first. Also confirm `head -4501 outputs/eye_hash_stream/after_all.csv | diff -q - outputs/eye_hash_stream/reference.csv && echo IDENTICAL` prints `IDENTICAL` (the first 4,500 bars of the longer run are the reference window).

- [ ] **Step 2: Write the receipt**

`eyes/docs/evidence/2026-09-12_per_bar_cost.md` with: the three commits' hashes, the reference timing table (Task 1 step 5) beside the after-all table, the acceptance ratio (block 8 / block 1) before and after, the hash-stream identity statement with the reference file's sha256 (`shasum -a 256 outputs/eye_hash_stream/reference.csv`), and the sentence that the reducer's per-event liquidity rebuild still grows with the one-minute candidate set and is bounded by the gate's weekly blocks, not by this change.

- [ ] **Step 3: AGENTS.md**

Add after the Eye-to-Brain link paragraph:

```markdown
Cost changes in the Eye are accepted only by the per-bar hash stream:
`python -m eyes.scripts.replay_hash_stream --start ... --end ... --output ...`
writes `content_hash` of every observation and event batch, and a change is
output-preserving when the stream diffs empty against a reference captured
before the edit. The 2026-09-12 receipt in `eyes/docs/evidence/` records the
three memoisation/index changes made this way.
```

- [ ] **Step 4: Commit**

```bash
git add eyes/docs/evidence/2026-09-12_per_bar_cost.md AGENTS.md
git commit -m "docs(eye): receipt for the output-preserving per-bar cost changes"
```
