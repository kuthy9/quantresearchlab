# Stop floor and event sleep implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A hard stop never nearer the entry than one bar of the thesis's governing scale (so the gate sizes on volatility), and a Brain that sleeps flat through CPI, NFP and FOMC statements and wakes fresh.

**Architecture:** One rule appended to `resolve_geometry`; a calendar module that parses an `.ics` into event windows; the controller gains an `events` section (schema 5) and an `EVENT_SLEEP` decision; the runtime archives on it; the order machine withdraws every expression on that bar; a build script assembles the calendar from committed sources.

**Tech Stack:** Python 3.12, pytest, pandas (tz handling); DeepSeek via `run_llm_brain`.

**Spec:** [brain/docs/specs/2026-09-21-stop-floor-event-sleep-design.md](../specs/2026-09-21-stop-floor-event-sleep-design.md)

## Global Constraints

- Test command: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider` (never `-q`).
- `STOP_FLOOR_GOVERNING_BARS = 1.0`; no risk fraction, leverage, quantity or reward-to-risk value changes.
- Event windows: CPI / NFP 60 minutes before to 30 after; FOMC 60 before to 90 after the 14:00 US-Eastern statement.
- No dates or times in code: they come from `brain/configs/economic_calendar.ics`, built from `brain/configs/calendar_sources/`.
- Nothing is committed until asked; nothing under `outputs/` is deleted.

---

### Task 1: The stop floor

**Files:**
- Modify: `brain/core/opportunity_geometry.py` (`resolve_geometry`, new `STOP_FLOOR_GOVERNING_BARS`, `_floor_stop`)
- Test: `brain/tests/test_opportunity_geometry.py`

**Interfaces:**
- Produces: `resolve_geometry(..., floor_bars: float = STOP_FLOOR_GOVERNING_BARS)`; stop rule id `stop.floor.governing_bar` when the floor moved the stop; `GeometryError("a stop floor needs a positive 1m atr")` when the opportunity has a `governing_timeframe` and `atr_1m` is missing.

- [x] Step 1: Write the failing tests — LONG 15m thesis with a near swing stop is floored (`atr_1m=2.0`: floor 7.746 → stop rounded away), a stop already beyond the floor keeps its object rule, the SHORT mirror, `CLOSE_BEYOND` composes (the farther wins), no governing scale → no floor, governing without atr → `GeometryError` and `coherence_error` text.
- [x] Step 2: Run them; expected: the floored-stop assertions fail (stop at the object), the atr error does not raise.
- [x] Step 3: Implement `_floor_stop(entry, stop, direction, governing, atr_1m, tick, floor_bars)` and call it in `resolve_geometry` after `_stop`.
- [x] Step 4: Run the geometry tests, then `execution/tests/test_plan_from_state.py`, `execution/tests/test_stack_e2e.py`, `brain/tests/test_reducer.py`, `brain/tests/test_main_brain.py`.

### Task 2: The event calendar module and the calendar build

**Files:**
- Create: `brain/core/event_calendar.py`, `brain/scripts/build_event_calendar.py`, `brain/configs/economic_calendar.ics`
- Test: `brain/tests/test_event_calendar.py`, `brain/tests/test_build_event_calendar.py`

**Interfaces:**
- Produces: `CalendarEvent(uid, summary, at, categories)`, `parse_ics(text) -> tuple[CalendarEvent, ...]`; `EventRule(kind, summary_pattern, sleep_before_minutes, sleep_after_minutes)`; `ScheduledEvent(kind, name, at, start, end)`; `EventFilter(events, sha256)` with `active(known_at)`, `ended_between(previous, known_at)`, `from_config(payload, root)`, `EventFilter.none()`; `build_calendar(bls_subset, bls_schedule, fomc_html) -> str` and `fomc_statement_dates(html) -> list[date]`.

- [x] Step 1: Write the failing tests: a BLS block parses to 13:30Z in January (EST) and 12:30Z in July (EDT); folded lines; a `Z` time; an all-day event skipped; rules match by full regex and build `[at − before, at + after)`; `active` / `ended_between` boundaries (start inclusive, end exclusive; `ended_between` fires once on the first bar at or after `end`); the FOMC parser reads `January 25-26`, `March 15-16*`, `April/May 30-1` (two-day, last day), skips `(unscheduled)`; the built calendar has the 2022 rows at `20221013T083000` US-Eastern and `FOMC Statement` at `20220727T140000`.
- [x] Step 2: Run; expected: `ModuleNotFoundError`.
- [x] Step 3: Implement the module and the script; run the script to write `economic_calendar.ics`.
- [x] Step 4: Run the tests; verify the ics with `parse_ics` counts: 24 events in 2022 + 46 BLS + FOMC 2021–2027.

### Task 3: The controller — schema 5, `EVENT_SLEEP`, the calendar wake

**Files:**
- Modify: `brain/core/sleep_controller.py`, `brain/configs/sleep_controller.json`
- Test: `brain/tests/test_sleep_controller.py`

**Interfaces:**
- Produces: `Decision.EVENT_SLEEP`; `ControllerConfig.events: EventFilter`; `ControllerConfig.from_json(path, *, root=None)`; `decide(events, *, active, config, relation_changes=(), known_at=None, previous_known_at=None)`; reasons `event:<kind>:<ISO>` / `event_ended:<kind>:<ISO>`.

- [x] Step 1: Failing tests: schema 5 loads with three rules; inside the 2022-10-13 CPI window a wake event gives `STAY_ASLEEP` with the event reason and an active episode gets `EVENT_SLEEP`; the first bar at 09:00 New York wakes with `event_ended:CPI:…` and no Eye event; outside, decisions are as before; the sha changes with the calendar bytes.
- [x] Step 2: Run; expected: schema error / `AttributeError`.
- [x] Step 3: Implement.
- [x] Step 4: Run the controller tests and `brain/tests/test_run_guards.py` (the config is loaded there).

### Task 4: The runtime archives on `EVENT_SLEEP`; the machine withdraws every expression

**Files:**
- Modify: `brain/core/runtime.py` (`StepResult.event`, the ACTIVE branch), `execution/core/stack.py`, `execution/core/order_fsm.py`, `execution/core/thesis.py`
- Test: `brain/tests/test_runtime.py`, `execution/tests/test_order_fsm.py`, `execution/tests/test_thesis.py`

**Interfaces:**
- Produces: `StepResult.event: str | None`; `OrderMachine.on_bar(..., event_sleep: str | None = None)`; trade records `event_sleep`, `cancel_requested` (reason `event_sleep`), `position_closed` (`exit_role` `event_sleep`), `flattened`; `EXIT_EVENT = "event_sleep"`; `REPLACEMENT_REASONS` includes `event_sleep`; the thesis closes `event_sleep` without a cooldown.

- [x] Step 1: Failing tests: runtime — an active episode on a bar inside a window archives with a `sleep` record whose reason starts with `event:`, no LLM call, status SLEEP; the next bars stay asleep; the bar after the window wakes a new episode with `event_ended` in its `wake` reasons. Machine — a working entry is cancelled with reason `event_sleep` and its expression refunded; an open position is flattened with `exit_role` `event_sleep`, thesis closed `event_sleep`, `cooldown_bars_left` 0.
- [x] Step 2: Run; expected failures on the missing parameter / records.
- [x] Step 3: Implement.
- [x] Step 4: Run the four test files and `execution/tests/test_stack_e2e.py`.

### Task 5: Prompt, summarizer, docs

**Files:**
- Modify: `brain/configs/prompts/main_brain_system.md`, `brain/scripts/summarize_run.py`, `brain/docs/README.md`, `risk/docs/README.md`, `execution/docs/README.md`
- Test: `brain/tests/test_main_brain.py` (prompt words), `brain/tests/test_summarize_run.py`

- [x] Step 1: Failing tests: the prompt names the floor ("one bar" and "governing scale" in the Expression section); `entry_quality` reports `median_risk_points` and `median_quantity`.
- [x] Step 2: Implement; patch the three READMEs.
- [x] Step 3: Run `brain/tests`, `execution/tests`, `risk/tests`; then the full suite.

### Task 6: Runs and receipts

- [x] Frozen window (`--label stop-floor-events`), the ten benchmark windows (`run_benchmark.py --label stop-floor-events --parallel 3`).
- [x] Summaries against `2710e0b78a707e99` and `benchmark_ids_pools2.txt`; receipts `brain/docs/evidence/2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md` and `…_benchmark_2022.md`; README receipts index.
- [x] Re-freeze `brain/docs/evidence/regression_baselines.json`; research regression test; full suite; memory note.
