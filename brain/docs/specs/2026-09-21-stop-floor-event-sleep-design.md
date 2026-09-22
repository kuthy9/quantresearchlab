# The stop floor and the event sleep

*2026-09-21. Branch `stop-floor-events` from `main` at `444993b`.*

The entry model (2026-09-20) put the limit at a retracement object and
stopped the chase. The benchmark that validated it showed the next defect:
the stop is sized by the object the LLM names and the contracts by that
stop, so a correct thesis is stopped by noise and a wrong one is carried at
full size — and a resting limit waits through a scheduled release as if it
were any other minute. This build does two things and nothing else: the
stop is never nearer the entry than one bar of the thesis's own scale, and
the Brain sleeps through CPI, NFP and FOMC statements, flat, and wakes fresh.

## 0. The problem, from the journals

Forty-four fills across the three benchmark passes and the two frozen
entry-model runs (`scratchpad/stop_scale.py`):

| measure | value |
| --- | --- |
| stop distance, median (min–max) | 18.5 points (6–75) |
| stop distance in 1m ATRs, quartiles | 0.89 / 1.52 / 2.19 |
| governing scale of the thesis | 15m ×34, 1H ×9, 5m ×1 |
| invalidation object's scale | 5m ×19, 15m ×23, 1H ×2 |
| fills whose adverse excursion reached 1R within 60 minutes | 31 of 44 (70 %) |
| … of which the favourable excursion within 240 minutes reached 2R | 17 |
| adverse excursion in 1m ATRs before a ≥ 2R run, median (75th) | 3.0 (6.7) |

One bar of the governing scale spans about √15 ≈ 3.9 one-minute ATRs on the
15m and √60 ≈ 7.7 on the 1H (the estimate `opportunity_geometry` already
uses for the `CLOSE_BEYOND` buffer). Half the stops sat inside one and a
half *one-minute* bars: 07-27 (`613f5d9e`) was stopped for 1R by a 14.5-point
stop before an 18R run, 09-13 (`53d8562d`) by a 6.25-point stop before a
21R run, 08-26's first pass (`b0d4b54b`) by 15 points before 19R. The gate
sizes on the same distance — three contracts on a 14.5-point stop —, so
10-13 (`d1cce179`) held three contracts into the CPI print through a limit
resting 14.5 points under price one minute before 08:30, and lost 1 055
points on the 350-point bar, a 21 % drawdown halt.

The invalidation-scale rule (reducer rule 4: the governing scale or one
below) does not bound the distance: a 15m FVG's far edge was 6.25 points
from a 5m entry on 09-13. Distance is a property of volatility, not of the
object's scale label.

## 1. Design

### 1.1 The stop floor (`brain/core/opportunity_geometry.py`)

`resolve_geometry` keeps naming the stop from the invalidation object
(`stop.<family>.far_edge` / `.price` / `.close_beyond`) and then applies
one rule:

> The hard stop is never nearer the entry than one bar's range of the
> thesis's governing scale: `STOP_FLOOR_GOVERNING_BARS` (1.0) ×
> `atr_1m` × √(minutes of `governing_timeframe`). A stop the object puts
> nearer is moved to the floor, rounded away from the entry to the tick,
> and the stop rule id becomes `stop.floor.governing_bar`.

- The floor is measured from the *entry*, on the losing side; it composes
  with the `CLOSE_BEYOND` buffer (the farther of the two wins).
- An opportunity with a `governing_timeframe` needs a positive `atr_1m`;
  without one `resolve_geometry` raises `GeometryError` (as `CLOSE_BEYOND`
  already does) and `coherence_error` reports it. An opportunity without a
  governing scale (states journaled before 2026-09-17) gets no floor.
- `reward_risk` is computed from the floored stop, so the gate's
  `min_reward_risk` (2.0) now judges the target against the thesis's
  noise: a target nearer than two governing bars is vetoed `reward_risk`.
  That is the intended consequence — the thesis must aim at a pool on its
  own scale, not the next 5m level.
- The `TradePlan.invalidation_level` (the object's far edge the close-beyond
  exit watches) is unchanged: the falsification is still judged at the
  object, the disaster stop sits at the floor or beyond.

Sizing follows without a new rule: the gate's `risk_points` is the floored
distance, so `by_budget = floor(equity × fraction ÷ (risk_points ×
point_value))` shrinks as volatility grows. On a CPI-day 1m ATR of 18 the
15m floor is 70 points, 1 400 USD per contract, one contract at 1.5 % of
100 000 — and one 1H-governed thesis (140 points) buys none, which the gate
already says as "the contract is too large for this stop". No risk
fraction, leverage or quantity value changes.

### 1.2 The prompt (`brain/configs/prompts/main_brain_system.md`)

Two additions, no new section:

- Expression → the invalidation bullet: the stop never sits nearer the
  entry than one bar of the governing scale (√minutes × the 1m ATR: ≈ 3.9
  ATRs for a 15m thesis, ≈ 7.7 for a 1H one); code needs a reward-to-risk
  of 2 from that distance, so the target is a pool or zone at least two
  governing bars away — name the destination on the governing scale.
- Hard rules → "Geometry must agree with direction": the same sentence in
  one line, so the rule is visible where the geometry rules are listed.

### 1.3 The event calendar (`brain/core/event_calendar.py`, new)

- `parse_ics(text) -> tuple[CalendarEvent, ...]`: RFC 5545 line unfolding,
  `VEVENT` blocks, `DTSTART` with a `TZID` (`US-Eastern` → `America/New_York`;
  any IANA name accepted), a trailing `Z`, or a floating time read in
  `X-WR-TIMEZONE`; all-day events (`VALUE=DATE`) are skipped. Each event:
  `uid`, `summary`, `at` (UTC), `categories`.
- `EventRule(kind, summary_pattern, sleep_before_minutes, sleep_after_minutes)`
  from the controller config; `EventFilter.from_config(payload, root)`
  loads the calendar, matches each event's `summary` against every rule
  (full-match regex) and keeps one `ScheduledEvent(kind, name, at, start,
  end)` per match, sorted by `start`.
- `EventFilter.active(known_at) -> ScheduledEvent | None` — the event whose
  `[start, end)` contains the bar; `ended_between(previous, known_at)` —
  the event whose `end` lies in `(previous, known_at]` (the first bar
  after the window). `sha256` of the calendar bytes.

`brain/configs/economic_calendar.ics` is built by
`brain/scripts/build_event_calendar.py` from committed sources under
`brain/configs/calendar_sources/`:

| source | what it gives | provenance |
| --- | --- | --- |
| `bls_news_release_subset.tsv` | the `Consumer Price Index` and `Employment Situation` events of the BLS calendar, 2025-01 → 2026-12: each event's `UID`, `DTSTART` and `SUMMARY` (every block of the feed has the same other fields, which the build script emits back) | https://www.bls.gov/schedule/news_release/bls.ics fetched 2026-09-21 in a browser (the BLS site refuses scripted downloads) |
| `bls_schedule_2022.txt` | the 2022 CPI and Employment Situation release dates and times, one line each | https://www.bls.gov/schedule/2022/home.htm (the BLS feed reaches back to 2025-01 only) |
| `fomc_calendar.html` | the FOMC meeting dates 2021–2027 | https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm fetched 2026-09-21 |

The script writes one `VCALENDAR` (BLS's `US-Eastern` `VTIMEZONE` block)
with the BLS events in the feed's block shape, the 2022 events in the same shape
(`SUMMARY:Consumer Price Index` / `SUMMARY:Employment Situation`, 08:30
US-Eastern), and one `SUMMARY:FOMC Statement` at 14:00 US-Eastern on the
last day of every scheduled two-day meeting (the Fed's `Month d-d`,
`Month/Month d-d` rows; unscheduled meetings are not statements at 14:00
and are skipped). `DESCRIPTION` carries the source URL. The Employment
Situation *of Veterans* release is not the jobs report and does not match
the rule's pattern.

### 1.4 The controller (`brain/core/sleep_controller.py`, `brain/configs/sleep_controller.json`)

Schema 5 adds:

```json
"events": {
  "calendar": "brain/configs/economic_calendar.ics",
  "rules": [
    {"kind": "CPI",  "summary_pattern": "Consumer Price Index", "sleep_before_minutes": 60, "sleep_after_minutes": 30},
    {"kind": "NFP",  "summary_pattern": "Employment Situation",  "sleep_before_minutes": 60, "sleep_after_minutes": 30},
    {"kind": "FOMC", "summary_pattern": "FOMC Statement",        "sleep_before_minutes": 60, "sleep_after_minutes": 90}
  ]
}
```

`ControllerConfig.events` is the `EventFilter`; `ControllerConfig.sha256`
covers the config bytes *and* the calendar bytes (a run's identity changes
with the calendar). `Decision` gains `EVENT_SLEEP`. `decide` takes
`known_at` and `previous_known_at`:

| runtime status | condition | decision, reasons |
| --- | --- | --- |
| any | `events.active(known_at)` | asleep: `STAY_ASLEEP`; active: `EVENT_SLEEP` — reason `event:<kind>:<release ISO>` |
| SLEEP | `events.ended_between(previous, known_at)` | `WAKE`, reason `event_ended:<kind>:<release ISO>` (no Eye event needed) |
| otherwise | as before | |

`known_at=None` (older callers, tests) skips the calendar.

### 1.5 The runtime and the machine (`brain/core/runtime.py`, `execution/core/stack.py`, `execution/core/order_fsm.py`)

- ACTIVE and `EVENT_SLEEP`: the runtime archives the episode with reason
  `event:<kind>:<release ISO>` (the journal's `sleep` record) without an
  LLM call, and `StepResult(decision=EVENT_SLEEP, status_after=SLEEP,
  slept=True, event=<reason>)`. The next bars are `STAY_ASLEEP` until the
  window ends, then `WAKE` on `event_ended` — a new episode, `prior=None`:
  nothing of the pre-event reading, opportunity or thesis book survives.
- The stack passes `event_sleep=result.event` to
  `OrderMachine.on_bar`. On that bar the machine withdraws every
  expression: the working entry is cancelled with reason `event_sleep`
  (the thesis book refunds it — the reason joins `REPLACEMENT_REASONS`) and
  every open position is flattened at market (`event_sleep` record, then
  `position_closed` with `exit_role` `event_sleep`, `flattened`); the
  thesis closes `event_sleep` without a cooldown. The bias-reversal check
  is skipped while the Brain is asleep (no bias). The sleep invariant —
  the Brain is flat when it sleeps — therefore holds for the event sleep
  as for every other sleep, and the wake input needs nothing new.
- `STAT_KINDS` gains `event_sleep`; `EXIT_EVENT = "event_sleep"`.

Why flatten rather than hold: the user's rule is that nothing of the
pre-event expression continues; a sleeping Brain cannot manage a position
(rule 5 forbids sleeping with one open), and 10-13 is the class of loss a
held position takes through a print. Holding through the release with the
bracket at the broker is a policy the user can choose later; it needs the
wake input to carry the execution view (a schema change) and is out of
scope here.

### 1.6 Metrics (`brain/scripts/summarize_run.py`)

`orders.entry_quality` gains `median_risk_points` (the plan's stop distance
at submission) and `median_quantity`. `sleeps_by_reason` already counts
`event:<kind>:…`; `orders.counts` already counts every trade kind, so
`event_sleep` appears there.

### 1.7 Docs

`brain/docs/README.md` (modules, controller table and schema 5, running —
the calendar build, tests), `risk/docs/README.md` (the sized distance is
the floored one), `execution/docs/README.md` (the `event_sleep` exit).

## 2. Validation

1. Unit tests first (TDD per task): geometry floor, calendar parsing and
   windows, controller decisions, runtime archive on `EVENT_SLEEP`, machine
   flatten and refund, summarizer fields, config schema 5.
2. Full suite green.
3. The frozen window `2021-12-27 → 2022-01-02T18:00–2022-01-03T17:00`
   (no release in it: the floor alone) against `2710e0b78a707e99` — the
   old defects (fills at the market, incoherent LONGs, expiry blocks) must
   not return; stop distances in ATRs are the new reading.
4. The ten benchmark windows (`run_benchmark.py --label stop-floor-events`)
   against the third pass (`benchmark_ids_pools2.txt`). Expected by design:
   09-13, 10-13 and 11-10 start asleep until 09:00 (CPI window
   07:30–09:00), 07-27 sleeps 13:00–15:30 (statement 14:00). Read: stop
   distance in ATRs, fills stopped inside 60 minutes, the right-direction
   fills that reach 2R, `reward_risk` vetoes, contracts per fill, and
   whether any release-bar fill remains.
5. Re-freeze `regression_baselines.json` on the new journals (the
   controller sha and the geometry changed; the old journals cannot replay
   under the event filter).

**Outcome (2026-09-21):** full suite 1543 passed; frozen window
`f4f6998a6ed59eb8` −77.25 (was −141.5), nine fills at the limit, stops ≥ 3.9
ATRs, one stop-out, no daily stop, seven bias-reversal exits; benchmark
(ids in the receipt) −286.5 (was −1327.25 / −272 without the halted
10-13), six fills all stopped and none on a trend day, `position_size`
vetoes 18 → 28, the model lowering the governing scale to 5m in 21
proposals, targets still the next 5m pool. Receipts:
[evidence/2026-09-21_stop_floor_event_sleep_benchmark_2022.md](../evidence/2026-09-21_stop_floor_event_sleep_benchmark_2022.md),
[evidence/2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md](../evidence/2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md).
Eleven baselines re-frozen.

## 3. Risks

- Fewer trades: with the floor most targets that were 2–8 R of a 15-point
  stop are inside two governing bars and are vetoed `reward_risk`. The LLM
  is told to aim at the governing scale; if it does not, the benchmark
  shows the veto count and no fills — a true reading, not a regression.
- A 1H-governed thesis on a volatile day buys no contract at 1.5 % of
  100 000 with NQ's 20 USD point: `POSITION_SIZE` vetoes will rise. That is
  the account's statement, not the strategy's; MNQ or a larger account is a
  config decision for the user.
- The 07-27 window is mostly inside its FOMC sleep and the three CPI
  windows lose their first 45 minutes: those windows now measure the
  post-release trade, which is the intended behaviour.
- The calendar is static: a run past 2026-12 or on a rescheduled release
  (a government shutdown moved the 2025-10 CPI to 10-24) needs the file
  rebuilt from the BLS feed; the build script is the procedure.
