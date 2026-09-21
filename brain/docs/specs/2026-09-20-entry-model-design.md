# Entry model — the bias picks the side, the pullback picks the price

*2026-09-20. Follows the direction fix
([2026-09-18-direction-eye-brain-execution-design.md](2026-09-18-direction-eye-brain-execution-design.md))
and its close ([evidence/2026-09-18_execution_entry_day_run_2022-01-03.md](../evidence/2026-09-18_execution_entry_day_run_2022-01-03.md) §6).*

## 0. The problem, from the journals

The direction fix raised `direction_accuracy_60m` from 0.359 to 0.465–0.502
and lost more with every layer. The receipt named two things it did not
touch: the entries are taken at the extreme of the leg that set the bias,
and `opportunity_incoherent` went from 0–1 to 8–22 rejections per run.
Reading the journals of runs X (`ab572b09d59bc4d7`), B′ (`3cc1bb402bb7b964`)
and the 2022-01-04 guard (`5cfe65bcc480a032`):

- **Every fill came within 1–9 minutes of its submission** (21 fills, median
  1 minute). The Brain marks the opportunity ACTIONABLE when the bias
  confirms and names the object price is *in* (`contains_price`), because
  the prompt told it to: "express at the object that `contains_price` — the
  order fills now". The confirmation point of a leg is its extreme, so the
  entry is the extreme: 07:09 LONG at 16438.75 four points under the
  overnight high, 09:47 LONG at 16413.5 six minutes before the flush. The
  bias and the entry are the same event.
- **The incoherent LONGs are a geometry artefact, not a model error.**
  Every one of the 8–22 rejections reads `LONG requires stop < entry <
  target` (or the SHORT mirror) with one of two entries: `16504.5` — the
  *upper* edge of the 1H FVG that contained price (`entry.zone.near_edge`
  resolves a LONG to the zone's upper edge whether or not price is inside
  it; at 13:45 that was 38 points above the market and above the 5m
  target's lower edge) — or `16374.13` / `16431.38` — the *value price* of a
  dealing range named as the entry (`entry.range.value`), which lies
  wherever the range's volume profile puts it, in these cases below the
  invalidation. Both are code choosing a price the Brain did not mean; the
  afternoon LONG thesis of 2022-01-03 was lost to the first.
- **A limit at a pullback object rarely gets its chance.** `order_ttl_bars`
  is 15 one-minute bars; a 5m pullback into a zone takes longer more often
  than not, an expiry blocks the same three objects for the episode, and a
  replaced entry object counts against the thesis's two expressions. The
  rules were built for a Brain that filled at the market; they punish one
  that waits.
- **The Brain is never told why an opportunity died.** The reducer's
  rejections are journaled and not shown; the next call sees
  `opportunity: NONE` and names the same objects (B′ 05:35, 05:36, 05:40:
  three identical rejections; 06:21 and 07:00 the same 16504.5 entry).

## 1. Design

The bias decides which side to trade and fires nothing. The entry is a
*resting limit at a retracement object* chosen on the scale below the bias
scale, with the invalidation the thesis's own and the target the next
liquidity. Code makes the chase impossible and the wait affordable; the
prompt tells the Brain what a good location is. Nothing is fitted to a day:
no threshold is added, and the one number that changes meaning
(`order_ttl_bars`) keeps its value.

### 1.1 Geometry (`brain/core/opportunity_geometry.py`)

- **No entry the market is already past.** `coherence_error` (the
  reducer's proposal-time check) refuses a LONG whose resolved entry lies
  above the bar's close and a SHORT whose entry lies below it:
  `opportunity_incoherent:LONG entry 16504.5 lies above price 16466.0 — a
  trade is expressed on a pullback, not at the market`. `resolve_geometry`
  itself does not judge the side: the plan is re-resolved on every bar
  while an order works, and a limit the tape has crossed must fill, not
  be cancelled.
- **Inside a zone, the midpoint.** When the entry zone contains price, a
  LONG's limit is the zone's midpoint if that is at or below the close,
  else the zone's lower edge (SHORT: midpoint if at or above the close,
  else the upper edge) — rule ids `entry.zone.inside_midpoint` /
  `entry.zone.inside_far_edge`. A zone below price (LONG) keeps
  `entry.zone.near_edge`.
- **A range is not an entry.** `resolve_geometry` refuses `range` as the
  entry object (`a range is not an entry object; name the zone, pool or
  swing inside it`). Ranges stay valid as invalidation and target.
- **The machine refuses a marketable submission.** `OrderMachine.on_bar`
  checks the same side rule against the plan's `close` before `admit`
  (a plan can reach submission on a later bar than its proposal: after a
  cooldown, after a refusal) and journals `thesis_refused` with reason
  `entry_marketable` once per thesis until the Brain's next call.

### 1.2 Feedback (`contract/brain/state.py`, `brain/core/reducer.py`)

`LastUpdate.rejections: tuple[str, ...]` (default empty; journals before
it read back as empty) carries the reducer's rejections of the update
that produced the state. `_prior_view` shows it as
`prior_state.last_update.rejections`. `LLM_INPUT_SCHEMA_VERSION` 2 → 3.

### 1.3 The wait (`risk/configs/risk.json`, `execution/core/order_fsm.py`,
`execution/core/thesis.py`)

- `order_ttl_bars` (15) counts **bars of the entry object's own scale**:
  a 5m object's limit works 75 one-minute bars, a 15m object's 225, a 1H
  object's 900. `risk.json` `schema_version` 2 → 3 for the change of
  meaning; `execution_view.order.ttl_bars` reports the 1m-bar figure the
  machine counts against. The Brain stays awake while an order works and
  is called on every reaction, so a long wait is supervised: a dropped or
  changed opportunity cancels it, and the Eye retiring the object cancels
  it.
- **A replacement is not an expression.** A cancel with reason
  `signature_changed` (the Brain moved the entry to another object) or
  `entry_object_not_visible` (the Eye retired it) gives the thesis its
  expression back like an expiry does; `plan_dropped` keeps counting.

### 1.4 The prompt (`brain/configs/prompts/main_brain_system.md`)

The "Expression" paragraph is replaced by a section whose rules are:

- the bias fires nothing; the trade is expressed on the scale below the
  bias scale at a retracement object — the zone the last displacement left
  behind, or the pool price will sweep on the way; code refuses an entry
  above price for a LONG / below it for a SHORT, and a range;
- **nearest first**: the shallowest valid object, the newest zone of the
  entry-scale displacement, not the leg's origin; the deep object is for a
  leg that has already turned;
- **follow the leg**: when the tape runs and prints a new zone, move the
  entry to it — a replacement costs the thesis nothing;
- the invalidation and the target are the thesis's; a nearer entry or a
  nearer invalidation earns the reward-to-risk, a farther target does not;
- the order waits `ttl_bars` (fifteen bars of the object's scale); an
  `expired` order means the pullback never came — name what the tape
  offers now;
- `prior_state.last_update.rejections` says what code refused and why;
  read it before naming the same objects again.

The geometry hard rule is rewritten to match (inside a zone → midpoint;
above price → refused; range → refused; the reason is shown).

### 1.5 Metrics (`brain/scripts/summarize_run.py`)

- `orders.entry_quality` (needs bars): per fill, its location in the
  range of the preceding 60 and 240 one-minute bars measured in the trade
  direction (0 = the best price of the window, 1 = the worst), the wait
  from submission in minutes, the maximum favourable and adverse
  excursions over the next 60 minutes in R (the plan's stop distance),
  and whether the close 60 minutes later was on the trade's side.
  Aggregates: `fills`, `median_location_60m`, `median_location_240m`,
  `chased` (location_240m ≥ 0.8), `median_wait_minutes`, `median_mfe_r`,
  `median_mae_r`, `right_60m`, `fill_rate` (fills / submissions).
- `brain.actionable_direction_accuracy_60m`: `direction_accuracy` over the
  ACTIONABLE replies' direction at their call time (the receipt computed
  it by hand until now).

### 1.6 The benchmark (`brain/configs/benchmark_windows.json`,
`brain/scripts/run_benchmark.py`)

Ten windows across regimes (market time), each run with the Eye warmed
`warmup_days` (7) before the window:

| window (ET) | regime |
| --- | --- |
| 2022-01-24 12:00–16:00 | extreme reversal |
| 2022-02-24 08:30–12:30 | V reversal |
| 2022-05-10 09:30–13:00 | two-sided chop |
| 2022-06-27 09:30–13:00 | ordinary low conviction |
| 2022-07-27 13:30–16:00 | FOMC, strong long |
| 2022-08-26 09:45–13:00 | strong short trend |
| 2022-09-13 08:15–11:30 | CPI, one-way down |
| 2022-10-13 08:15–12:30 | CPI, huge reversal |
| 2022-10-19 09:30–12:30 | mixed, unclean trend |
| 2022-11-10 08:15–11:30 | CPI, strong long trend |

`run_benchmark.py` reads the file, builds one `run_llm_brain` command per
window (`--dry-run` prints them), runs them `--parallel` at a time and
writes each log beside the journals. The verdict per window reads
`entry_quality`, `actionable_direction_accuracy_60m`, `rejections_by_kind`,
`missed_trends`, the closed trades and the realized points.

### 1.7 Every pool is placed (amendment after the first benchmark, 2026-09-20)

The first benchmark pass (runs labelled `entry-model`) left twelve
`opportunity_incoherent` rejections across nine runs; seven named a target
or an invalidation the model could not see the side of — an object
outside the 4-ATR `price_relations` window, most often a sell-side pool
price had already fallen through on a one-way day (09-13, 08-26). A pool
is where a trade goes, so `_relations_view` now places every liquidity
pool whatever its distance; zones, swings and ranges keep the
`relation_atr_limit`. Median pools per input: 67, of which 19 rows were
placed before. The frozen window and the ten benchmark windows are run
again under the label `entry-model-pools`; the `entry-model` runs stay as
the paired "before".

## 2. Validation

1. Unit tests, red first: geometry (inside-zone midpoint, range refused,
   the side rule in `coherence_error` and not in `resolve_geometry`),
   reducer (`last_update.rejections`), state round-trip, prompt words,
   FSM (scale TTL, `entry_marketable`, replacement refund), thesis book,
   summarizer (`entry_quality`, actionable accuracy), benchmark runner
   (dry run, warmup arithmetic), risk config schema.
2. The frozen window 2022-01-02 18:00 → 2022-01-03 17:00 with DeepSeek,
   against run X: no fill at the market (median wait well above one
   minute), no `opportunity_incoherent` from a containing zone or a range,
   the afternoon LONG thesis expressed, `direction_accuracy_60m` held.
3. The ten benchmark windows; the receipt reads the entry-quality
   aggregates across regimes and says where the model still loses.
4. `regression_baselines.json` re-frozen on the new frozen-window run
   (the input schema changed; the old baselines cannot replay).
   *Outcome (2026-09-21):* the second pass under §1.7 was cut by HTTP 402
   (the API balance) and repeated as a third pass (`entry-model-pools-2`);
   the baselines are its ten windows and its frozen 2022-01-03 run, all
   replaying — the halted 2022-10-13 window after the replay learned to
   stop at the halt as the runner does (receipt
   [evidence/2026-09-20_entry_model_benchmark_2022.md](../evidence/2026-09-20_entry_model_benchmark_2022.md) §3, §5;
   [evidence/2026-09-20_entry_model_frozen_window_2022-01-03.md](../evidence/2026-09-20_entry_model_frozen_window_2022-01-03.md) §7).

## 3. Risks

- A resting limit fills less often: `fill_rate` and `missed_trends` will
  rise where the tape runs without a pullback. That is the model's
  choice by design; the benchmark's trend windows (08-26, 09-13, 11-10)
  measure the price of it.
- A 1H entry object waits 900 bars; the Brain is awake for all of them.
  The prompt steers the entry to the scale below the bias.
- The midpoint rule fills inside a zone that price then leaves upward
  (a LONG at the midpoint of a zone on the way up is still a pullback of
  half a zone). No threshold is added to prevent it.
- `last_update.rejections` grows the input by a few hundred characters
  on the calls after a rejection; the pending-evidence bound still holds.
- The benchmark's paired comparison is first pass against third (the
  pool fix in between); the frozen window pairs run X, the first pass and
  the third.
- Found by the benchmark, not fixed here: the invalidation is a 5m object
  a few points away on days whose one-minute bar is tens or hundreds of
  points, and the position is sized to it (2022-10-13: a LONG limit one
  minute before the CPI print, a 14.5-point stop, three contracts, the
  bias-reversal flatten at the next open 352 points lower, a 21 %
  drawdown halt); a resting order has no notion of a scheduled release;
  the model repeats a wrong-side invalidation despite reading its
  rejection; the reasoning budget on CPI inputs.
