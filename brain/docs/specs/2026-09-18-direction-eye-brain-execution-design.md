# Direction: Eye → Brain → Execution — design

Fixes the direction problem diagnosed in
[evidence/2026-09-18_direction_root_cause_2022-01-03.md](../evidence/2026-09-18_direction_root_cause_2022-01-03.md),
in three layers built and validated one after the other. Owner's brief:
first make the Eye reflect the structure that is *forming*, then fix the
Brain's multi-scale direction and bias transitions, then the entry under a
correct thesis; each layer on its own, validated on the frozen backtest
before the next; wake/sleep trimmed without losing coverage of sharp
moves; the smallest structurally correct change at each layer, not a fit to
one day. The session was autonomous and the owner delegated the design
calls, so each decision below is stated with its reason instead of having
passed an approval gate.

## 0. Validation protocol and non-goals

- **Frozen backtest**: the 2022-01-03 session, `--warmup-start 2021-12-27
  --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00`, deepseek-flash at
  high effort, simulated executor, 100 000 USD — the window of
  `7ec17f066d232ba4` and `72ea13c7fcbc1cff`, so every layer is read against
  them. Each layer ends with one such run, summarized with
  `summarize_run.py` and compared; the execution layer's run is frozen into
  `regression_baselines.json`, and a 2022-01-04 run is made at the end as
  the second day the changes were never tuned on.
- **Deterministic checks first**: the Eye is deterministic, so its layer is
  also judged without an LLM by `brain/scripts/audit_scales.py`, which
  prints the per-scale facts the Brain reads at every change point over a
  window (§1.5). Wake/sleep changes are judged by replaying the run's own
  triggers under the new rule (the scratch simulation in §2.3, made a
  repository command in the same script).
- **Cross-run metric**: `direction_accuracy_60m` — the share of state
  revisions whose stated direction (the bias from §2; the opportunity's
  direction for earlier runs) matches the sign of the close 60 minutes later
  minus the close now. It is computable for every run and does not depend on
  fills, so it is the number the three layers are compared on, beside
  calls, coverage, orders and P&L.
- **Non-goals**: no new order types (no stop entries), no change to the
  swing/structure protocol (spans, confirmation, BOS scope), no prices in
  the LLM contract, no retuning of Risk v2 parameters.

## 1. Eye — the facts the Brain reads per scale

Root causes fixed here: the delivery phase named the leg that had *ended*
(the confirmed one), the displacement score had no age, and a broken
protected swing left a scale directionless with a `balance` phase while
price trended outside its range. Nothing about swings, BOS or MSS changes.

### 1.1 The active leg is the forming leg

`TimeframeDeliveryState` (`eyes/core/market_state.py`) gains
`last_leg_direction` (the confirmed leg — today's meaning of
`active_leg_direction`) and `forming_leg_points` (signed: close minus the
confirmed leg's `end_price`; `None` without a leg). `active_leg_direction`
becomes the leg price is in *now*: `LONG` when `forming_leg_points > 0`,
`SHORT` when `< 0`, `None` when there is no leg or the close sits on the
swing. Without any leg the active leg is the internal direction, as today.

`_delivery_phase` keeps its precedence and takes the forming leg as the
active leg: REVERSAL_ATTEMPT (internal ≠ external and the last MSS is
internal) first; EXPANSION when the active leg is the external direction;
RETRACEMENT when it is not and the protected swing is intact; TRANSITION
otherwise. With `external_direction=None` the phase is BALANCE only when an
active dealing range *contains price* (`0 ≤ normalized_location ≤ 1`),
otherwise TRANSITION — a range price has left is not balance.

Both producers implement the rule: `MarketSnapshotPublisher._phase` /
`_timeframe_state` (price and `frame.structural_legs` at hand) and
`reduce_timeframe_state` (the `BAR_COMPLETED` close and the reduced legs,
whose `end_price` comes from the leg event). The projection-equivalence
test (`test_snapshot_projection_replays_to_same_hierarchical_state`) holds
them together.

On 2022-01-03 this reads: 4H `active_leg=long, phase=retracement` from
02:00 NY (was `short, expansion` for 15 h); 1H `active_leg=long,
retracement` at 05:00 (was `short, expansion`); 15m forming leg long from
13:30 while the structure still read internal short.

### 1.2 Displacement direction and age

`TimeframeDeliveryState` gains `displacement_direction` and
`displacement_at`, set with the score on `DISPLACEMENT_OBSERVED` in both
producers; the 5m live path (`_displacement` from `current_metrics`) sets
`displacement_at` to the frame cutoff and the direction from the
observation. The score is not decayed: the age is the fact and the Brain
reads it. The view publishes `displacement_direction` and
`displacement_age_bars = floor((known_at − displacement_at) / scale
minutes)`.

### 1.3 The protection break

`TimeframeStructureState` gains `protection_broken_direction` and
`protection_broken_at`, set where the exact-protection acceptance sets
`external_direction=None` (`market_state.py` ~2607 and the formal-structure
merge ~5643, which preserves them through the directionless stretch) and
cleared when the candidate external structure is not `None`. The view
publishes `structure.reset = {"direction", "bars_ago"}` or `null`. On
2022-01-03 the 1H reads `reset: {direction: long, bars_ago: n}` from 09:39
to the close instead of a bare `external_direction: null`.

### 1.4 The Brain's input (`brain/core/eye_view.py`)

Per scale, `delivery` gains `last_leg_direction`, `forming_leg_atr` (points
divided by that scale's ATR, `frame.metrics["atr"]`; `None` when unknown)
`displacement_direction`, `displacement_age_bars`; `structure` gains
`reset`; the 1m block gains the same delivery keys. `session` gains
`drift_atr` = (close − `session_open`) / `atr_1m`, signed. `LLM_INPUT_SCHEMA_VERSION`
becomes 2. Journals of earlier runs already do not replay under a changed
input; `regression_baselines.json` stays empty until §3's run.

### 1.5 Validation

- Unit tests (synthetic states and bars): phase from the forming leg on
  both producers and their parity; BALANCE only inside the range;
  displacement age and direction; reset fields set, preserved, cleared;
  the view keys and `drift_atr`.
- `brain/scripts/audit_scales.py --emit-start … --end …` (Eye only, no
  LLM): the change points of `external / internal / active_leg / last_leg /
  phase / reset / displacement(direction, age)` per scale beside the close,
  the same way the diagnosis was made by hand. Acceptance on 2022-01-03: the
  three readings in §1.1 and the reset in §1.3; on 2022-01-04 the script
  runs and no scale shows `expansion` against a forming leg of the opposite
  sign for more than one bar of its scale.
- DeepSeek run **E** (Eye only, prompt untouched) on the frozen window,
  compared with `72ea13c7` on `direction_accuracy_60m`, calls, coverage,
  orders, P&L. It measures what the facts alone move.

## 2. Brain — bias, scales, wake/sleep

Root causes fixed here: no explicit bias, the 4H read as governing though
never named so, `external long / internal short` read as a bounce, the tape's
own drift never used, the direction flipped on 5m events eight times in
four hours.

### 2.1 `bias` in the reply and the state

`LLMUpdate` gains a required `bias` object: `direction` ∈ `LONG | SHORT |
NEUTRAL`, `scale` ∈ `4H | 1H | 15m` (`BIAS_SCALES`; the 5m was admitted
until run B, §2.6), `basis` (one sentence naming the live delivery that
sets it). `BrainState` gains `bias` (`Bias` dataclass,
`BRAIN_STATE_SCHEMA_VERSION` 2; `from_dict` of a schema-1 state yields
`NEUTRAL` on `15m` with an empty basis). `prior_state` shows it. No new
journal record: the state stream carries every change.

Reducer rule 4b (`brain/core/reducer.py`): an opportunity whose `direction`
is not the bias direction, or any DEVELOPING/ACTIONABLE opportunity under a
NEUTRAL bias, is dropped with `opportunity_against_bias`; a
`governing_timeframe` above `bias.scale` is dropped with
`opportunity_scale_above_bias`. The Brain must change its bias to change
its side, and the journal records why.

### 2.2 The prompt

A section "Bias — which scale sets the direction", placed before "The
opportunity — a thesis, expressed":

- Live delivery on a scale = its `active_leg_direction` (the leg price is
  in now) with `forming_leg_atr` at or beyond one ATR of that scale, and
  either a displacement in that direction at most three bars old on that
  scale or an MSS/BOS in that direction as its latest structural event. A
  label without that is location, not direction: `expansion` on a scale
  whose active leg points the other way is the previous leg.
- A scale that went live stays live while its active leg keeps its sign; it
  stops being live when that leg ends or a structural event on that scale
  goes against it, not when the excursion shrinks under one ATR (the
  hysteresis of §2.6). The bias is carried forward until then.
- The bias scale is the 15m unless the 1H or 4H delivery is live in its own
  right; then the highest live scale sets it, and the scales above it are
  premium/discount and draw; the 5m never sets it. `structure.reset` in a direction makes that
  side live on that scale until a structure confirms. `external long` with
  `internal short`, the protected low intact and the active leg long is a
  pullback in an uptrend, not a counter-trend bounce. `session.drift_atr`
  and the forming legs are evidence of direction. The bias is NEUTRAL when
  no scale is live and the 15m active leg disagrees with its structure.
- Expression: after a BOS/MSS on the bias scale in the bias direction with
  the forming leg past one ATR, express at the object that `contains_price`
  (a fill now) rather than at a retracement object; a retracement object
  only once the 5m active leg has turned against the bias.
- "The thesis is judged on its governing scale" becomes: the *invalidation*
  is judged on the governing scale; the *bias* on the bias scale; the
  `watch_next` rule stays.

The reply example and the tests that guard prompt wording are updated.

### 2.3 Wake/sleep (`brain/configs/sleep_controller.json` schema 4)

- `relation_change_debounce_bars: 15`: a watched alias's relation flip
  triggers an UPDATE at most once per 15 one-minute bars (the runtime keeps
  the last trigger bar per alias; reset with the episode). On `72ea13c7`
  the 75 relation-only calls had a median gap of 2 minutes between repeats
  of the same alias; the replayed rule removes 44 calls and no coverage
  (79/81 sharp moves, RTH 24/25, both unchanged).
- `delivery_phase_entered` / `delivery_phase_exited` become bookkeeping:
  they are derived labels, verdicted NEUTRAL 68–100 % of the time on every
  scale, and §1.1 publishes the phase in `scales` anyway.
- `level_reached` and `fvg_first_retest` stay reactions (SUPPORT on 16 of
  26 and 7 of 17 on the 15m).

`audit_scales.py --triggers <run-dir>` replays a run's triggers under the
configured rule and prints calls and sharp-move coverage, so the number
above is reproducible.

### 2.4 Metrics (`brain/scripts/summarize_run.py`)

A `bias` section: changes per session, share of ACTIONABLE replies in the
bias direction (100 % by rule 4b), and `direction_accuracy_60m` as defined
in §0 (from `opportunity.direction` when a run predates the bias).

### 2.5 Validation

Unit tests (contract, state round-trip, reducer 4b, prompt words,
controller schema 4 and debounce, summarizer); DeepSeek run **B** (Eye +
Brain + controller) on the frozen window against run E and `72ea13c7`.

### 2.6 Amendment after run B (2026-09-19)

Run B (`07f897f45fab5025`, receipt
[evidence/2026-09-18_brain_bias_day_run_2022-01-03.md](../evidence/2026-09-18_brain_bias_day_run_2022-01-03.md))
raised `direction_accuracy_60m` to 0.445 and restored sleep, and lost
more than either baseline because the bias changed 28 times (median
segment 18 minutes): the "live delivery" rule is a threshold on
`forming_leg_atr` with nothing to hold it, and the 15m forming leg crosses
one 15m ATR as a matter of course; the 5m set the bias 18 times. Two
minimal changes, each general rather than a fit to the day:

- `bias.scale` ∈ `4H | 1H | 15m` (`BIAS_SCALES` in `contract/brain/state.py`;
  the reply parser refuses a 5m bias as malformed).
- The prompt's hysteresis bullet: a scale that went live stays live while
  its active leg keeps its sign; it stops being live when the leg ends or a
  structural event on that scale goes against it, not when the excursion
  shrinks under one ATR; the bias is carried forward until then.

The thresholds (one ATR, three bars), rule 4b, the debounce and the
bookkeeping kinds are unchanged. Run **B′** repeats the frozen window with
only these two changes and is the layer's gate: fewer bias changes and
NEUTRAL revisions, `direction_accuracy_60m` at least held, calls no
higher, coverage unchanged.

Run B′ (`3cc1bb402bb7b964`, receipt §8): bias changes 28 → 14, NEUTRAL
revisions 37 → 21, `direction_accuracy_60m` 0.445 → 0.502, calls 297 →
302, coverage 77 → 80 of 81; the gate is met and the layer is closed.
Realized P&L fell again (−139.5) for two execution reasons that repeat in
every run — the entry at the extreme of the leg that set the bias, and a
position held through the bias flip against it; the second is §3's
bias-reversal exit.

## 3. Execution — the entry under a correct thesis

Root cause fixed here: a right-side thesis died after two retracement
limits that price missed by 1–4 points, because an expiry counted as an
expression.

- **An expiry gives its expression back** (`execution/core/thesis.py`
  `outcome(kind="expired")`: `expressions` decremented, never below zero).
  A cancel by the Brain (plan dropped, signature changed) still counts — it
  is the churn the cap exists for.
- **A marketable limit fills at the open** (`SimulatedExecutor.poll`): a
  BUY limit at or above the bar's open fills at the open, a SELL limit at or
  below it likewise — the fill a limit crossing the market gets, instead of
  the pessimistic limit price. IBKR needs nothing: a limit through the
  market fills at the market.
- **Metric** `missed_trends` in `summarize_run.py`: expired entries where,
  during the TTL window, price travelled at least one R (the plan's stop
  distance) in the thesis direction from where it was at submission
  without touching the limit.
- **A bias reversal against an open position flattens it** (added
  2026-09-19 after run B′). `OrderMachine.on_bar` takes the Brain's
  `bias_direction` (`LONG` / `SHORT`, or `None` when NEUTRAL); a position
  whose plan direction is the opposite side is flattened at market with
  `exit_role = "bias_reversed"` (journal kind `bias_reversed`, the flatten
  path of the close-beyond exit), and `ThesisBook.outcome` closes the
  thesis with `closed_reason = "bias_reversed"` without the stop cooldown.
  A working entry needs nothing: rule 4b drops the opportunity and the
  machine already cancels it as `plan_dropped`. NEUTRAL is not a reversal.
  Evidence: rule 4b refuses a *new* opportunity against the bias while the
  same thesis's open position survives; in runs B and B′ the SHORT held
  through the LONG flip was the worst trade of the day (−32.5 and −58.5;
  an exit at the flip: +19 and +3), and every one of the four runs lost a
  SHORT into the 13:45 breakout.
- Validation: unit tests; DeepSeek run **X** (all layers) on the frozen
  window, frozen as the regression baseline; then 2022-01-04.

## 4. Files

| layer | change |
| --- | --- |
| Eye | `eyes/core/market_state.py` (state fields, `_delivery_phase`, both producers), `contract/eye` if the state types live there, `eyes/tests/test_hierarchical_market_state.py` + a new `eyes/tests/test_forming_leg.py`, `brain/core/eye_view.py`, `contract/brain/llm.py` (schema 2), `brain/tests/test_eye_view.py`, `brain/scripts/audit_scales.py` (new), `eyes/docs` + `brain/docs/README.md` |
| Brain | `contract/brain/llm.py`, `contract/brain/state.py`, `brain/core/reducer.py`, `brain/core/main_brain.py`, `brain/configs/prompts/main_brain_system.md`, `brain/core/sleep_controller.py` + `runtime.py` + `sleep_controller.json`, `brain/scripts/summarize_run.py`, tests, docs |
| Execution | `execution/core/thesis.py`, `execution/core/simulated_executor.py`, `brain/scripts/summarize_run.py`, tests, `execution/docs/README.md`, `regression_baselines.json` |

## 5. Risks

- The forming-leg phase makes every scale's phase move with price; a scale
  with a two-bar swing span will flip its active leg on every counter-swing
  of a bar. That is the true state of a forming leg; the Brain is told the
  size (`forming_leg_atr`) so a one-tick flip reads as nothing.
- Rule 4b can drop a correct opportunity when the model declares its bias
  carelessly; the rejection is journaled and counted, so a run shows it.
- The prompt's "live delivery" thresholds (one ATR, three bars) are
  reading aids, not tuned parameters; they are stated once, in the prompt,
  and the receipt of run B records how often they bound.
- One day validates timeliness, not edge; the 2022-01-04 run is the guard
  against fitting the 3rd.
- The bias-reversal exit turns a bias whipsaw into a realized loss at
  market; with the hysteresis the bias changed 14 times in run B′ and two
  positions were open at an opposing flip. Run X counts `bias_reversed`
  exits beside stops so the trade is visible.
- Run X's first attempt (`09c02ebd39b02f1a`, stopped): from 10:15 NY every
  DeepSeek reply came back empty (the reasoning runs close to
  `max_tokens` on RTH inputs — completion tokens of 25–31k were common
  before), each empty reply left its evidence unjudged, the runtime
  re-offered all of it and the input grew from 29k to 52k characters in
  25 calls with no reply. Fixed 2026-09-19 outside the three layers:
  `max_pending_evidence` (32) bounds the ledger's pending items
  (`evidence_expired`), and the client's incident names the finish
  reason, completion tokens and reasoning size. `max_tokens` itself is
  not raised here.
- Entries are taken at the extreme of the leg that set the bias (receipt
  §8). Not addressed in this design: the expression rule is the Brain's
  and a change to it needs a day other than the 3rd.
