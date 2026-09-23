# The thesis scale, the structural exit and the bias decay

*2026-09-22. Branch `scale-exit-bias` from `stop-floor-events` at `8d4f3f7`.*

The stop floor (2026-09-21) sized the stop by the thesis's scale and the
contracts by that stop. Its benchmark showed what the floor exposed: the
contract is too large for the account at that stop, the model lowers the
scale to fit the stop, the target stays the next 5m pool, the bias flip
closes seven trades in eight, and a live 4H bias pins a reversal day. This
build makes five changes, each a rule in code, and no entry-timing change:

1. the risk budget is derived from the floored stop distribution so one NQ
   contract is fundable, and the daily stop, the drawdown halt and the open
   risk are validated against it;
2. the thesis scale is set by code from the bias, never by the reply;
3. the target lies on the thesis scale or above it (the invalidation rule
   already binds it to the thesis scale or one below; the stop is floored
   to it);
4. a position leaves on its own stop, its close-beyond exit or a structural
   reversal on its thesis scale — never on a bias flip;
5. a bias decays to NEUTRAL when the scales below it deliver against it,
   and a decayed bias is re-set by structure on its own scale only.

## 0. The problem, from the journals

Eleven journals of the stop-floor pass (`benchmark_ids_stop_floor.txt`,
`f4f6998a6ed59eb8`; `scratchpad/design_evidence.py`, `decay_sim.py`).

**Sizing.** 62 proposals reached the gate. The 28 that passed had a median
stop of 48 points (max 71); the 19 refused `position_size` a median of
129 points (2 580 USD a contract, 75th percentile 136). Over all 62 the
stop's 50th / 75th / 90th percentiles are 69 / 117 / 169 points — 1 378 /
2 332 / 3 385 USD a contract. At 1.5 % of 100 000 USD one contract needs
a stop under 75 points; the trend days ran 1m ATRs of 25–40, one 15m bar
being 3.9 of them.

**Scale.** 314 ACTIONABLE revisions. Under a 15m bias the thesis was
always 15m (180 of 180). Under a 1H bias: 15m ×64, 5m ×18. Under a 4H
bias: 15m ×32, 5m ×17. The 5m governing scale appears only under a bias
two or three scales above it — the model reached for it when the floor of
the 15m did not fit the budget (0 such proposals before the floor). The
target object was a 15m one in 250 revisions, a 1H one in 54, a 5m one in
10; the invalidation a 5m object under a 4H bias in 21.

**Exits.** The frozen window closed seven of eight trades by
`bias_reversed` at an average of −4 points, 5 to 90 minutes after the fill,
on a day with 17 bias flips; the afternoon LONG that was right an hour
later (16 386.75, target 16 512) left at +18.25 on a flip.

**Direction.** The bias timelines against the tape:

| window | bias | price after | the Eye's evidence against it | what the model did |
| --- | --- | --- | --- | --- |
| 01-24 | SHORT@4H from 12:00, never changed | low 13 709 at 12:17, high 14 502 at 15:58 | 15m long displacements 13:00, 13:15, 13:45, 14:00, 14:30, 15:00, 15:15; 15m MSS long 15:30; 1H long displacement 16:00 | every one verdicted CONTRADICT; the bias stood — the 4H leg was still short |
| 02-24 | SHORT@4H from 08:44, never changed | low 13 035 at 08:26, high 13 556 at 11:19 | 15m long displacement and MSS 09:45; 1H long displacement 10:00; 1H structure long 11:00 | CONTRADICT ×3, then SUPPORT; the bias stood |
| 10-13 | SHORT@15m 09:00 → SHORT@4H 10:00 (a 4H BOS short printed near the low) → LONG@15m 11:18 | low 10 496 at 09:32, high 11 106 at 11:40 | 15m long displacement 09:45; 1H long displacement 11:00 | flipped 105 minutes after the low, 230 points under the high |
| 05-10 | LONG@15m 09:30, NEUTRAL 10:48, SHORT@15m 11:04, SHORT@1H 11:15, SHORT@15m 11:40, SHORT@1H 12:00 | high 12 534 at 09:30, low 12 178 at 11:25 | — | each bias set after the move it names |

The prompt's rule — the highest live scale sets the bias and stays live
until its own leg ends or its own structural event turns — has no
downgrade: a 4H leg does not end inside a session and a 4H event does not
come, so the scales below it are verdicted and discarded by design. The
Eye was not late: its 15m counter-displacements came 40–80 minutes after
the lows.

**A decay rule, simulated on the eleven journals.** Structural Eye events
(`mss_core_confirmed`, `qualified_bos`, `displacement_observed`) on the
bias scale and the bias scales below it — plus the 5m under a 15m bias —
counted from the moment the model set that bias; a same-direction event
resets the count, an opposite MSS / BOS on the bias scale ends it at once,
two opposite events end it:

| window | model's bias | code would set NEUTRAL | price then (day's extreme) |
| --- | --- | --- | --- |
| 01-24 reversal | SHORT@4H 12:00 | 13:15 (15m displacement) | 14 054 (high 14 502) |
| 02-24 reversal | SHORT@4H 08:44 | 10:00 (15m displacement) | 13 312 (high 13 556) |
| 10-13 reversal | SHORT@15m 09:00 / SHORT@4H 10:00 / LONG@15m 11:18 | 09:45 / 11:00 / 12:15 | 10 527 (low 10 496) / 10 650 / 10 968 (after the 11:40 high) |
| 05-10 chop | LONG@15m 09:30 / SHORT@15m 11:04 | 09:35 / 11:55 | 12 452 (high 12 534) / 12 238 (low 12 178) |
| 06-27 low conviction | LONG@4H 09:30 | 09:45 | 12 056 |
| 10-19 unclean | SHORT@1H 09:31 / SHORT@15m 11:00 | 10:30 / 11:25 | 11 268 / 11 186 |
| 07-27, 08-26, 09-13, 11-10 trend days | — | never | — |
| frozen 01-03 (a chop night, 19 model flips) | — | 11 times | — |

Counting only the scale one below the bias frees nothing on 01-24 (the 1H
turned at 16:00); counting the 5m under every bias ends the 09-13 and
11-10 trend biases on 5m pullbacks. Three events instead of two leave the
05-10 chop and 10-19 untouched and free 01-24 half an hour later.

## 1. Design

### 1.1 The risk budget (`risk/configs/risk.json`, `risk/core/gate.py`, `execution/core/order_fsm.py`)

Schema 4. The per-trade fraction is derived from the floored stop
distribution, not chosen: the 75th percentile stop (117 points, 2 332 USD)
over the sim account's 100 000 USD, rounded up to the half percent.

| key | was | now | why |
| --- | --- | --- | --- |
| `risk_fraction.BASE` | 0.015 | 0.025 | one contract at the 75th-percentile floored stop |
| `risk_fraction.A_PLUS` | 0.02 | 0.03 | the same step above BASE as before |
| `daily_loss_fraction` | 0.025 | 0.05 | two full BASE losses end the session |
| `max_drawdown_fraction` | 0.065 | 0.10 | two daily stops halt the run |
| `max_open_positions` | 3 | 2 | the open risk at BASE never exceeds the day's budget |

`RiskConfig.from_json` refuses a config whose
`max_open_positions × risk_fraction["BASE"]` exceeds `daily_loss_fraction`
or whose `2 × daily_loss_fraction` exceeds `max_drawdown_fraction` — the
three limits are one policy. `RiskGate.assess` takes `open_risk` (the sum
of the open positions' `risk_amount`, from the machine's intents) and
vetoes `ACCOUNT_RISK` when `open_risk + quantity × per_contract` exceeds
the session's remaining budget: `session_open_equity ×
daily_loss_fraction − max(0, session_open_equity − equity)`. The quantity
is the one the other caps gave; the check does not shrink it (a smaller
trade is not the thesis's trade). Nothing else in the gate changes; the
leverage cap (8×), the margin cap and `max_quantity` stay.

### 1.2 The thesis scale (`contract/brain/state.py`, `contract/brain/llm.py`, `brain/core/reducer.py`)

`THESIS_SCALE_OF_BIAS = {"4H": "1H", "1H": "15m", "15m": "15m"}`: the
thesis rests one scale below the bias scale, never below the 15m — the
scale the prompt already names as the expression scale, and the scale the
model used in 276 of 314 revisions. The reply's `opportunity` loses
`governing_timeframe` (contract keys, example, parser); the reducer sets
`opportunity.governing_timeframe` from the effective bias (§1.4) before
rule 4's scale checks and the coherence check, so the invalidation rule
(thesis scale or one below), the stop floor and the target rule all key on
it. `opportunity_scale_above_bias` cannot occur and is retired. The state's
`Opportunity.governing_timeframe` and the plan's are unchanged.

Under a 4H bias the thesis is a 1H one: a 7.7-ATR floor, a 1H or 15m
invalidation, a 1H destination. On a 30-ATR day that is 230 points — 4 600
USD a contract, refused by the budget. That is the honest reading of a 4H
bias at this account, not a defect; §1.4 makes the 4H bias short-lived
when the scales below it turn.

### 1.3 The target scale (`brain/core/reducer.py`)

Rule 4c: the target object's scale is at or above the thesis scale
(`scale_gap(governing, target_scale) ≤ 0`), else
`opportunity_target_scale:<alias>` and the opportunity drops. A 5m pool is
never the destination of a 15m thesis; a 15m pool never that of a 1H one.
Objects of unknown scale are not judged (as for the invalidation rule).

### 1.4 The bias decay (`contract/brain/state.py`, `brain/core/reducer.py`)

`Bias` gains two reducer-owned fields — `since` (when this direction and
scale were set) and `decayed` (the `DIRECTION@scale` code ended, or
`None`) — and the state schema becomes 3 (schema 2 journals read back with
both unset). The reply's `bias` keeps `direction`, `scale`, `basis`.

Rule 4d, applied after the evidence bookkeeping (rule 2) and before the
opportunity rules, computes the *effective bias* the state carries:

1. **Continuity.** The reply's pair (direction, scale) equal to the prior
   state's keeps its `since`; a new pair takes `known_at`. A NEUTRAL reply
   keeps the prior NEUTRAL's `since` and `decayed`; a NEUTRAL reply after
   a directional bias starts fresh (`decayed = None`).
2. **Re-assertion.** A reply asserting the pair in the prior state's
   `decayed` is refused unless the ledger holds an MSS or BOS on that
   scale in that direction with `known_at ≥ prior since` (the decay time):
   the state keeps the prior NEUTRAL bias and records
   `bias_reassert_refused:<DIRECTION@scale>`. Structure, not a candle,
   re-sets a bias code ended.
3. **Decay.** For a LONG / SHORT bias on scale S, walk the ledger's
   structural items (`mss_core_confirmed`, `qualified_bos`,
   `displacement_observed`, with a direction) whose scale is in
   `BIAS_DECAY_SCALES[S]` — `4H: (4H, 1H, 15m)`, `1H: (1H, 15m)`,
   `15m: (15m, 5m)` — and whose `known_at ≥ since`, in time order: an
   item in the bias direction resets the count to 0; an MSS / BOS on S
   against it ends the bias; any other item against it counts one, and
   `BIAS_DECAY_EVENTS` (2) end it. Ending means: the state's bias becomes
   `NEUTRAL` on S with `basis` `"code: <n> structural events against
   <DIRECTION> on <scales> since <time>"`, `since = known_at`,
   `decayed = "<DIRECTION@S>"`, and `bias_decayed:<DIRECTION@S>:<n>` is
   recorded in the rejections. Items the model verdicted NEUTRAL or has
   not verdicted count like the rest — the rule reads the Eye, not the
   verdicts.

Rules 4 and 4b then judge the opportunity against the effective bias: a
decayed bias drops the opportunity (`opportunity_against_bias:NEUTRAL`),
and the thesis scale (§1.2) follows the effective bias's scale. An open
position is untouched (§1.5). The rejections travel to the next call in
`prior_state.last_update.rejections`; `prior_state.bias` shows `since` and
`decayed`.

### 1.5 The structural exit (`execution/core/stack.py`, `execution/core/order_fsm.py`, `execution/core/thesis.py`)

The stack no longer passes the bias to the machine. It passes
`structure_events`: the `(timeframe, direction)` pairs of this bar's
`mss_core_confirmed` and `qualified_bos` events (directions upper-cased).
The machine's `_structure_reversed` flattens every position whose plan's
`governing_timeframe` printed one against its direction: a
`structure_reversed` record (signature, thesis, timeframe, event kind,
position), then the flatten and `position_closed` with `exit_role`
`structure_reversed`; the thesis closes `structure_reversed`, no cooldown
(a reversal is a new reading, not a loss to sit out). `EXIT_BIAS`,
`_bias_reversed`, the `bias_reversed` record and the `bias_direction`
parameter are removed; `STAT_KINDS` swaps `bias_reversed` for
`structure_reversed`. The stop, the target and the close-beyond exit are
the thesis's own invalidation and stay as they are.

### 1.6 The prompt (`brain/configs/prompts/main_brain_system.md`)

- *Bias*: the decay rule in the model's terms (which scales count, two
  events, a same-direction event resets, an MSS / BOS on the bias scale
  ends it at once); a decayed bias comes back only on an MSS / BOS on its
  own scale; after a decay re-read from the 15m up; in a balance — the
  bias scale's legs alternating inside the dealing range, the scale
  below delivering both ways — NEUTRAL and no trade, because the live
  rule in a range is a late reading of the last swing.
- *Expression / The opportunity*: the thesis scale is set by code from the
  bias (one below it, never below the 15m); the invalidation lies on it or
  one below, the target on it or above, the stop floor is one of its bars;
  `governing_timeframe` leaves the reply.
- *Execution feedback*: no bias-reversal exit; `structure_reversed` as an
  outcome and a `closed_reason`; a bias change while a position is open
  changes nothing at the broker — the position leaves on its stop, its
  close-beyond exit or a reversal on its thesis scale.
- *Output contract*: the example without `governing_timeframe`.

### 1.7 Metrics (`brain/scripts/summarize_run.py`)

Order counts include `structure_reversed` (not `bias_reversed`); the bias
block adds `decays` and `reasserts_refused` from the rejections.

### 1.8 Docs

`risk/docs/README.md` (config table, the validation, the new veto),
`execution/docs/README.md` (machine and thesis rows, the exits table, the
tests row), `brain/docs/README.md` (reducer row, state schema 3, the
contract, the prompt paragraph, the receipts index).

### 1.9 The second pass (2026-09-22, after the first benchmark and a review)

The first benchmark of this build (−718 points, eleven fills) showed the
decay memory dying at the episode boundary — a wake after an archive let
the model re-assert the decayed pair with no structure (01-24's short at
14:01, 05-10's second long) — and an adversarial review of the diff
confirmed four more defects. All are in the tree with tests:

- **The memory crosses the archive.** The runtime hands the archived
  bias to the wake (`ReduceContext.carried_bias`); the reducer uses it
  as the prior bias on revision 0 — also when the wake call is an
  incident (`empty_state(bias=…)`).
- **The memory survives a detour.** `Bias.decayed` / `decayed_at` travel
  through every later reply, NEUTRAL or another pair, until the decayed
  scale prints an MSS / BOS in that direction; a re-assertion after a
  detour lands on NEUTRAL (`bias_reassert_refused`) rather than on the
  detour's pair.
- **`since` is the direction's age.** A scale change in the same
  direction (SHORT@1H → SHORT@4H) keeps `since`, so alternating scales
  cannot restart the count.
- **A RESOLVE keeps the item.** The resolved evidence is filed under its
  resolution (judged, not erased), so a resolved structural event still
  counts toward the decay and can still release a decayed pair.
- **Same-bar events have a defined order.** On one bar the events in
  the bias direction are read first: a same-bar pair resets, then
  counts one, whatever the ids.
- **The open risk is the risk from here.** `OrderMachine.open_risk(close)`
  is the distance from the bar's close to each stop on the *filled*
  contracts, the same marked base as the session's opening equity — a
  position carried across the 18:00 New York boundary is no longer
  measured entry-to-stop against a budget marked at the boundary, and an
  underwater position is no longer counted twice.
- **A reversal also withdraws the working entry** on that scale (reason
  `structure_reversed`, the thesis closes) and a plan against the bar's
  reversal is refused (`thesis_refused` `structure_reversed`) instead of
  submitted into it.
- `bias_decayed:<PAIR>:<n>` reports the events actually counted (an MSS /
  BOS on the bias scale ends it at one); the prompt names `account_risk`,
  `decayed_at`, the `event_sleep` outcome and the wake's rejections.

Accepted, recorded: the evidence ledger does not cross the archive (a
count in progress restarts at the wake; the memory does not); pending
items expired by `max_pending_evidence` leave the count; the per-trade
budget is a fraction of the current equity while the session budget is
the opening's, so two BASE positions above the opening can be one too
many; the wake input cannot show the decayed pair — the rejection on the
next call does.

## 2. Validation

- Unit tests per task, TDD: gate (schema 4 loads; the two policy checks;
  the open-risk veto and its remaining-budget arithmetic; the machine
  passes the open risk), contract (schema 3 round trip, schema 2 reads
  back, the reply refuses `governing_timeframe`), reducer (thesis scale
  from the bias, target scale, continuity, decay on the third-scale
  sequence of 01-24 and the same-direction reset, the MSS-on-S end, the
  re-assert refusal and its release by an MSS), machine (a 15m MSS against
  a 15m thesis flattens it; a 5m one does not; a bias flip does not), stack
  (the events reach the machine), prompt words, summarizer counts.
- Full suite (`pytest -p no:cacheprovider`, ~28 min).
- The frozen window and the ten benchmark windows with `--label
  scale-exit-bias`; two receipts against the stop-floor pass; regression
  baselines re-frozen on the eleven new journals; the research regression
  test.

## 3. Risks

- At 2.5 % of 100 000 USD one contract still needs a stop under 125
  points: the 25th percentile of the refused proposals stays refused; the
  1H theses a 4H bias implies are mostly refused. A larger account or MNQ
  remains a configuration decision.
- The 15m decay counts the 5m: two 5m counter-displacements without a
  same-direction 5m / 15m event in between end a 15m bias on a trend day
  with a deep pullback; the bias is then re-set by a 15m MSS / BOS or set
  on the 1H. The simulation shows no such case on the four trend days.
- The eleven stop-floor journals no longer replay (the reply contract
  changed); the baselines are re-frozen on this pass's journals.
- The reasoning budget (`max_tokens` truncations) is untouched.
