# Trading Brain

The Brain interprets the Eye's published facts. It owns no market state and no
history: it reads one `MarketObservation` per completed 1m bar and, when the
Eye reports something worth reasoning about, asks an LLM to evolve a
persistent `BrainState`. It names Eye objects; it never names a price.

Design: [specs/2026-09-16-llm-brain-design.md](specs/2026-09-16-llm-brain-design.md).
Plan: [plans/2026-09-16-llm-brain.md](plans/2026-09-16-llm-brain.md).

```text
Market Data ─► Trading Eye (unchanged) ─► MarketObservation
                                             │
                     eye_view.py             EyeContext: aliased objects, events, price relations
                                             │
                     sleep_controller.py     WAKE / STAY_ASLEEP / UPDATE / TICK / EVENT_SLEEP
                                             │
                     runtime.py ─────────────┤ SLEEP ↔ ACTIVE
                          ├─ main_brain.py   LLMInput → DeepSeek → LLMUpdate
                          ├─ reducer.py      BrainState_t + evidence + update → BrainState_t+1
                          ├─ opportunity_geometry.py  aliases → entry / stop / target, R
                          └─ journal.py      hash-chained JSONL per episode
```

## What replaced what, and why

| retired | date | why |
| --- | --- | --- |
| the typed playbooks, DOL ranking and Signal Policy | 2026-09-07 | a fixed taxonomy asserted about the market |
| the frozen six-path competition set and the global mode library | 2026-09-09 | a global fit cannot say what is likely *from here* |
| the local conditional kNN Brain (`forecast.py`, `hypothesis_*.py`, `belief_updater.py`, `trajectory.py`), its information-gain and Setup gates, the old `decision.py` / `risk.py`, the un-importable engine | 2026-09-16 | measured three times on 2022-01 and never beat the driftless rate — [evidence/](evidence/) |

The Eye's facts were never the problem: the repairs that followed those runs
are on `main`. What failed was every *fixed* pipeline over them. The LLM Brain
reasons over the same objects on demand, under a deterministic gate, and
writes a journal that replays without hindsight.

## Modules — `brain/core/`

| module | owns |
| --- | --- |
| `eye_view.py` | `build_eye_context`: the aliased, JSON-ready view of one observation (objects per scale, structure / delivery / range / liquidity summaries, session, the ACTIVE interaction paths aliased to their source object with `last_step_at` and `stepped_since_last_call`, this bar's evidence events, price relations); `delivery_payload` / `reset_payload` (the forming leg, the displacement's age, the protection break); `interaction_rows`; `assert_causal`; `visible_liquidity_ids` |
| `object_registry.py` | `ObjectRegistry`: `FVG_5m_3` ↔ Eye entity id, assigned on first appearance inside an episode in a deterministic order |
| `sleep_controller.py` | `decide(events, active, config, relation_changes, known_at, previous_known_at)` → `WAKE` / `STAY_ASLEEP` / `UPDATE` / `TICK` / `EVENT_SLEEP`; `ControllerConfig` (schema 5: the bookkeeping kinds, the wake set, `relation_change_timeframes`, `relation_change_debounce_bars`, the idle-archive threshold, and since 2026-09-21 `events` — the calendar and the sleep window per release kind; the `sha256` covers the config and the calendar) from `configs/sleep_controller.json` |
| `event_calendar.py` | (2026-09-21) `parse_ics` (RFC 5545 unfolding, `TZID` / Zulu / floating times, all-day events skipped), `EventRule`, `ScheduledEvent`, `EventFilter` (`active(known_at)`, `ended_between(previous, known_at)`, `from_config`); the data is `configs/economic_calendar.ics`, built by `scripts/build_event_calendar.py` from `configs/calendar_sources/` (the BLS feed's CPI and Employment Situation events, the 2022 BLS schedule, the Fed's FOMC calendar page) |
| `main_brain.py` | `MainBrain.step`: build the `LLMInput`, prove it causal, call under the retry policy, parse, reduce; `MainBrainConfig` from `configs/main_brain.json` + `configs/prompts/main_brain_system.md` |
| `llm_client.py` | `DeepSeekClient` (urllib, JSON mode; key from `DEEPSEEK_API_KEY`, else the gitignored `brain/configs/deepseek.key` — `resolve_api_key`), `ScriptedClient`, `EchoClient`, `RecordedClient`, `call_with_policy` |
| `reducer.py` | the pure function `apply(prev, evidence, update, ctx)`; `empty_state`; `pending_evidence`; `sleep_blockers`; since 2026-09-17 rule 4 also refuses an invalidation object more than one scale below the thesis's `governing_timeframe` and a `thesis_id` that flips direction; since 2026-09-18 rule 4b drops an opportunity against the reply's `bias` (`opportunity_against_bias`), any opportunity under a NEUTRAL bias, and a thesis on a scale above the bias scale (`opportunity_scale_above_bias`); since 2026-09-20 rule 4 also refuses an entry the market is already past (`coherence_error` → `entry_side_error`: a LONG entry above the close, a SHORT below it), and every rejection of an update travels in the state's `last_update.rejections` — what the next call reads as `prior_state.last_update.rejections`; since 2026-09-22 the thesis scale is the reducer's (`THESIS_SCALE_OF_BIAS`: one below the bias scale, never below the 15m — the reply's `governing_timeframe` is gone and `opportunity_scale_above_bias` with it), rule 4c refuses a target below it (`opportunity_target_scale`), and rule 4d (`effective_bias`) carries the bias with its `since` (keyed on the direction, so a scale change does not restart the count), keeps the decay memory (`decayed` / `decayed_at`) through NEUTRAL and other pairs until its scale prints an MSS / BOS, refuses the decayed pair re-asserted without one (`bias_reassert_refused`) and decays a bias to NEUTRAL when the scales below it print `BIAS_DECAY_EVENTS` structural events against it (`bias_decayed:<PAIR>:<n>`; a same-direction event resets — first on its bar —, an MSS / BOS on the bias scale ends it at once); a wake carries the archived bias in (`ReduceContext.carried_bias`, kept even when the wake call is an incident); a RESOLVE files the resolved item under its resolution rather than erasing it, so the decay still reads it |
| `opportunity_geometry.py` | `resolve_geometry` / `coherence_error`: object aliases → `OpportunityGeometry`; a `CLOSE_BEYOND` invalidation puts the hard stop `CLOSE_BEYOND_BUFFER_ATR` × 1m ATR × √(scale minutes) beyond the object; since 2026-09-20 a zone that *contains* price is entered at its midpoint, or at its far edge when price is already past the midpoint (`entry.zone.inside_midpoint` / `entry.zone.inside_far_edge` — the near edge of a containing zone is a buy at the market), a range is refused as an entry object, and `entry_side_error` judges which side of the close a limit rests on — at proposal (`coherence_error`) and at submission (the order machine), never on the bars between, where a working limit the tape crosses must fill; since 2026-09-21 the hard stop is never nearer the entry than `STOP_FLOOR_GOVERNING_BARS` (1.0) bars of the thesis's `governing_timeframe` — 1m ATR × √minutes — and a stop the floor moved carries the rule id `stop.floor.governing_bar` (an opportunity with a governing scale needs `atr_1m`) |
| `position_ledger.py` | `PositionLedger` protocol (`has_open_position`, `has_working_order`, `execution_view`), `engaged`, `IDLE_VIEW`; `InMemoryPositionLedger`; the real one is `execution.core.order_fsm.ExecutionLedger` |
| `journal.py` | `BrainJournal` (writer), `JournalReader`, `record_hash` |
| `runtime.py` | `BrainRuntime.step(observation)`: the SLEEP ↔ ACTIVE machine; on `EVENT_SLEEP` the episode is archived without a call (journal `sleep` with reason `event:<kind>:<release>`, `StepResult.event` for the executor); `StepResult.llm_latency_ms`; optional `Timings` (`controller`, `journal`; the Brain records `input`, `llm`, `reduce`) |
| `brain_entry_sequence.py` | the Brain-side reading of the Eye's interaction facts (also consumed by `shares/core/scene_graph.py`) |

Contracts: `contract/brain/state.py` (`BrainState`, schema 3 since 2026-09-22 — `Bias(direction LONG | SHORT | NEUTRAL, scale, basis, since, decayed, decayed_at)`, the last three reducer-owned and shown to the LLM in `prior_state.bias`; schema 2 (2026-09-18) added `bias`; a schema-1 journal reads as NEUTRAL on 15m, a schema-2 one with `since` / `decayed` unset; `THESIS_SCALE_OF_BIAS`, `BIAS_DECAY_SCALES`, `BIAS_DECAY_EVENTS`, the structural / reversal evidence kinds; `LastUpdate.rejections` since 2026-09-20, empty for older journals), `contract/brain/llm.py` (the reply; since 2026-09-22 its `opportunity` carries no `governing_timeframe` — the thesis scale is the reducer's)
(`LLMInput`, `LLMUpdate`, `parse_update`), `contract/decision/opportunity.py`
(`OpportunityGeometry`).

## The Sleep Controller

Once per completed 1m bar, from the Eye's transition events alone
(`*_state` re-publications and the `bar_completed` / `market_epoch_reset`
heartbeats are not transitions):

| runtime status | condition | decision |
| --- | --- | --- |
| any | the bar lies inside a scheduled release's window (`events` in the config, 2026-09-21: CPI and NFP from 60 minutes before to 30 after the 08:30 print, an FOMC statement from 60 before to 90 after 14:00, New York) | asleep: `STAY_ASLEEP`; active: `EVENT_SLEEP` — the episode is archived without a call and the executor withdraws every expression (reason `event:<kind>:<release>`) |
| SLEEP | the first bar at or after a window's end | `WAKE` with reason `event_ended:<kind>:<release>`, Eye event or not — a new episode reads the post-release market from nothing |
| SLEEP | an event whose kind is in the wake set for its timeframe | `WAKE` |
| SLEEP | otherwise | `STAY_ASLEEP` |
| ACTIVE | a *reaction* at 5m or above (a transition that is not a bookkeeping kind), **or** a watched object on one of `relation_change_timeframes` (15m, 1H, 4H) changed its side of price since the last LLM call — a 5m pool crossing price no longer counts (49 % of the 2022-01-03 run's calls), and since 2026-09-18 one alias triggers at most once per `relation_change_debounce_bars` (15) 1m bars (`72ea13c7`'s 75 relation-only calls had a median gap of 2 minutes between repeats of the same alias; the replayed rule drops 44 calls and no coverage) | `UPDATE` |
| ACTIVE | otherwise | `TICK` — no LLM call; the state's revision and `updated_at` advance; the bar's 5m+ bookkeeping evidence is deferred and delivered with the next call |

`configs/sleep_controller.json` (schema 5) names the *bookkeeping kinds*
once — formation (`fvg_created`, `swing_confirmed`, `structural_leg_created`,
`liquidity_level_created`, …), touches and level bookkeeping, and since
2026-09-18 the delivery-phase transitions (`delivery_phase_entered` /
`_exited`: derived labels the `scales` block already carries, verdicted
NEUTRAL 68–100 % of the time on every scale) — and uses the
list twice: they never wake the Brain, and they never trigger an UPDATE on
their own. Wake set: any non-bookkeeping transition on 15m / 1H / 4H, plus
`mss_core_confirmed`, `qualified_bos`, `sweep_confirmed` and
`displacement_observed` on 5m. Measured on the 2022-01-03 Globex session
(1380 bars, [evidence](evidence/2026-09-16_controller_eye_audit_2022-01-03.md)):
14 % wake bars; of the 81 sharp 15-bar moves (range > 2.5 × ATR) the wake
set fires within [-3, +10] minutes of 77, RTH 24 of 25 — without 5m
displacement it was 62 and 19. The 1m tape is context (counts per kind and
the last eight reaction events since the previous LLM call), never evidence.
The controller has no weights, no direction and no price threshold; going
back to sleep is the reducer's decision — the LLM's own request, or the idle
rule below.

**Idle rule** (`idle_archive_after_updates`, default 6): the runtime counts
consecutive accepted updates that propose no opportunity and keep the
understanding; the update that reaches the threshold archives the episode
regardless of `watch_next` and of the interaction gate (a position or an
unjudged item still holds it open). The 2022-01-03 real run showed the model
never asks to sleep on its own; this is the rail.

**Event windows** (`events`, schema 5, 2026-09-21): `calendar` is an
`.ics` file (`configs/economic_calendar.ics`) and each rule names a
release kind, the full-match regex on the event's `SUMMARY`, and the
minutes slept before and after it. The calendar is data, not code: it is
built by `scripts/build_event_calendar.py` from `configs/calendar_sources/`
(the BLS feed's CPI and Employment Situation blocks, the BLS 2022 schedule
page's rows, the Federal Reserve's FOMC calendar page — a statement at
14:00 New York on the last day of every scheduled two-day meeting); each
source file's header names its URL and fetch date, and a test checks the
committed calendar equals the script's output. Rebuild it when the BLS
feed moves a release or a year runs out. The calendar's bytes are part of
the controller's `sha256`, so a run's identity changes with it.

## The Main Brain

The system prompt carries the fourteen-step framework (定位 / 推演 / 交互 /
反应 / 对齐 / 评估 / 反向 / 重构 / 预期 / 目标 / 表达 / 证伪 / 风控 / 跟踪)
with the instruction to reason through the steps the situation needs and mark
the rest `"n/a"`, the incremental rule, the "Bias" section (2026-09-18: which
scale sets the direction — the 15m unless the 1H or 4H delivery is *live*:
active leg past one ATR of its scale with a displacement at most three bars
old or an MSS / BOS as the latest structural event, and a live scale stays
live until its leg ends, not when the excursion dips under one ATR
(2026-09-19); the 5m never sets the bias; a stale phase is location, not
direction; a `reset` makes its side live), the "Expression" section
(2026-09-20: the bias picks the side and fires nothing; the trade is a
resting limit at a retracement object on the scale below the bias scale —
nearest first, follow the leg, the wait is fifteen bars of the object's
scale, `prior_state.last_update.rejections` says what code refused and
why; it replaced the 2026-09-18 rule "after a BOS on the bias scale, the
object that `contains_price` — the order fills now"), the hard
rules (objects only, no prices, a counter candle is not delivery, the
invalidation judged on the thesis scale and the direction on the bias
scale, sleep only when nothing is pending; since 2026-09-22 the Bias
section carries the decay rule — code ends a bias the scales below it
deliver against, re-sets it on structure only, and says NEUTRAL in a
balance — the thesis scale is code's and the target lies on it, and a
position leaves on a structural reversal, never on a bias flip)
and the output contract rendered from `LLM_UPDATE_EXAMPLE`. Its sha256 is in
every run's identity.

`LLMInput` (schema 3 since 2026-09-20 — `prior_state.last_update.rejections`; schema 2 since 2026-09-18): `episode_id`, `known_at`, `trigger`, `bar` (close, 1m ATR),
`session` (with `drift_atr`, the close against the session open in 1m
ATRs), `scales` (4H / 1H / 15m / 5m summaries with aliases; 1m structure
and delivery only — each `delivery` carries `active_leg_direction` (the leg
price is in now), `last_leg_direction`, `forming_leg_atr`, the
displacement's score, `displacement_direction` and `displacement_age_bars`;
each `structure` carries `reset` when an acceptance broke the protected
swing and no structure has confirmed since), `interaction`, `new_evidence` (this bar's evidence, then
the bookkeeping evidence of the TICK bars since the last call, then every
item an incident bar left unjudged, marked `pending_since`, until the LLM
verdicts it), `tape_since_last_update`, `price_relations` (for every aliased object
within `relation_atr_limit` ATRs of the close, plus every object the prior
state names: `position` — where the *object* lies, `above_price` /
`below_price` / `contains_price` — and `offset_atr`, its signed distance
in 1m ATRs, positive above price; since 2026-09-20 every liquidity pool
is placed whatever its distance — a pool the LLM could not see the side
of was named as a target on the wrong side in seven of the twelve
incoherent proposals of the entry-model benchmark; renamed on 2026-09-17 from
`relation` / `distance_atr`, which described price's place against the
object and which the model read the other way round on every real run), `prior_state`
(the state without its registry, each evidence list bounded to its last
`prior_evidence_limit` items in a compact shape with notes cut at
`note_limit`, plus counts; pending items are not repeated there — and at
most `max_pending_evidence` (32) of them stay in the ledger and are
re-offered, the oldest expiring as `evidence_expired`, since 2026-09-19:
run X re-offered 124 unjudged items after 25 empty replies in a row and
its input grew until no reply could come; and, since
2026-09-17, `execution` — the order machine's view: `status`, the working
`order`, the `position`, the `last_outcome` and the `last_veto` of the
episode, as aliases and counts, never prices — with prompt rules that a
veto is a statement about geometry and the account, not the market, that
the same three objects are not re-proposed while its cause stands, and that
an expired, rejected or closed intent is not traded again in the episode
unless the opportunity changes). On the
2022-01-03 journal the full input was 48 k characters a call; the registry
alone was 11.6 k and 63 of 91 relation rows were more than 3 ATR away.

`LLMUpdate` (strict JSON): one verdict per evidence item (`SUPPORT` /
`CONTRADICT` / `NEUTRAL` / `RESOLVE` with `resolves_evidence_id` and
`resolution`), `understanding_holds`, `market_understanding`,
`active_expectation`, `watch_next`, `destination_candidates`, `opportunity`
(state, direction, three object aliases), `reasoning_confidence`,
`continue_active`, `framework_trace` (`step_1` … `step_14`).
`parse_update` refuses unknown keys, unknown aliases, out-of-vocabulary enums
and any evidence id that was not in `new_evidence`; there is no numeric price
field to refuse.

DeepSeek: `POST {DEEPSEEK_BASE_URL}/chat/completions` in JSON mode, model
from `configs/main_brain.json` (`deepseek-flash`, `reasoning_effort`
`high` — set explicitly on 2026-09-17 after the same-window comparison in
[evidence/2026-09-17_week_backtest_2022-01-03_07.md](evidence/2026-09-17_week_backtest_2022-01-03_07.md);
`low` and `max` remain available through `--reasoning-effort` and
`configs/main_brain_max.json`; `max_tokens` 32768, `timeout_s` 300 — reasoning tokens count against the cap, and a reply with
`finish_reason=length` is refused as malformed; `max` reasoning runs past
32 768 tokens on a real input, so `configs/main_brain_max.json` carries
65 536 and a 900 s timeout for that effort); TLS is verified against
certifi's bundle; `reasoning_content` is journaled, never parsed. Retry policy: timeouts, 429 (honouring
`Retry-After`), 5xx and dropped connections (the server closing without a
response, a reset, a bad status line — bare `http.client` / `OSError`
failures `urlopen` raises outside `URLError`) retry with exponential backoff
up to `max_retries`; a
malformed reply gets exactly one repair attempt; anything past that is an
*incident* — the state carries forward and the runtime stays ACTIVE.

## The reducer

`apply` is a pure function of the prior state, this bar's evidence, the
update and a `ReduceContext` (known_at, open position, open interaction,
visible aliases, registry, geometry coherence):

1. bookkeeping — `revision + 1`, `updated_at = known_at`, registry merged;
2. evidence — every item gets its verdict's list; an item without a verdict
   lands in `unresolved` with a rejection as a *pending* item (`verdict`
   null) and is re-offered on every later call until it is judged; an item
   already in the ledger is skipped with `evidence_duplicate`; `NEUTRAL`
   items also sit in `unresolved`, judged, and may be `RESOLVE`d later;
   `RESOLVE` removes its target from `unresolved` and files the new item by
   `resolution`;
3. an abandoned understanding must be replaced, else the whole update is
   refused (`understanding_not_replaced`);
4. an opportunity must name three distinct, currently visible aliases whose
   geometry is coherent for its direction — since 2026-09-20 including a
   limit on the resting side of the close — else it downgrades to `NONE`;
   the update's rejections are carried in `last_update.rejections` for the
   next call to read;
5. an open position forces `continue_active`;
6. sleep is granted only when all five exit conditions hold — no open
   position, no *open* interaction path (one that stepped since the last LLM
   call; the Eye keeps a path ACTIVE for as long as its context lives, so
   "any ACTIVE path" held the Brain awake on every bar of the real tape),
   no *pending* evidence (a `NEUTRAL` verdict never blocks sleep), nothing
   in `watch_next`, `opportunity.state == NONE`; a refused sleep is recorded
   as `sleep_refused:<condition>`;
7. the idle rule — with `ReduceContext.idle_archive_after` set, an accepted
   update that keeps the understanding and proposes no opportunity, when
   `idle_updates + 1` reaches the threshold, archives the episode unless a
   position is open or an item is still unjudged; `ReduceResult.sleep_reason`
   says which rule slept (`continue_active=false` or `idle`).

## The runtime

```text
SLEEP  ──WAKE──►  ACTIVE  (new episode EP_<YYYYMMDD>_<NNN>, new registry, revision 0)
ACTIVE ──UPDATE──► LLM → reduce ──slept──► archive (status ARCHIVED, journal `sleep` with its reason) → SLEEP
ACTIVE ──TICK──►  revision + 1, no LLM; 5m+ bookkeeping evidence deferred to the next call
ACTIVE ──incident──► state carried forward, still ACTIVE
ACTIVE ──EVENT_SLEEP──► archive without a call (journal `sleep`, reason `event:<kind>:<release>`) → SLEEP; the executor cancels the working entry and flattens every position on the same bar
SLEEP  ──window ends──► WAKE on the first bar after it (reason `event_ended:…`), a new episode from nothing
```

A wake whose first reasoning already satisfies the exit conditions archives
on the same bar. `known_at` must strictly increase across steps.

## Permission boundary

The LLM's writable surface is `LLMUpdate`; every object it names is a
registry alias validated against the input. `opportunity_geometry` derives
entry (a zone's near edge, a range's value price, a pool's midpoint, a
swing's price), stop (the invalidation object's far edge ± one tick) and
target (the target object's near edge / midpoint / price), checks
`stop < entry < target` (mirrored for SHORT) and computes reward-to-risk.
Position sizing, order compliance, hard stops and the Risk veto are a later
phase; `PositionLedger` is the boundary they attach to.

## The journal

`outputs/brain_journal/<run_id>/run.json`, `index.jsonl`,
`episodes/<episode_id>.jsonl`. `run_id` is the first 16 hex of the sha256 over
the tape's sha, the window, the model, the prompt and config shas and the
Eye's identities; `run.json` also records the git revision (and a dirty flag)
and, at the end, the counts of every decision, the LLM calls and incidents.

Records, each `{seq, record, episode_id, known_at, prev_hash, hash, payload}`
with `hash = sha256(prev_hash + canonical(body))` chaining from `run_id`:
`episode_opened → wake → llm_call (input_sha, the full input, the reply,
reasoning_content, usage, latency, finish_reason, attempts, the repair
reason and the refused reply when a repair was needed) → state (the full
BrainState, rejections) | tick (revision) → opportunity (opportunity +
geometry) → incident (kind, message) → sleep`. `trade` is reserved for the ledger. The writer refuses a
record whose `known_at` steps back or whose revision is not the previous plus
one.

## Causality and replay

`build_eye_context` and `MainBrain.build_input` assert that every ISO
timestamp they serialize is `<= known_at`; the runtime asserts `known_at`
increases. `brain/scripts/replay_journal.py --run-dir <run>` re-drives the
Eye over the run's window, rebuilds every `LLMInput`, compares its sha with
the journal's, feeds the recorded replies (and incidents) back through the
reducer, and requires every state and tick revision to match — exit 0 only
when every episode reproduces; since 2026-09-21 it stops at the gate's
drawdown halt as the runner does (the 2022-10-13 benchmark window halted
at 08:32 and replayed with extra revisions until then). `brain/tests/test_replay.py` proves it on a
synthetic tape, including a tampered record and a dropped revision.

## Running

```bash
.venv/bin/python -m brain.scripts.run_llm_brain --client deepseek \
  --warmup-start 2021-12-30 --emit-start 2022-01-04 --end "2022-01-04 12:00" \
  --max-llm-calls 60
```

`--client echo` needs no key: a contract-valid reply for every call, for
smoke runs. `--reasoning-effort low|high|max` overrides the config's
effort and enters the run identity (`deepseek:deepseek-flash@low`);
`run.json` ends with `machine_stats` and per-component `timings`.
`--label <text>` (2026-09-19) enters the run identity for a deliberate
re-run on the same Brain inputs — the executor code is not hashed, so run
X (execution layer) would otherwise have been refused as run B′.

`brain/scripts/run_benchmark.py --client deepseek --broker sim --reasoning-effort high --label <text> --parallel 3 [--only <text>] [--dry-run]`
(2026-09-20) runs one `run_llm_brain` per window of
`configs/benchmark_windows.json` (ten 2022 windows across regimes, market
time, the Eye warmed `warmup_days` before each), `--parallel` at a time,
logs under `outputs/brain_journal/benchmark_logs/`; `--dry-run` prints
the commands.

`brain/scripts/build_event_calendar.py [--output <path>]` (2026-09-21)
rebuilds `configs/economic_calendar.ics` from `configs/calendar_sources/`;
the committed calendar must equal its output (`test_build_event_calendar.py`).

`brain/scripts/summarize_run.py --run-dir <run> [--run-dir …] [--write]`
turns a journal into `summary.json` (calls, tokens and cost at the rates
in `configs/llm_pricing.json`, latency, triggers, sleeps, sharp-move
coverage, opportunities, vetoes and their repeats after the LLM saw them,
the order lifecycle, invariants, the account, timings, and since 2026-09-18
a `bias` section — bias changes, NEUTRAL revisions, opportunities dropped
against the bias, and `direction_accuracy_60m`: the share of state
revisions whose stated direction matched the sign of the close an hour
later; since 2026-09-19 `orders.missed_trends` — expired entries the tape
ran at least one R away from without touching the limit — and the
`bias_reversed` exit kind (since 2026-09-22 `structure_reversed` in its place, and the bias block's `decays` / `reasserts_refused`); since 2026-09-20 `orders.entry_quality` — each
fill's location in the range of the 60 and 240 bars before it (0 the
window's best price for the trade, 1 its worst; `chased` counts fills at
or past 0.8 of the 240-bar window), its wait, its excursions over the next
hour in R and whether the close an hour later was on its side, plus
`fill_rate` — and `brain.actionable_direction_accuracy_60m`) and prints several
runs side by side; `--until <UTC time>` bounds the counts to a common
window when runs differ in length.
`brain/scripts/audit_scales.py --warmup-start … --emit-start … --end …`
drives the Eye alone over a window and prints, per scale, every change
point of the facts the Brain reads (directions, protection, reset, active
and last leg, phase, displacement) beside the close — the deterministic
check of an Eye change before an LLM run; `--triggers <run-dir>` replays a
run's call triggers under the controller as configured and prints the
calls kept and their sharp-move coverage — the deterministic check of a
wake/sleep change. `--broker sim|ibkr` adds the Risk gate and the order machine
([execution/docs/README.md](../../execution/docs/README.md)); the machine's
ledger is the Brain's `PositionLedger`, so an engaged Brain (a position or a
working order) cannot sleep and the idle rule does not fire. The DeepSeek key is read from `DEEPSEEK_API_KEY`, or, when that
is unset, from the key file `brain/configs/deepseek.key` (path overridable
with `DEEPSEEK_API_KEY_FILE`); `*.key` is gitignored, and the key is never
written to a command line the repository owns or to a journal.

## Receipts — `brain/docs/evidence/`

[2026-09-22_scale_exit_bias_frozen_window_2022-01-03.md](evidence/2026-09-22_scale_exit_bias_frozen_window_2022-01-03.md):
the frozen window under the thesis scale, the structural exit and the
bias decay (`2cd6fd64acccccfa`, −125.75 closed and +49.25 with the open
LONG marked, for −77.25 / −30.75): the 4H short of 19:00 decayed at 22:30
instead of being read through the night and the day, five decays and no
refusal, the bias NEUTRAL in 128 revisions (24), 85 ACTIONABLE proposals
(207) and 16 distinct opportunities (47); no bias-flip exit (seven
before), one structural flatten at 07:01, the 11:12 LONG held through two
decays to +87.5 a contract at the close; no size veto, eight leverage
vetoes (the 8× cap holds two NQ); replay OK.

[2026-09-22_scale_exit_bias_benchmark_2022.md](evidence/2026-09-22_scale_exit_bias_benchmark_2022.md):
the thesis scale, the structural exit and the bias decay over the ten 2022
windows (label `scale-exit-bias-2`), paired with the stop floor's pass —
no 5m thesis (was 10 of 41 plans) and no 5m target (was 4), the fills'
stops 3.9–4.6 ATRs, `bias_reversed` gone and one `structure_reversed`,
five decays and two re-asserts refused on the reversal and chop days
(01-24 NEUTRAL 57 minutes after the low, 10-13 within the hour), the
memory carried across sleeps and episodes; −299.25 on seven fills (−137.0
per contract, +103.5 open at the window ends, one right) against −286.5:
the three stops were the two- and three-contract fills at 35–50-point
stops, the reversal days' correct theses were 130–290-point stops one
contract cannot fund at 2.5 % of 100 000 USD, and the trend days still
fill nothing (entries paused).

[2026-09-21_stop_floor_event_sleep_benchmark_2022.md](evidence/2026-09-21_stop_floor_event_sleep_benchmark_2022.md):
the stop floor and the event sleep over the ten 2022 windows, paired with
the entry model's third pass — no stop inside 2.2 one-minute ATRs (median
3.9, was 1.6), one contract instead of three, no order resting through a
release (10-13: −67.5 for −1 055 and a halt), the calendar wake on the
first bar after each window; −286.5 in all with six fills, all stopped,
none on a trend day: the floored stop prices a 15m thesis at 1 900–2 300
USD a contract on a 25–40 ATR tape, more than 1.5 % of 100 000 buys, the
model lowers the governing scale to 5m to fit it, and targets stay the
next 5m pool (ratios 1.1–1.7 vetoed).

[2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md](evidence/2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md):
the frozen window under the stop floor (`f4f6998a6ed59eb8`, −77.25 for
−141.5): nine fills at the limit (waits up to 101 minutes), stops of 3.9
ATRs and more, one stop-out instead of three and no daily stop; seven of
eight exits are the bias's own flips (average −4 points), which now set
the P&L; nine reasoning-budget truncations recorded; replays.

[2026-09-20_entry_model_benchmark_2022.md](evidence/2026-09-20_entry_model_benchmark_2022.md):
the entry model over ten 2022 windows across regimes — no chase anywhere
(2 of 20 fills past 0.8 of the four-hour range), resting limits that fill,
the direction right on the trend days and wrong on every reversal, the
right thesis expressed on one trend day of four (pools the model could not
place — spec §1.7 — and stops sized by a 5m object blocked the others);
the second pass under the pool fix was cut by HTTP 402 (the API balance)
and is recorded, not read.

[2026-09-20_entry_model_frozen_window_2022-01-03.md](evidence/2026-09-20_entry_model_frozen_window_2022-01-03.md):
the frozen window under the entry model (`b8870c30ec04d5df`): no fill at
the market (waits 80/15/5/4/1/1 minutes, two chases refused), the code-made
`opportunity_incoherent` gone, ACTIONABLE accuracy 0.41 (0.32), the wait
and the replacement at work, and still −167 — six stops decided by the
direction at the open and a 2.87 R winner with no way to keep it; replays;
§7 the same window under the pool fix (`2710e0b78a707e99`, −141.5, the
day's reading in a 0.44–0.50 band run to run).

[2026-09-18_execution_entry_day_run_2022-01-03.md](evidence/2026-09-18_execution_entry_day_run_2022-01-03.md):
the execution layer of the direction fix and the build's close — the
expiry refund, the marketable fill, the bias-reversal exit and
`missed_trends`; run X (`ab572b09d59bc4d7`) had no short into the 13:45
breakout for the first time and still lost −146.5 (seven entries at the
extreme of the leg that set the bias tripped the daily stop before 10:00,
and the correct afternoon thesis could not be expressed); the 2022-01-04
guard run (`5cfe65bcc480a032`) read the sell-off short all afternoon and
closed at −3.5; both replay and are the regression baselines; the
pending-evidence runaway of run X's first attempt (`max_pending_evidence`)
and `--label` are documented there; the layered table 72ea13c7 → E → B →
B′ → X closes the receipt.

[2026-09-18_brain_bias_day_run_2022-01-03.md](evidence/2026-09-18_brain_bias_day_run_2022-01-03.md):
the Brain layer of the direction fix — the `bias` in reply and state,
reducer rule 4b, the prompt's bias section, the relation debounce; run B
(`07f897f45fab5025`) raised `direction_accuracy_60m` to 0.445 (E 0.381,
`72ea13c7` 0.359), turned the overnight side long, restored sleep and cut
calls to 297 with coverage unchanged — and lost more (−121 points) because
the bias flipped 28 times (median segment 18 min) on a threshold rule with
no hysteresis and the 5m set the bias 18 times; the amendment (bias scale
floor 15m, a live scale stays live until its leg ends) in run B′
(`3cc1bb402bb7b964`) halved the changes, took the accuracy to 0.502 and
the coverage to 80 of 81 — and lost −139.5, because the entry is taken at
the extreme of the leg that set the bias and a position is held through
the bias flip against it (a SHORT stopped by the 13:45 breakout in all
four runs); the layer is closed and the second fact goes to the execution
layer as the bias-reversal exit.

[2026-09-18_eye_forming_leg_day_run_2022-01-03.md](evidence/2026-09-18_eye_forming_leg_day_run_2022-01-03.md):
the Eye layer of the direction fix — the per-scale facts now describe the
forming leg, the displacement's age and a broken protection; the
deterministic audit (`audit_scales.py`) shows the 4H and 1H reading
`active=long, retracement` where they read `expansion short` for hours;
run E (`5c491789ccef7367`, same prompt as `72ea13c7`) read the overnight
rise two hours earlier with two target hits, halved the RTH direction
churn, kept coverage and calls, and left the day's direction where it was
(`direction_accuracy_60m` 0.38 against 0.36, both below one half): the
facts were the Eye's to fix, the reading is the Brain layer's.

[2026-09-18_direction_root_cause_2022-01-03.md](evidence/2026-09-18_direction_root_cause_2022-01-03.md):
what moves the Brain's direction and why it did not move with the tape in
`72ea13c7fcbc1cff` — no code owns the reading (the model rewrites text under
rule 3; verdicts are bookkeeping), the Brain flips only on an Eye structural
event on its governing scale, and on 2022-01-03 those scales read short or
balance through the whole rise: the non-5m displacement score is frozen
between events (1H "0.64 short" for 16 h), the delivery phase describes the
leg that just ended (4H "expansion short" for 15 h), a broken protected
swing leaves the 1H directionless until two aligned swings confirm (no 1H
BOS/MSS from 10:00 to 17:00), and the one long thesis after the 13:45 15m
MSS expired twice at a retracement limit missed by 1–4 points; diagnosis
only, candidates listed for the owner.

[2026-09-18_risk_v2_day_run_2022-01-03.md](evidence/2026-09-18_risk_v2_day_run_2022-01-03.md):
the first run of the thesis lifecycle and Risk v2 on the same session
(`72ea13c7fcbc1cff`): 343 calls, 26 orders, 9 fills, 6 stops / 2
invalidation exits / 1 target, −2 060 USD by the fills; each of the five
fixes measured against `7ec17f066d232ba4` (no re-entry within 15 min of a
stop, every invalidation on the thesis's scale, no nearer invalidation
after a sizing veto, 5m relation flips gone from the triggers), the risk
parameters judged, and the two defects the run exposed (the simulator's
position averaging, the leverage cap ignoring open contracts) fixed after
it — so it is evidence, not the regression baseline.

[2026-09-17_trade_quality_root_cause_2022-01-03.md](evidence/2026-09-17_trade_quality_root_cause_2022-01-03.md):
why the complete 2022-01-03 run (`7ec17f066d232ba4`) lost 23 of 28 closed
trades — one 4H-short thesis re-expressed through swapped 5m objects (47
orders, 47 signatures, 14 re-entries within 15 min of a stop), invalidation
objects on 5m for a 4H thesis (stops at 0.58 × ATR(5m), one tick past a
pool), the 25-point budget cap and the veto feedback steering the Brain to
nearer invalidations, and the counterfactual showing wider stops do not
rescue the day (direction and timing are the root); diagnosis only.

[2026-09-17_week_backtest_2022-01-03_07.md](evidence/2026-09-17_week_backtest_2022-01-03_07.md):
the execution-feedback phase — the inverted `price_relations` reading that
had killed every opportunity of every real run (86 of 88 proposals), the
first trades on the tape, the veto feedback measured (0 re-proposals after
13 vetoes), low / high / max on the same RTH day, the order lifecycle and
the per-component timings (the LLM call is 99.9 % of the wall time), and
the two complete day runs frozen as the regression baseline; the week runs
stopped when the DeepSeek balance ran out and await a rerun.

[2026-09-16_controller_eye_audit_2022-01-03.md](evidence/2026-09-16_controller_eye_audit_2022-01-03.md):
the Eye audited against the raw tape over a full Globex session (FVGs,
liquidity candidates, session levels, price, continuity — all correct), the
wake-set measurement that added 5m displacement, the token breakdown behind
the input cut, and the third real run (six episodes, five sleeps on the real
model, 57 % fewer tokens, no coherent opportunity at low reasoning effort).

[2026-09-16_llm_brain_audit_2022-01-03.md](evidence/2026-09-16_llm_brain_audit_2022-01-03.md):
the audit of the sleep cycle — the three defects that made SLEEP unreachable
on the real tape (NEUTRAL in `unresolved`, incident-parked evidence with no
way out, "any ACTIVE path" as open interaction), the echo run that proves the
cycle after the fixes (18 episodes), and the second real DeepSeek run
(2022-01-03 09:00–12:00 NY, 79 calls, 5 journaled opportunities, replay OK).
[2026-09-16_llm_brain_e2e_2022-01-04.md](evidence/2026-09-16_llm_brain_e2e_2022-01-04.md):
the first real DeepSeek run (2022-01-04 08:30–11:30 NY, 45 LLM revisions,
replay OK at that revision) and what it changed. The three earlier receipts
measure the mechanical Brain this design replaced.

## Tests — `brain/tests/`

| file | covers |
| --- | --- |
| `test_brain_state.py` | round trip, invariants, unknown keys; `Bias.since` / `decayed` round trip, a schema-2 state reads them unset, the thesis-scale and decay constants (2026-09-22) |
| `test_llm_contract.py` | the reply gate: 16 malformed shapes, a reply naming `governing_timeframe` refused (2026-09-22), canonical input hashing |
| `test_opportunity_geometry.py` | entry / stop / target per object kind, LONG / SHORT mirror, incoherence; a containing zone's midpoint / far edge, a range refused as entry, the side rule at proposal only; the stop floor (one governing bar, both sides, composed with the close-beyond buffer, needs the ATR, none without a governing scale) |
| `test_object_registry.py`, `test_eye_view.py` | aliases, causality, price relations, reproducibility on the synthetic Eye; interaction rows (source alias, `last_step_at`, open since the last call); the forming-leg, displacement-age and reset keys, `drift_atr` |
| `test_audit_scales.py` | the change points of the per-scale facts and `scale_facts` on the synthetic Eye; `replay_triggers` keeps wakes, reactions and undebounced relation flips |
| `test_sleep_controller.py` | the wake rule kind by kind, UPDATE / TICK, the tape rule; schema 5's event filter (the calendar in the hash, CPI / NFP / FOMC windows), `EVENT_SLEEP` inside a window, the calendar wake at its end |
| `test_event_calendar.py` | `parse_ics` (TZID under EST and EDT, Zulu, folded lines, floating times, all-day skipped), rules → windows, `active` / `ended_between` boundaries, `from_config` and its hash |
| `test_build_event_calendar.py` | the FOMC page parser (`Month d-d`, `Mon/Mon d-d`, projection meetings, unscheduled rows skipped; eight 2022 statements from the real page), the built calendar's 2022 and 2025–2026 events, the committed file equals the script's output |
| `test_reducer.py` | every verdict route, RESOLVE, missing verdict, pending items (re-offered, verdicted late, resolved by a carrier, kept through another incident), NEUTRAL never blocks sleep, understanding replacement, opportunity downgrade, position, each sleep condition, TICK, incidents, determinism; the thesis scale from the bias, the target scale, the bias decay (continuity of `since`, two events against, the same-direction reset, the 5m counting under a 15m bias only, an MSS on the bias scale, the re-assertion refused until structure, the opportunity dropped under a decayed bias) |
| `test_llm_client.py` | retry policy (timeout / 429 / 5xx / dropped connection / 400 / malformed / repair), DeepSeek request shape and error mapping against a local HTTP server, empty content, key resolution (environment, key file, neither) |
| `test_journal.py` | hash chain, tampering, monotone `known_at`, revision gaps, the ledger |
| `test_main_brain.py` | prompt loading (the bias, decay, thesis-scale, structural-exit and Expression sections' words), input shape, wake / update / incident steps, bounded prior evidence, pending items re-offered and filed by their late verdict, the prior view's `last_update.rejections` |
| `test_runtime.py` | SLEEP → WAKE → UPDATE → TICK → sleep on the synthetic Eye, a NEUTRAL-only sleeper sleeps, evidence parked by an incident is verdicted later and sleep follows, incident keeps ACTIVE, open position, `known_at` monotone, episode numbering; an event window archives the episode without a call and the window's end wakes a fresh one |
| `test_replay.py` | a journal replays to identical states; tampering and a dropped revision are detected |
| `test_summarize_run.py` | the summary of a stack run on the synthetic tape, the cost arithmetic, the veto metrics on a hand-built journal, sharp-move coverage, rendering, `missed_trends`, `entry_quality` (with the sized distance and the contracts since 2026-09-21), the bias block's `decays` / `reasserts_refused` and the `structure_reversed` count (2026-09-22) |
| `test_run_benchmark.py` | the windows file, one command per window with the warmup, a dry run launches nothing |
| `test_run_guards.py` | `tape_is_current` for `--broker ibkr`, the effort label, `drive` timings |
| `test_regression_baseline.py` | `research_orchestration`: the frozen week backtest replays and summarizes identically ([evidence/regression_baselines.json](evidence/regression_baselines.json)) |
| `test_eye_link_real_tape.py` | `research_orchestration`: the real Eye on 2022-01-04 wakes the controller, never on 1m alone |
