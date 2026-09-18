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
                     sleep_controller.py     WAKE / STAY_ASLEEP / UPDATE / TICK
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
| `eye_view.py` | `build_eye_context`: the aliased, JSON-ready view of one observation (objects per scale, structure / delivery / range / liquidity summaries, session, the ACTIVE interaction paths aliased to their source object with `last_step_at` and `stepped_since_last_call`, this bar's evidence events, price relations); `interaction_rows`; `assert_causal`; `visible_liquidity_ids` |
| `object_registry.py` | `ObjectRegistry`: `FVG_5m_3` ↔ Eye entity id, assigned on first appearance inside an episode in a deterministic order |
| `sleep_controller.py` | `decide(events, active, config)` → `WAKE` / `STAY_ASLEEP` / `UPDATE` / `TICK`; `ControllerConfig` (schema 3: the bookkeeping kinds, the wake set, `relation_change_timeframes`, the idle-archive threshold) from `configs/sleep_controller.json` |
| `main_brain.py` | `MainBrain.step`: build the `LLMInput`, prove it causal, call under the retry policy, parse, reduce; `MainBrainConfig` from `configs/main_brain.json` + `configs/prompts/main_brain_system.md` |
| `llm_client.py` | `DeepSeekClient` (urllib, JSON mode; key from `DEEPSEEK_API_KEY`, else the gitignored `brain/configs/deepseek.key` — `resolve_api_key`), `ScriptedClient`, `EchoClient`, `RecordedClient`, `call_with_policy` |
| `reducer.py` | the pure function `apply(prev, evidence, update, ctx)`; `empty_state`; `pending_evidence`; `sleep_blockers`; since 2026-09-17 rule 4 also refuses an invalidation object more than one scale below the thesis's `governing_timeframe` and a `thesis_id` that flips direction |
| `opportunity_geometry.py` | `resolve_geometry` / `coherence_error`: object aliases → `OpportunityGeometry`; a `CLOSE_BEYOND` invalidation puts the hard stop `CLOSE_BEYOND_BUFFER_ATR` × 1m ATR × √(scale minutes) beyond the object |
| `position_ledger.py` | `PositionLedger` protocol (`has_open_position`, `has_working_order`, `execution_view`), `engaged`, `IDLE_VIEW`; `InMemoryPositionLedger`; the real one is `execution.core.order_fsm.ExecutionLedger` |
| `journal.py` | `BrainJournal` (writer), `JournalReader`, `record_hash` |
| `runtime.py` | `BrainRuntime.step(observation)`: the SLEEP ↔ ACTIVE machine; `StepResult.llm_latency_ms`; optional `Timings` (`controller`, `journal`; the Brain records `input`, `llm`, `reduce`) |
| `brain_entry_sequence.py` | the Brain-side reading of the Eye's interaction facts (also consumed by `shares/core/scene_graph.py`) |

Contracts: `contract/brain/state.py` (`BrainState`), `contract/brain/llm.py`
(`LLMInput`, `LLMUpdate`, `parse_update`), `contract/decision/opportunity.py`
(`OpportunityGeometry`).

## The Sleep Controller

Once per completed 1m bar, from the Eye's transition events alone
(`*_state` re-publications and the `bar_completed` / `market_epoch_reset`
heartbeats are not transitions):

| runtime status | condition | decision |
| --- | --- | --- |
| SLEEP | an event whose kind is in the wake set for its timeframe | `WAKE` |
| SLEEP | otherwise | `STAY_ASLEEP` |
| ACTIVE | a *reaction* at 5m or above (a transition that is not a bookkeeping kind), **or** a watched object on one of `relation_change_timeframes` (15m, 1H, 4H) changed its side of price since the last LLM call — a 5m pool crossing price no longer counts (49 % of the 2022-01-03 run's calls) | `UPDATE` |
| ACTIVE | otherwise | `TICK` — no LLM call; the state's revision and `updated_at` advance; the bar's 5m+ bookkeeping evidence is deferred and delivered with the next call |

`configs/sleep_controller.json` (schema 3) names the *bookkeeping kinds*
once — formation (`fvg_created`, `swing_confirmed`, `structural_leg_created`,
`liquidity_level_created`, …), touches and level bookkeeping — and uses the
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

## The Main Brain

The system prompt carries the fourteen-step framework (定位 / 推演 / 交互 /
反应 / 对齐 / 评估 / 反向 / 重构 / 预期 / 目标 / 表达 / 证伪 / 风控 / 跟踪)
with the instruction to reason through the steps the situation needs and mark
the rest `"n/a"`, the incremental rule, the hard rules (objects only, no
prices, a counter candle is not delivery, sleep only when nothing is pending)
and the output contract rendered from `LLM_UPDATE_EXAMPLE`. Its sha256 is in
every run's identity.

`LLMInput`: `episode_id`, `known_at`, `trigger`, `bar` (close, 1m ATR),
`session`, `scales` (4H / 1H / 15m / 5m summaries with aliases; 1m structure
and delivery only), `interaction`, `new_evidence` (this bar's evidence, then
the bookkeeping evidence of the TICK bars since the last call, then every
item an incident bar left unjudged, marked `pending_since`, until the LLM
verdicts it), `tape_since_last_update`, `price_relations` (for every aliased object
within `relation_atr_limit` ATRs of the close, plus every object the prior
state names: `position` — where the *object* lies, `above_price` /
`below_price` / `contains_price` — and `offset_atr`, its signed distance
in 1m ATRs, positive above price; renamed on 2026-09-17 from
`relation` / `distance_atr`, which described price's place against the
object and which the model read the other way round on every real run), `prior_state`
(the state without its registry, each evidence list bounded to its last
`prior_evidence_limit` items in a compact shape with notes cut at
`note_limit`, plus counts; pending items are not repeated there; and, since
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
   geometry is coherent for its direction, else it downgrades to `NONE`;
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
when every episode reproduces. `brain/tests/test_replay.py` proves it on a
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
`brain/scripts/summarize_run.py --run-dir <run> [--run-dir …] [--write]`
turns a journal into `summary.json` (calls, tokens and cost at the rates
in `configs/llm_pricing.json`, latency, triggers, sleeps, sharp-move
coverage, opportunities, vetoes and their repeats after the LLM saw them,
the order lifecycle, invariants, the account, timings) and prints several
runs side by side; `--until <UTC time>` bounds the counts to a common
window when runs differ in length. `--broker sim|ibkr` adds the Risk gate and the order machine
([execution/docs/README.md](../../execution/docs/README.md)); the machine's
ledger is the Brain's `PositionLedger`, so an engaged Brain (a position or a
working order) cannot sleep and the idle rule does not fire. The DeepSeek key is read from `DEEPSEEK_API_KEY`, or, when that
is unset, from the key file `brain/configs/deepseek.key` (path overridable
with `DEEPSEEK_API_KEY_FILE`); `*.key` is gitignored, and the key is never
written to a command line the repository owns or to a journal.

## Receipts — `brain/docs/evidence/`

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
| `test_brain_state.py` | round trip, invariants, unknown keys |
| `test_llm_contract.py` | the reply gate: 16 malformed shapes, canonical input hashing |
| `test_opportunity_geometry.py` | entry / stop / target per object kind, LONG / SHORT mirror, incoherence |
| `test_object_registry.py`, `test_eye_view.py` | aliases, causality, price relations, reproducibility on the synthetic Eye; interaction rows (source alias, `last_step_at`, open since the last call) |
| `test_sleep_controller.py` | the wake rule kind by kind, UPDATE / TICK, the tape rule |
| `test_reducer.py` | every verdict route, RESOLVE, missing verdict, pending items (re-offered, verdicted late, resolved by a carrier, kept through another incident), NEUTRAL never blocks sleep, understanding replacement, opportunity downgrade, position, each sleep condition, TICK, incidents, determinism |
| `test_llm_client.py` | retry policy (timeout / 429 / 5xx / dropped connection / 400 / malformed / repair), DeepSeek request shape and error mapping against a local HTTP server, empty content, key resolution (environment, key file, neither) |
| `test_journal.py` | hash chain, tampering, monotone `known_at`, revision gaps, the ledger |
| `test_main_brain.py` | prompt loading, input shape, wake / update / incident steps, bounded prior evidence, pending items re-offered and filed by their late verdict |
| `test_runtime.py` | SLEEP → WAKE → UPDATE → TICK → sleep on the synthetic Eye, a NEUTRAL-only sleeper sleeps, evidence parked by an incident is verdicted later and sleep follows, incident keeps ACTIVE, open position, `known_at` monotone, episode numbering |
| `test_replay.py` | a journal replays to identical states; tampering and a dropped revision are detected |
| `test_summarize_run.py` | the summary of a stack run on the synthetic tape, the cost arithmetic, the veto metrics on a hand-built journal, sharp-move coverage, rendering |
| `test_run_guards.py` | `tape_is_current` for `--broker ibkr`, the effort label, `drive` timings |
| `test_regression_baseline.py` | `research_orchestration`: the frozen week backtest replays and summarizes identically ([evidence/regression_baselines.json](evidence/regression_baselines.json)) |
| `test_eye_link_real_tape.py` | `research_orchestration`: the real Eye on 2022-01-04 wakes the controller, never on 1m alone |
