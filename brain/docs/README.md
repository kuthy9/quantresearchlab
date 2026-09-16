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
| `eye_view.py` | `build_eye_context`: the aliased, JSON-ready view of one observation (objects per scale, structure / delivery / range / liquidity summaries, session, open interaction paths, this bar's evidence events, price relations); `assert_causal`; `visible_liquidity_ids` |
| `object_registry.py` | `ObjectRegistry`: `FVG_5m_3` ↔ Eye entity id, assigned on first appearance inside an episode in a deterministic order |
| `sleep_controller.py` | `decide(events, active, config)` → `WAKE` / `STAY_ASLEEP` / `UPDATE` / `TICK`; `ControllerConfig` from `configs/sleep_controller.json` |
| `main_brain.py` | `MainBrain.step`: build the `LLMInput`, prove it causal, call under the retry policy, parse, reduce; `MainBrainConfig` from `configs/main_brain.json` + `configs/prompts/main_brain_system.md` |
| `llm_client.py` | `DeepSeekClient` (urllib, JSON mode, key from `DEEPSEEK_API_KEY`), `ScriptedClient`, `EchoClient`, `RecordedClient`, `call_with_policy` |
| `reducer.py` | the pure function `apply(prev, evidence, update, ctx)`; `empty_state`; `sleep_blockers` |
| `opportunity_geometry.py` | `resolve_geometry` / `coherence_error`: object aliases → `OpportunityGeometry` |
| `position_ledger.py` | `PositionLedger` protocol; `InMemoryPositionLedger` |
| `journal.py` | `BrainJournal` (writer), `JournalReader`, `record_hash` |
| `runtime.py` | `BrainRuntime.step(observation)`: the SLEEP ↔ ACTIVE machine |
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
| ACTIVE | a transition at 5m or above, **or** a watched object's price relation (above / inside / below) changed since the last LLM call | `UPDATE` |
| ACTIVE | otherwise | `TICK` — no LLM call; the state's revision and `updated_at` advance |

Wake set (`configs/sleep_controller.json`): any non-formation transition on
15m / 1H / 4H, plus `mss_core_confirmed`, `qualified_bos` and
`sweep_confirmed` on 5m. Measured on the cached 2022-01 event log: ≈ 87 wake
bars per Globex day (≈ 10 %); 5m-and-above transitions, the UPDATE evidence,
on ≈ 240 bars per day (≈ 27 %). The 1m tape is context (counts per kind and the
last eight reaction events since the previous LLM call), never evidence. The
controller has no weights, no direction and no price threshold; going back to
sleep is the reducer's decision.

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
and delivery only), `interaction`, `new_evidence`, `tape_since_last_update`,
`price_relations` (relation and signed distance in ATR for every aliased
object), `prior_state` (the full state with the evidence ledger bounded to its
last `prior_evidence_limit` supporting / contradicting items plus counts).

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
from `configs/main_brain.json` (`deepseek-flash`); `reasoning_content` is
journaled, never parsed. Retry policy: timeouts, 429 (honouring
`Retry-After`) and 5xx retry with exponential backoff up to `max_retries`; a
malformed reply gets exactly one repair attempt; anything past that is an
*incident* — the state carries forward and the runtime stays ACTIVE.

## The reducer

`apply` is a pure function of the prior state, this bar's evidence, the
update and a `ReduceContext` (known_at, open position, open interaction,
visible aliases, registry, geometry coherence):

1. bookkeeping — `revision + 1`, `updated_at = known_at`, registry merged;
2. evidence — every item gets its verdict's list; an item without a verdict
   lands in `unresolved` with a rejection; an item already in the ledger is
   skipped with `evidence_duplicate`; `RESOLVE` removes its target from
   `unresolved` and files the new item by `resolution`;
3. an abandoned understanding must be replaced, else the whole update is
   refused (`understanding_not_replaced`);
4. an opportunity must name three distinct, currently visible aliases whose
   geometry is coherent for its direction, else it downgrades to `NONE`;
5. an open position forces `continue_active`;
6. sleep is granted only when all five exit conditions hold — no open
   position, no open interaction path, no unresolved evidence, nothing in
   `watch_next`, `opportunity.state == NONE`; a refused sleep is recorded as
   `sleep_refused:<condition>`.

## The runtime

```text
SLEEP  ──WAKE──►  ACTIVE  (new episode EP_<YYYYMMDD>_<NNN>, new registry, revision 0)
ACTIVE ──UPDATE──► LLM → reduce ──slept──► archive (status ARCHIVED, journal `sleep`) → SLEEP
ACTIVE ──TICK──►  revision + 1, no LLM
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
reasoning_content, usage, latency, attempts) → state (the full BrainState,
rejections) | tick (revision) → opportunity (opportunity + geometry) →
incident → sleep`. `trade` is reserved for the ledger. The writer refuses a
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
smoke runs. `DEEPSEEK_API_KEY` is read from the environment only; it is never
written to a file, a command line the repository owns, or a journal.

## Tests — `brain/tests/`

| file | covers |
| --- | --- |
| `test_brain_state.py` | round trip, invariants, unknown keys |
| `test_llm_contract.py` | the reply gate: 16 malformed shapes, canonical input hashing |
| `test_opportunity_geometry.py` | entry / stop / target per object kind, LONG / SHORT mirror, incoherence |
| `test_object_registry.py`, `test_eye_view.py` | aliases, causality, price relations, reproducibility on the synthetic Eye |
| `test_sleep_controller.py` | the wake rule kind by kind, UPDATE / TICK, the tape rule |
| `test_reducer.py` | every verdict route, RESOLVE, missing verdict, understanding replacement, opportunity downgrade, position, each sleep condition, TICK, incidents, determinism |
| `test_llm_client.py` | retry policy (timeout / 429 / 5xx / 400 / malformed / repair), DeepSeek request shape and error mapping against a local HTTP server, empty content, missing key |
| `test_journal.py` | hash chain, tampering, monotone `known_at`, revision gaps, the ledger |
| `test_main_brain.py` | prompt loading, input shape, wake / update / incident steps, bounded prior evidence |
| `test_runtime.py` | SLEEP → WAKE → UPDATE → TICK → sleep on the synthetic Eye, refused sleep, incident keeps ACTIVE, open position, `known_at` monotone, episode numbering |
| `test_replay.py` | a journal replays to identical states; tampering and a dropped revision are detected |
| `test_eye_link_real_tape.py` | `research_orchestration`: the real Eye on 2022-01-04 wakes the controller, never on 1m alone |
