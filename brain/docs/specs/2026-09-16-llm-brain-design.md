# LLM Trading Brain with a Sleep Controller — design, 2026-09-16

Status: approved in conversation on 2026-09-16; implementation plan in
[../plans/2026-09-16-llm-brain.md](../plans/2026-09-16-llm-brain.md).

## 1. Why

The mechanical Brain (local conditional kNN hypotheses, 2026-09-09 → 09-15)
was measured three times and never beat the driftless rate: the information-gain
gate failed on all eighteen cells, the Setup first-passage gate paid below the
driftless ratio in every cell, and direction alone was a coin flip
(`../evidence/`). The Eye's *facts* are sound — the repairs that followed those
runs are merged into `main` — but no fixed pipeline over them produced a
forecast. The redesign replaces the mechanical Brain with an LLM that reasons
over the Eye's objects on demand, and puts a deterministic gate in front of it
so the LLM is only consulted when the Eye reports something worth reasoning
about.

```text
Market Data ─► Trading Eye (unchanged, per completed 1m bar)
                 └─► MarketObservation
                       └─► Sleep Controller ──WAKE / UPDATE / TICK──► Main LLM Brain
                                                                          └─► BrainState_t+1 ─► (Risk / Execution: later phase)
```

Nothing upstream of `MarketObservation` changes. The Eye's atomic identity
(`144f1d6c…3ee94d`) is untouched.

## 2. Scope

In scope (five deliverables from the brief):

1. Sleep Controller — the wake / continue gate.
2. Main Brain — LLM reasoning over the 14-step framework, on demand.
3. Persistent `BrainState` and the incremental `BrainStateReducer`.
4. Permission isolation — the LLM names Eye object IDs only; geometry, prices
   and R are computed by code.
5. Sleep-exit rules and a replayable, hindsight-free `BrainJournal`.

Out of scope: position sizing, order compliance, hard-stop execution, the Risk
veto engine and Execution. This design defines the boundary they will attach to
(`OpportunityGeometry`, `PositionLedger`) and nothing more.

Test window: 2022-01 (the same tape and the same warm-up rule the retired gates
used). End-to-end run: a bounded slice of 2022-01 against the real DeepSeek API.

## 3. Module map

```text
contract/brain/state.py            BrainState, ActiveExpectation, EvidenceItem, Opportunity, enums
contract/brain/llm.py              LLMInput, LLMUpdate, EvidenceVerdict; JSON schema for the reply
contract/decision/opportunity.py   OpportunityProposal (IDs only) / OpportunityGeometry (prices, R)

shares/core/eye_factory.py         build_eye(model_path, root, audit_journal_dir) — moved from brain/research

brain/core/eye_view.py             MarketObservation → EyeContext (objects, aliases, events, price relations)
brain/core/object_registry.py      alias ↔ Eye entity id, per episode; geometry lookup
brain/core/sleep_controller.py     ControllerDecision from events_this_update + BrainStatus
brain/core/reducer.py              BrainStateReducer.apply(prev, evidence, update, ledger) → ReduceResult
brain/core/llm_client.py           LLMClient protocol; DeepSeekClient; ScriptedClient; RecordedClient
brain/core/main_brain.py           build input → call → parse → reduce; incident handling
brain/core/opportunity_geometry.py resolve_geometry(registry, snapshot, opportunity) → OpportunityGeometry
brain/core/journal.py              BrainJournal writer (hash chain) + JournalReader
brain/core/position_ledger.py      PositionLedger protocol + InMemoryPositionLedger
brain/core/runtime.py              BrainRuntime: the SLEEP ↔ ACTIVE machine, one step per bar

brain/configs/sleep_controller.json
brain/configs/main_brain.json
brain/configs/prompts/main_brain_system.md

brain/scripts/run_llm_brain.py     drive the Eye over a window, write a journal run
brain/scripts/replay_journal.py    rebuild every input from the Eye, verify hashes, re-reduce
```

`brain/core/brain_entry_sequence.py` stays (it is the Brain-side reading of the
Eye's interaction facts and `shares/core/scene_graph.py` depends on it).

## 4. What the LLM sees — `EyeContext`

`eye_view.build_eye_context(observation, registry)` is the only module that
reads Eye types on the Brain side. It publishes, per completed bar:

| field | source | content |
| --- | --- | --- |
| `known_at` | `snapshot.asof` | bar close, UTC |
| `bar` | `snapshot.price`, 1m frame metrics | close, high, low, `atr_1m` |
| `session` | `snapshot.session` | name, phase, elapsed minutes, session high/low, prior-day high/low, overnight high/low |
| `scales[tf]` for 4H, 1H, 15m, 5m, 1m | `timeframe_states[tf]` | structure (external/internal direction, protected high/low with alias, last BOS/MSS direction), delivery (phase, active leg direction, displacement score), range (alias, low, high, location label, normalized location), liquidity (unswept BSL/SSL candidates with alias, price, rank, swept-recently), zones (active FVG / OB with alias, bounds, lifecycle, direction) |
| `interaction` | `observation.interaction_update` via `brain_entry_sequence` | open interaction paths: alias of the object being interacted with, step kind, direction |
| `events` | `observation.events_this_update`, transition kinds at 5m and above | `{evidence_id, kind, timeframe, direction, side, object_id (alias or null), known_at}` — the reducer's evidence |
| `tape` | 1m transition events since the last LLM call | counts per kind, plus the last eight reaction events (`sweep_confirmed`, `acceptance_confirmed`, `mss_core_confirmed`, `qualified_bos`, `liquidity_sweep`, `level_reached`) with their object alias — context only, never evidence |
| `price_relations` | computed | for every published object: its `position` relative to price (`above_price` / `below_price` / `contains_price`) and `offset_atr`, its signed distance in `atr_1m` (positive above price); until 2026-09-17 `relation` / `distance_atr` from price's point of view, which the model inverted |

Rules:

- Every object the LLM may name carries an **alias**, assigned by
  `ObjectRegistry` on first appearance inside the episode and never reused:
  `FVG_5m_3`, `OB_15m_1`, `BSL_1H_2`, `SSL_5m_4`, `DR_15m_1`, `SWING_H_1H_2`,
  `SWING_L_5m_7`. The registry maps alias → `(entity_id, kind, timeframe)` and
  is carried in `BrainState.object_registry`, so a journal is self-describing.
- Aliases are assigned in a deterministic order (timeframe order, then the
  Eye's own tuple order), so a replay reproduces them.
- The Eye's 1m tape events are context, not evidence: they earn no verdict,
  do not trigger an UPDATE, and no 1m object is aliased for opportunities
  (the Eye treats 1m as microstructure).
- Nothing in `EyeContext` is derived from any bar after `known_at`; the builder
  asserts that every timestamp it serializes is `<= known_at`.
- `eye_view.visible_liquidity_ids(observation)` replaces the retired
  `brain.core.risk._visible_level_ids` with the same semantics (VISIBLE
  lifecycle, `confirmed_at <= asof`, ambiguous duplicates dropped); the two
  Eye tests that pinned the downstream visibility contract point at it.

## 5. Sleep Controller

`sleep_controller.decide(observation, status, state) -> ControllerDecision`.

Configuration, `brain/configs/sleep_controller.json`:

```json
{
  "schema_version": 1,
  "wake": {
    "timeframes_any_reaction": ["15m", "1H", "4H"],
    "reaction_kinds_excluded": [
      "swing_confirmed", "structural_leg_created", "liquidity_level_created",
      "structure_break_failed", "raw_boundary_break", "delivery_phase_updated",
      "level_penetrated", "level_touched", "liquidity_retired", "level_invalidated",
      "fvg_created", "base_origin_core_created", "fvg_invalidated",
      "fvg_partially_filled", "fvg_fully_filled", "fvg_midpoint_touched",
      "dealing_range_replaced"
    ],
    "timeframe_specific_kinds": {
      "5m": ["mss_core_confirmed", "qualified_bos", "sweep_confirmed"]
    }
  },
  "evidence": {
    "timeframes": ["5m", "15m", "1H", "4H"],
    "tape_timeframe": "1m",
    "tape_reaction_kinds": ["sweep_confirmed", "acceptance_confirmed", "mss_core_confirmed", "qualified_bos", "liquidity_sweep", "level_reached"],
    "tape_recent_limit": 8,
    "heartbeat_kinds_excluded": ["bar_completed", "market_epoch_reset"],
    "state_republication_suffix": "_state"
  }
}
```

Decision table (evaluated once per completed 1m bar):

| status | condition | decision |
| --- | --- | --- |
| SLEEP | any event with `kind ∈ wake set` for its timeframe | `WAKE` (reason = the matching event ids) |
| SLEEP | otherwise | `STAY_ASLEEP` |
| ACTIVE | any transition event at 5m or above, **or** any `watch_next` / opportunity object whose price relation changed since the last state | `UPDATE` (reason = event ids / object aliases) |
| ACTIVE | otherwise | `TICK` |

Measured on the cached 2022-01 event log: the wake set fires on ≈ 87 bars per
Globex day (≈ 10 %); 5m-and-above transitions, the UPDATE evidence, occur on
≈ 240 bars per day (≈ 27 %), so an ACTIVE Brain is consulted on roughly one
bar in four and ticks through the rest. The controller has no weights, no direction, no price
threshold and no memory beyond "the last published price relations"; the
sleep transition is not its call — the reducer decides it (§ 8).

## 6. BrainState

`contract/brain/state.py`. The JSON from the brief, plus four fields the
machine needs (`schema_version`, `revision`, `object_registry`, `last_update`):

```json
{
  "schema_version": 1,
  "episode_id": "EP_20220104_001",
  "status": "ACTIVE",
  "revision": 3,
  "started_at": "2022-01-04T14:35:00Z",
  "updated_at": "2022-01-04T14:41:00Z",
  "market_understanding": "...",
  "active_expectation": {
    "thesis": "...",
    "expected_next": ["..."],
    "should_not_happen": ["..."]
  },
  "evidence": {
    "supporting":    [{"evidence_id": "ev_…", "known_at": "…", "kind": "sweep_confirmed", "timeframe": "5m", "object_id": "SSL_5m_2", "verdict": "SUPPORT", "note": "…"}],
    "contradicting": [],
    "unresolved":    []
  },
  "watch_next": [{"object_id": "FVG_5m_3", "question": "…"}],
  "destination_candidates": ["BSL_1H_1"],
  "opportunity": {
    "state": "NONE",
    "direction": null,
    "entry_object_id": null,
    "invalidation_object_id": null,
    "target_object_id": null
  },
  "reasoning_confidence": "LOW",
  "continue_active": true,
  "object_registry": {"FVG_5m_3": {"entity_id": "…24 hex…", "kind": "fvg", "timeframe": "5m"}},
  "last_update": {"known_at": "…", "llm_called": true, "verdicts": {"SUPPORT": 1, "CONTRADICT": 0, "NEUTRAL": 0, "RESOLVE": 0}, "incident": null}
}
```

Enums: `status ∈ {ACTIVE, ARCHIVED}` (a SLEEPing Brain has no state — the
runtime's status is SLEEP and the last archived state is on disk);
`opportunity.state ∈ {NONE, DEVELOPING, ACTIONABLE}`; `direction ∈ {LONG,
SHORT, null}`; `reasoning_confidence ∈ {LOW, MEDIUM, HIGH}`; `verdict ∈
{SUPPORT, CONTRADICT, NEUTRAL, RESOLVE}`.

Invariants enforced in `__post_init__`: `opportunity.state != NONE` ⇒ all three
object ids present and in `object_registry`; `direction` present iff
`state != NONE`; every evidence id unique across the three lists; timestamps
UTC-aware; `updated_at >= started_at`.

Serialization: `to_json()` / `from_json()` with sorted keys and no floats
outside evidence notes, so a state hashes stably.

## 7. LLM input and output

### 7.1 `LLMInput` (`contract/brain/llm.py`)

```json
{
  "schema_version": 1,
  "episode_id": "EP_20220104_001",
  "known_at": "2022-01-04T14:41:00Z",
  "trigger": {"kind": "WAKE | UPDATE", "reasons": ["ev_…", "FVG_5m_3"]},
  "bar": {"close": 0.0, "high": 0.0, "low": 0.0, "atr_1m": 0.0},
  "session": {…},
  "scales": {"4H": {…}, "1H": {…}, "15m": {…}, "5m": {…}, "1m": {…}},
  "interaction": [...],
  "new_evidence": [{"evidence_id": "ev_…", "kind": "…", "timeframe": "5m", "direction": "up", "side": null, "object_id": "SSL_5m_2", "known_at": "…"}],
  "tape_since_last_update": {"bars": 3, "counts": {"level_touched": 4}, "recent": [{"kind": "sweep_confirmed", "object_id": "SSL_1m_9", "known_at": "…"}]},
  "price_relations": [{"object_id": "FVG_5m_3", "position": "above_price", "offset_atr": 1.4}],
  "prior_state": { …BrainState or null on WAKE… }
}
```

Amended 2026-09-17: `price_relations` rows are `{object_id, position,
offset_atr}` — the *object's* place relative to price (`above_price` /
`below_price` / `contains_price`, offset signed the same way). The earlier
`{relation: above|inside|below, distance_atr}` described where *price* sat
against the object; every real run read it the other way round, so every
stop and target landed on the wrong side and died as
`opportunity_incoherent` (25 of 25 proposals on the 2022-01-03 RTH day at
low effort, 11 of 11 in run 3). `prior_state` carries no `object_registry`, its
evidence lists are bounded (`prior_evidence_limit`, `note_limit`), and it
gains `execution` — the order machine's view of the last opportunity
(status, working order, position, last outcome, last Risk veto, as aliases
and counts) so the LLM knows a veto, an order or a position exists; design
in [2026-09-17-veto-feedback-week-backtest-design.md](2026-09-17-veto-feedback-week-backtest-design.md).

The system prompt (`brain/configs/prompts/main_brain_system.md`) carries the
14-step framework, the output contract, and the rule that the model reasons
through the steps it needs rather than filling a form. It contains the word
"json" and an example reply (DeepSeek's JSON mode requires both). Its sha256
is part of the run identity.

### 7.2 `LLMUpdate` (the reply, strict JSON)

```json
{
  "evidence_verdicts": [{"evidence_id": "ev_…", "verdict": "SUPPORT", "note": "…", "resolves_evidence_id": null, "resolution": null}],
  "understanding_holds": true,
  "market_understanding": "…",
  "active_expectation": {"thesis": "…", "expected_next": ["…"], "should_not_happen": ["…"]},
  "watch_next": [{"object_id": "FVG_5m_3", "question": "…"}],
  "destination_candidates": ["BSL_1H_1"],
  "opportunity": {"state": "NONE", "direction": null, "entry_object_id": null, "invalidation_object_id": null, "target_object_id": null},
  "reasoning_confidence": "LOW",
  "continue_active": true,
  "framework_trace": {"step_1": "…", "step_2": "…", "…": "…", "step_14": "n/a"}
}
```

`parse_update(text) -> LLMUpdate` validates against the schema: unknown keys
rejected, enums checked, every `evidence_id` must be in `new_evidence`, every
`object_id` must be an alias the input published. A reply that fails is a
`MalformedReply` incident (§ 9); the LLM is never asked to emit a price.

Amended 2026-09-17 (thesis lifecycle): `opportunity` also carries
`thesis_id`, `governing_timeframe` (`4H` / `1H` / `15m` / `5m`), `grade`
(`BASE` / `A_PLUS`) and `invalidation_mode` (`TOUCH` / `CLOSE_BEYOND`) —
all `null` when the state is `NONE`, required otherwise. The reducer's
rule 4 refuses an invalidation object more than one scale below the
governing one (`opportunity_invalidation_scale`) and a thesis id that flips
direction (`thesis_direction_changed`); `resolve_geometry` puts a
`CLOSE_BEYOND` hard stop one 1m ATR × √(scale minutes) beyond the object.
The controller (schema 3) remembers watched relations only on
`relation_change_timeframes`. Design:
[../../../execution/docs/specs/2026-09-17-thesis-lifecycle-risk-v2-design.md](../../../execution/docs/specs/2026-09-17-thesis-lifecycle-risk-v2-design.md).

### 7.3 DeepSeek client

`DeepSeekClient(model="deepseek-flash", base_url=env DEEPSEEK_BASE_URL or
"https://api.deepseek.com", api_key=env DEEPSEEK_API_KEY or the gitignored brain/configs/deepseek.key, timeout_s, max_retries)`.
`POST {base_url}/chat/completions` with `response_format={"type":"json_object"}`,
`messages=[system, user(LLMInput as JSON)]`, `max_tokens` from config. Raw
HTTP through `urllib.request` — no new dependency. The reply's
`choices[0].message.content` is the update; `reasoning_content` (thinking
mode) is journaled, never parsed. The key is read from the environment only;
a missing key raises `LLMClientError` at construction, before any bar is read.

Error mapping: socket / HTTP timeout → `LLMTimeout`; HTTP 429 → `LLMRateLimited`
with `retry_after` from the header (default 5 s); HTTP 5xx → `LLMServerError`;
HTTP 4xx other → `LLMRequestRejected` (no retry); empty `content` or
non-JSON → `MalformedReply`. Retries: timeouts, 429 and 5xx are retried up to
`max_retries` with exponential backoff (429 honours `retry_after`); a
`MalformedReply` gets exactly one repair attempt (the same request plus the
parser's error message appended to the user turn); everything past that
becomes an incident.

`ScriptedClient(replies)` returns canned replies / raises canned errors in
order (tests). `RecordedClient(journal)` answers every input with the journal's
recorded reply keyed by `input_sha` and raises if an input is not in the
record (replay).

## 8. BrainStateReducer

`reducer.apply(prev: BrainState | None, evidence: tuple[EvidenceItem, ...],
update: LLMUpdate | None, *, known_at, ledger, registry) -> ReduceResult
(state, rejections, slept)`.

Rules, in order:

1. **Bookkeeping**: `revision = prev.revision + 1` (0 on WAKE), `updated_at =
   known_at`, `episode_id` / `started_at` unchanged, `object_registry` = the
   registry after this bar.
2. **Evidence**: every item in `evidence` must receive one verdict. SUPPORT /
   CONTRADICT / NEUTRAL append the item (with note) to `supporting` /
   `contradicting` / `unresolved` respectively; NEUTRAL items are recorded so
   they can be RESOLVEd later. RESOLVE with `resolves_evidence_id` removes that
   id from `unresolved` and appends the new item to `supporting` or
   `contradicting` according to `resolution`, which a RESOLVE verdict must
   carry as `SUPPORT` or `CONTRADICT` (null on every other verdict).
   Evidence without a verdict goes to `unresolved` and is a rejection; such
   an item is *pending* (`verdict` null) and the Main Brain re-offers it in
   `new_evidence` (marked `pending_since`) on every later call until the LLM
   verdicts it — a pending item is then filed by that verdict, never rejected
   as a duplicate (amended 2026-09-16 after the first real run: an incident
   bar's evidence could otherwise only leave through a RESOLVE carried by a
   new item, and 112 items accumulated in one session).
3. **Understanding**: if `understanding_holds` is false, `market_understanding`
   and `active_expectation.thesis` must both differ from `prev`; otherwise the
   update is rejected as a whole (state carried forward, incident
   `understanding_not_replaced`). If true, the update may still edit
   `expected_next` / `should_not_happen` / `watch_next` freely.
4. **Opportunity**: `state != NONE` requires all three ids to be registry
   aliases that are visible on this bar, distinct, and geometrically coherent
   for `direction` (LONG: target above entry, invalidation below entry;
   SHORT: mirrored) as resolved by `opportunity_geometry`. Any failure
   downgrades to `NONE` with a rejection naming the rule.
5. **Position**: `ledger.has_open_position()` ⇒ `continue_active = true`
   regardless of the reply.
6. **Sleep eligibility**: the reply's `continue_active = false` is honoured only
   when all five hold — no open position; no open interaction path — one
   whose latest step was observed after the previous LLM call, or on the
   wake bar itself (amended 2026-09-16: the Eye keeps a path ACTIVE for as
   long as its context lives, and every bar of the real tape carried one, so
   the original "no ACTIVE path" reading made sleep unreachable; the rows
   now name the path's source object and carry `last_step_at` /
   `stepped_since_last_call`); no *pending* item in `unresolved` (a `NEUTRAL`
   item is judged and never blocks sleep — amended 2026-09-16: 74 NEUTRAL
   verdicts in one session had made sleep unreachable); `watch_next` empty;
   `opportunity.state == NONE`. If the reply asked to sleep but a condition
   fails, `continue_active` is forced true and a rejection
   `sleep_refused:<condition>` is recorded. `slept` is true only when the
   final `continue_active` is false; the runtime then archives.
7. **TICK** (`update is None`): revision advances, `updated_at` advances,
   `last_update.llm_called = false`, nothing else changes.
8. **Idle archive** (added 2026-09-16 after the second real run, in which
   the model never set `continue_active` false): `ReduceContext` carries
   `idle_updates` (the runtime's count of consecutive accepted updates with
   no opportunity and an unchanged understanding) and `idle_archive_after`
   (from `sleep_controller.json`). An accepted update that keeps the
   understanding and proposes no opportunity, with `idle_updates + 1` at the
   threshold, sleeps with `sleep_reason = "idle"` unless a position is open
   or an item is still unjudged; `watch_next` and the interaction gate do
   not hold an idle episode open.

The reducer is a pure function of its arguments; the same journal replays to
the same states.

## 9. Runtime: SLEEP ↔ ACTIVE

```text
                      ┌─────────────────────────── TICK (no new evidence; no LLM) ──┐
                      │                                                             │
   ┌───────┐  WAKE    ▼   ┌────────┐  UPDATE → LLM → reduce                         │
   │ SLEEP │ ───────────► │ ACTIVE │ ◄──────────────────────────────────────────────┘
   └───────┘             └────────┘
       ▲                      │ reduce.slept (five conditions hold, continue_active=false)
       └──── archive ◄────────┘
            (state.status = ARCHIVED, journal `sleep`, runtime status = SLEEP)

   ACTIVE + LLM failure after retries ─► incident record, state carried forward (TICK-shaped), still ACTIVE
   ACTIVE + open position ─► continue_active forced true (reducer rule 5)
```

`BrainRuntime.step(observation) -> StepResult` performs exactly one of the
five transitions per bar and returns what it did. Episode ids are
`EP_<YYYYMMDD>_<NNN>` numbered per session day in the run.

## 10. Permission boundary

- The LLM's writable surface is `LLMUpdate` (§ 7.2). It has no numeric price
  field; `object_id`s are aliases and are validated against the registry.
- `opportunity_geometry.resolve_geometry(registry, snapshot, opportunity) ->
  OpportunityGeometry(entry_price, stop_price, target_price, reward_risk,
  rule_ids)` is deterministic code over Eye geometry:
  - entry: FVG / OB → the edge nearest the current price (LONG: `upper_bound`,
    SHORT: `lower_bound`); dealing range → `value_price`; liquidity pool →
    `midpoint`; swing → `price`.
  - stop: the invalidation object's far edge plus one tick (LONG: `lower_bound
    − tick`; SHORT: `upper_bound + tick`); for a swing, its price ± tick.
  - target: the target object's near edge (LONG: `lower_bound`; SHORT:
    `upper_bound`); for a pool, `midpoint`; for a swing, its price.
  - `reward_risk = |target − entry| / |entry − stop|`; a zero or negative risk
    distance is a resolution failure (the reducer downgrades to NONE).
- `PositionLedger` protocol: `has_open_position() -> bool`, `open_positions()
  -> tuple[PositionRecord, ...]`. `InMemoryPositionLedger` is the only
  implementation now; Execution will provide the real one. Hard stops and
  sizing are explicitly not here.

## 11. BrainJournal

Layout: `outputs/brain_journal/<run_id>/run.json`, `index.jsonl`,
`episodes/<episode_id>.jsonl`.

`run.json`: model, base url host, `system_prompt_sha256`,
`sleep_controller_sha256`, `main_brain_config_sha256`, source parquet sha,
window (`warmup_start`, `emit_start`, `end`), Eye `atomic_definition_identity`
and `scale_registry_id`, git revision (+ dirty flag), started/finished
timestamps. `run_id` = first 16 hex of the sha256 over those identity fields.

Record envelope (every line): `{seq, record, episode_id, known_at, prev_hash,
hash, payload}` where `hash = sha256(prev_hash + canonical(payload))` and the
first record of an episode chains from the run's `run_id`. Record kinds, in
the chain order an episode follows:

| record | payload |
| --- | --- |
| `episode_opened` | episode_id, known_at |
| `wake` | controller reasons, aliases published |
| `llm_call` | `input_sha`, the full `LLMInput`, raw reply text, `reasoning_content`, usage, latency ms, attempt count |
| `state` | the full `BrainState` after reduce, rejections |
| `tick` | known_at only |
| `opportunity` | the `Opportunity` and its `OpportunityGeometry` when state becomes DEVELOPING / ACTIONABLE or changes |
| `trade` | reserved for ledger events (open / close); written by the ledger hook, none in this phase |
| `incident` | kind, message, attempts, the request that failed (input_sha) |
| `sleep` | archived state summary, revisions, reason |

The writer refuses a record whose `known_at` precedes the previous record's,
and refuses a `state` record whose `revision` is not `previous + 1`.
`index.jsonl` lists every episode with first/last `known_at`, revisions and
outcome.

Hindsight guard: the journal never stores anything the Eye had not published
by `known_at`; `llm_call.input` is the exact bytes sent, and the replay tool
rebuilds it from the Eye and compares `input_sha`.

## 12. Causality and replay

- `iter_completed_bars` drives the Eye strictly forward; `EyeContext` is built
  from the observation of the current bar only.
- `eye_view` asserts every serialized timestamp `<= known_at`; the runtime
  asserts `known_at` strictly increases across steps.
- `replay_journal.py --run <run_id>` re-runs the Eye over the run's window with
  the same warm-up, re-derives every `LLMInput` where the journal has an
  `llm_call`, checks `input_sha` byte-equality, feeds the recorded replies
  through `RecordedClient` and the reducer, and asserts every `state` record's
  hash matches. Output: a verdict per episode and a non-zero exit on the first
  mismatch.

## 13. Cleanup (retired with this design)

Deleted:

- `brain/core/{forecast, hypothesis_proposer, hypothesis_pool, belief_updater, trajectory, decision, risk}.py`, `brain/configs/hypothesis_protocol.json`
- `brain/research/` (all), `brain/scripts/` (all; `build_eye` moves to `shares/core/eye_factory.py`)
- `brain/tests/` (all sixteen retired modules)
- `brain/docs/plans/*` and `brain/docs/specs/*` written for the retired gates
- `contract/brain/forecast.py` (the kNN Brain's `MarketBeliefState` and its representation constants)
- `shares/core/engine.py`; `shares/tests/{test_io_engine, test_scene_graph_scale_contract, test_v2_protocols}.py`; `execution/core/simulation.py`; `execution/tests/test_sequential_replay.py` — every one bound to the retired typed vertical and uncollectable today
- `configs/model.json`: the `hypothesis_protocol`, `action_pipeline`, `decision` and `risk` blocks (read only by deleted modules; not part of the atomic identity, verified by the existing `eyes/tests/test_semantic_selection.py`)

Kept: `brain/core/brain_entry_sequence.py`; `contract/brain/{belief, context, hypothesis, plan, vocabulary}.py`, `contract/decision/action.py`, `contract/risk/` and `contract/research/` — the typed-vertical contracts that `shares/core/scene_graph.py`, `shares/core/visualization.py`, `market_cases.py` and the shares test helpers still consume as inert dataclasses (measured during Task 3: deleting them cascades into the visualization projection and four passing test modules). `CONTRACT_ORDER` is unchanged; `contract/decision/opportunity.py` sits beside `action.py`. `brain/docs/evidence/*` stay (historical, with a retired banner). `eyes/tests/test_semantic_selection.py`, skipped since the engine's retirement, is revived on `shares.core.eye_factory.build_eye` with the current identity.

Re-bound: `eyes/scripts/replay_hash_stream.py` → `shares.core.eye_factory.build_eye`; `eyes/tests/test_v3_group12_primitives.py` and `test_range_auction_primitives.py` → `brain.core.eye_view.visible_liquidity_ids`.

Left as found (pre-existing breakage, outside this design):
`shares/scripts/audit_market_clock.py` and
`execution/scripts/materialize_mbo_execution.py` import the retired
`brain.core.validation`.

## 14. Tests

| area | cases |
| --- | --- |
| controller | 1m-only events do not wake; formation-class 15m events do not wake; a 15m `sweep_confirmed` wakes; a 5m `qualified_bos` wakes; a 5m `fvg_created` does not; ACTIVE with no transition and unchanged relations → TICK; ACTIVE with a watched object's relation change → UPDATE |
| reducer | each verdict routes correctly; RESOLVE clears an unresolved id; missing verdict → unresolved + rejection; `understanding_holds=false` without a new thesis → whole update rejected; unknown alias → opportunity NONE; incoherent geometry → NONE; open position forces ACTIVE; each of the five sleep conditions individually refuses sleep; TICK advances revision only; determinism (same inputs → identical JSON) |
| state | round-trip JSON; invariants raise on malformed states |
| llm client | ScriptedClient errors: timeout retried then incident; 429 honours `retry-after` then succeeds; 5xx retried; 400 not retried; malformed JSON repaired once then incident; empty content is malformed; DeepSeekClient request body shape and header (against a local fake HTTP server); missing key refuses construction |
| geometry | entry/stop/target per object kind; LONG/SHORT mirror; zero risk distance fails |
| journal | hash chain verifies; non-monotone `known_at` refused; revision gap refused; reader reproduces records |
| causality | `EyeContext` timestamps `<= known_at` on a real 2022-01-04 slice; journal replay of a ScriptedClient run reproduces every state hash |
| runtime | SLEEP → WAKE → UPDATE → TICK → sleep on a scripted episode; incident keeps ACTIVE; episode ids number per day |
| real Eye link | `build_eye` from `configs/model.json` over 2022-01-03 → 2022-01-04 (warm-up 2021-12-27) drives the controller: at least one WAKE occurs, no 1m-only wake, `visible_liquidity_ids` equals the retired helper's answer on the same snapshots |

## 15. End-to-end run

`brain/scripts/run_llm_brain.py --emit-start 2022-01-04 --end 2022-01-05
--warmup-start 2021-12-27 --model deepseek-flash --max-llm-calls N` writes
`outputs/brain_journal/<run_id>/`. The report includes one complete episode:
`episode_opened → wake → llm_call → state → … → sleep`.

## 16. Documentation to update

`AGENTS.md` (architecture tree, package table, test commands), `brain/docs/README.md`
(rewritten for this design), `shares/docs/current_implementation_status.md`
(dated section), `contract/README.md` (layer list), `brain/docs/evidence/*`
(retired banner).
