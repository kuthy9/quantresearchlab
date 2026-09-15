# Trading Eye

The Eye normalizes bars, detects the registered semantic primitives, emits
canonical events, stores the complete atomic history and reduces it to one
current market view. It imports no downstream module — no `eyes/core/` module
imports `brain`, `execution`, or the orchestration half of `shares`, and
`eyes/tests/test_eye_module_boundary.py` enforces that.

## Documents here

- [smc_semantic_specification_v1.3.md](smc_semantic_specification_v1.3.md) —
  the registered atomic semantic definitions (`smc_semantics_v1.3`).
- [canonical_semantic_foundation_v2.1.md](canonical_semantic_foundation_v2.1.md) —
  the additive Foundation projection (`smc_semantic_foundation_v2.1`), which
  declares v1.3 as its parent. This pairing is not a unified full-stack v2.
- [evidence/](evidence/) — bounded-study receipts for the v1.3 reading, the
  balance-range candidate hypotheses and the structural range-width strata.

## Where the Eye's own inputs live

The Eye's protocol files are the eight the atomic semantic identity hashes, so
they stay at the repository root in `configs/`: `model.json`,
`data_splits.json`, and `primitives_structure_liquidity.json`,
`primitives_displacement.json`, `primitives_zones.json`, `primitives_range.json`,
`primitives_entry.json`, `primitives_interaction.json`. Moving them would change
`atomic_definition_identity`, because that identity hashes the reference strings
as well as the file contents.

`semantics/` stays at the repository root for the same reason, by a different
route: `registry_v1_3.yaml` names its parameters file by the root-relative string
`semantics/parameters_v1_3.yaml`, and the registry's own bytes are
`registry_sha256`. Relocating the directory rewrites that line and moves the
identity to `fb7d604d…3b75`. See the provenance-mapping note in
[AGENTS.md](../../AGENTS.md).

## The cost of one bar

Driving the Eye degraded within a single run: over 2022-02-01 the same bar cost
10.0 ms at bar 200 and 71.4 ms at bar 1,800. The cause was not memory or GC —
it was the liquidity candidate collection, which is effectively append-only by
design. A sweep disarms a level and keeps it so a later re-approach is
recognisably the same level, so over 2022-02 the Eye created 15,000 levels and
retired 84.

`reduce_timeframe_state` folded the snapshot-derived candidate projections —
IRL/ERL membership, `normalized_location_in_range`, and the Swing `rank` — back
into the stored candidate at the tail of *every* reduced event. Each fold
walked the whole collection and re-sorted it, so per-bar cost was linear in
accumulated history and a run's total cost quadratic in its length. Measured
over 1,400 bars: `_liquidity_state` ran 27.1 times per bar for 18.6% of
runtime, and `dataclasses.replace` was 34.8%, three quarters of it rebuilding
candidates.

The registry binds those fields as "snapshot-derived only … no canonical
membership event is claimed", so they now run where a hierarchy is
materialized. `_settled_candidate_state` is the single choke point, used by
`MarketSnapshotPublisher` and by `replay_atomic_market_snapshot` alike, so a
checkpoint-restored view and a log-rebuilt one cannot disagree about a field
the reducer no longer folds in.

Measured on the same window, per-bar cost is `15.18 + 0.1529·N` ms before and
`15.17 + 0.1074·N` ms after, where `N` is the candidate population: the fixed
term is untouched and the per-candidate term falls 30%. A 5,160-bar replay of
2022-02-01→05 went from 455.7 s to 348.4 s and emitted the identical event
store — same 67,026 events, same fingerprint `f18ce950…ee78f`, zero differing
keys across all fifteen census sections.

**This lowers the constant; it does not change the asymptotics.** The
collection still grows without bound, so per-bar cost is still linear in
history. Bounding the population — the only asymptotic fix — would stop an
evicted level from ever re-arming and so would change the event stream.

`reduce_timeframe_state` is exported, and its contract moved with this change:
a `TimeframeState` it returns no longer carries those projections. A direct
caller that reads `range_role`, `normalized_location_in_range` or candidate
`rank` must settle the state first.

## One crossing, one touch

Two emitters published the same crossing. The structure frame's swing-break
path publishes `LEVEL_TOUCHED` / `LEVEL_PENETRATED` when a confirmed swing
breaks; the inventory-crossing path publishes them when the 1m candle crosses
a candidate level. Each kept its own dedup registry, so a 1m swing broken on
the bar that crossed it was published twice — identical facts, distinct
`event_id`. Over 2022 that was 9.9 % of all events; on 2022-02-01 alone, 246
touches and 246 penetrations. The registry binds `liquidity_sweep` as "first
emit LEVEL_TOUCHED and LEVEL_PENETRATED for one frozen level and
crossing_generation_id", so this was a violation, not a design.

The two paths now share the `(level_id, clock)` touch registry and the
`(level_id, 1m, clock)` penetration registry whenever they speak the same
clock: whichever path sees the crossing first publishes it, the other reuses
the id. A higher-timeframe break cites a different bar and stays its own
crossing generation — the store refuses a 1m penetration whose touch cites a
5m bar, and rightly so. Same-fact duplicates on 2022-02-01: 492 → 0.

Two levels at one price are two entities, not a duplicate. Every crossing
event now cites its level in `source_entity_ids` — the provenance namespace
for entities; `entity_id` is reserved for typed lifecycle transports and the
memory would try to build a timeline for it — so a consumer can tell them
apart without parsing `details.level_id`.

## One formation clock

Every `MarketEvent` carries `formed_at` and `known_at`. The legacy lifecycle
transports filled `formed_at`; the canonical atomic emitter carried the same
clock only as `event_time` and left `formed_at` empty (41 % of events had it,
none of them atomic), so a consumer had to know an event's origin to find when
its fact formed. `_append_semantic_atomic` now fills `formed_at` from the
clock it already carries as `event_time`. The confirmation lag is real and
visible on that pair: on 2022-02-01, 29 % of events formed before the bar they
became known on — a 1m swing two bars, a 5m swing three bars (15 min), a
15m swing 45 min, a 1H swing 3 h.

The candidate state was already its own channel: a swing whose pivot has
completed but whose confirmation bars have not closed is published in
`frames[tf].swings` as `forming` with no `confirmed_at`, and the same identity
is published again as `confirmed` when it is. `test_one_formation_clock.py`
guards that channel; it is not a lagged confirmation re-labelled.

## Target outcomes on the target's own timeframe

The Eye enumerates candidate targets — every `LiquidityInventoryItem`, with
`formed_at` / `confirmed_at` and a lifecycle — and published what happened to
them only as 1m crossing events (`LEVEL_TOUCHED` on `timeframe=1m`, the
level's own scale buried in `details.source_timeframe`) plus the legacy
`LIQUIDITY_RETIRED` transport, six times in four days. "Was the 5m target
reached before the 15m one" had to be re-derived from the tape; the
information-gain gate computed its first-passage targets from future highs
and lows for exactly that reason.

Every inventory item now ends in one of two atomic outcomes, registered under
the `target_outcome` concept and published on the level's own timeframe with
the level cited in `source_entity_ids`: `LEVEL_REACHED` beside the first
`LEVEL_TOUCHED` of the item (citing the admission and that touch, known when
the touch is known), and `LEVEL_INVALIDATED` when the item leaves the
candidate set untouched (citing the admission and the last completed bar,
with the removal reason — `reference_period_replaced` today; candidate
retirement adds its own reasons). The two are exclusive per item. FVG and
order-block outcomes already lived on their own timeframe.

The registry bytes changed, so `atomic_definition_identity` moved from
`f92b24c8…1f0c` and, after the later protocol and registry changes in this series, settled at `29cc2ba1…500a`, then at `144f1d6c…e94d` when the Group 5 terminal-context, liquidity and Group 3 terminal-state retention parameters were registered; `configs/model.json` pins the current value.

## The 1m tape is microstructure

On 2022-02-01→05 the 1m scale was 83.8 % of everything the Eye published and
86.0 % of its transition events (`bar_completed` and `*_state` excluded); 48 %
of the store was legacy `*_state` re-publication. The information-gain gate
had to define its own clocks (≥5m, ≥15m) to see past it.

`MarketObservation` (schema 6) now names the scales that form its main event
clock in `published_timeframes` — every active scale except 1m unless
`ObserverConfig.published_timeframes` says otherwise.
`semantic_events_this_update` carries only those; the 1m events of the update
move to `microstructure_events_this_update`, the channel a trigger reads;
`events_this_update` is the union, and the two channels partition it — nothing
is dropped, and the event store, the reducers and `recent_events` (the
memory's window, which the risk engine reads for 1m sweeps) keep every scale.
This had to wait for the crossing and outcome repairs above: until then a 5m
level's touch was itself a 1m event.

## A primitive is a definition, not a timeframe role

Displacement, the FVG / order-block zones and the dealing range each ran on
one scale fixed in their protocol file (5m, 5m, 1H) and repeated in the
tracker, the entity validators, the emitter's recorder and the store's
contract. On the 2022 gate dataset twelve of the Brain's 150 features were
NaN on every row for that reason alone: `{1m,15m,1H,4H}_displacement_score`
and `{1m,5m,15m,4H}_range_{width_atr,location}`. (The `*_dist_prot_*`
features at 41–83 % NaN are a different thing — a protected swing exists only
after a qualified BOS — and are left alone.)

Structure and liquidity already ran one tracker per scale. Displacement and
zones now do the same: `primitives_displacement.json` and
`primitives_zones.json` name the scales they publish on (`timeframes`:
5m, 15m, 1H, 4H), one tracker runs per active scale, every observation,
transition and state carries its timeframe, the validators use the scale's
own bar length (`Timeframe.minutes`), and a regular completed bar is one whose
whole scheduled span was observed, so a session-tail 4H block counts. The
protocol's first scale stays the one `MarketObservation.displacement`, the
typed transition delta channel and the Group 5 entry contract read; the other
scales publish their facts and their frames the same way. On 2022-02-01 the
5m facts are unchanged in every kind, and 15m / 1H gained 35 / 3
displacements, 14 / 3 FVGs and 10 / 1 base origin cores.

**The dealing range followed on 2026-09-14, under the same semantic
version.** Group 4 fuses the range with the 1m manipulation machinery, and its
range parameters are written in bars (`candidate_real_h1_bars` 8–24,
`compression_*_real_h1_bars`); the decision was to read those counts as bars
of the range's own scale rather than to write a second set per scale.
`primitives_range.json` names `timeframes` (15m, 1H, 4H); the tracker keeps
its ATR window, prior close, clocks and idempotency memo per scale and runs
the same formation, balance and break rules on each scale's completed bar,
with at most one live range per scale. The manipulation funnel is still one:
mature boundaries of every scale and every pool feed the same
one-live-at-a-time resolver, and a range boundary source carries its range's
scale. Every range, boundary item, funnel snapshot and RANGE event carries
its scale; the store's close-beyond contract requires the parents and the BAR
on that scale; the DTO validators reject 1m only. Field names such as
`candidate_real_h1_bars` and `age_h1_bars` are kept so journals still reduce
and mean native bars. What this does *not* claim: the 1H range's natural
observation coverage was "sparse" before and the 15m and 4H ranges have none
yet — they are the same definition applied more widely, and their behaviour
on tape is measured, not assumed (`test_ranges_on_every_scale.py` checks the
contract, not the market).

## Bounded candidates, and what still grows

The candidate collection was append-only by design: a sweep disarmed a level
and kept it so a re-approach could re-arm the same identity, and nothing ever
removed one — 102 1m candidates at bar 500, 508 at bar 3,000, and the Brain's
`unswept_*` counts taken over every level since the run started. A candidate
now retires past `candidate_retirement_max_native_age_bars` of its own scale
(480: 8 h on 1m, 40 h on 5m, five sessions on 15m) or beyond
`candidate_retirement_max_distance_atr` (20) of its scale's ATR — both in
`configs/primitives_structure_liquidity.json`, recorded in
`semantics/parameters_v1_3.yaml`. Retirement is an atomic `LIQUIDITY_RETIRED`
fact (concept `candidate_retirement`) derived from the published snapshot and
appended after it, which the timeframe reducer consumes at once, so a cold
replay of the log agrees with the hot view; a retired item never reached also
ends as `LEVEL_INVALIDATED`, and its still-visible inventory item leaves the
crossing pipeline so a later touch cannot reach a level the Eye no longer
offers. Previous-period reference levels are retired at their rollover.
Equal-liquidity pools and range boundaries were first left to their own
lifecycles — and that lifecycle ended in the tracker alone: a resolved pool was
compacted from the snapshot and nothing told the reducer, so 177 of the 261
1m candidates at bar 6,000 of 2022-02 were pools that no longer existed, and
the set reached 897 by month end. A pool or range-boundary candidate now ends
as `LIQUIDITY_RETIRED` (`pool_resolved` / `range_boundary_retired`) on the bar
its entity leaves the authoritative set (`test_pool_candidate_retirement.py`);
with that, the 1m set holds 83 candidates at bar 6,000, 5 of them older than
the age limit and all of those reference levels waiting for rollover.

That bound did not flatten the per-bar curve, because the candidate
projection was only one of the owners. Profiling bars 2,000–2,500 against
bars 0–500 attributed the remaining growth to three full-rebuild patterns,
now replaced by bounded work:

- `CausalLiquidityTracker` deep-copied its entire `__dict__` as a rollback on
  every new swing once zone retention was at capacity (128 on 1m after a
  day) — 6 M `deepcopy` calls per 500 bars. The rollback is now a one-level
  snapshot that copies the mutable records and shares the frozen states
  (`test_liquidity_rollback_cost.py`).
- `InteractionUpdate.validate_canonical_bindings` re-admitted every nested
  Group 5 DTO through its constructor on every bar, over every closed context
  path retained up to `maximum_context_states` — +18.7 s per 500 bars. A DTO
  admitted once is remembered with the values it was admitted with, under a
  weak reference, and re-admitted only if those values changed
  (`test_interaction_admission_memo.py`).
- the store's structural-leg contract scanned the lifetime event map for the
  eligible `BAR_COMPLETED` roots of a leg's scale on every leg — +8.6 s per
  500 bars. The store keeps that sequence per (semantic version, scale,
  contract) as bars commit (`test_event_store_eligible_bar_index.py`).

Measured after those three, profiling bars 2,000–2,500 against 0–500 (call
counts, which do not depend on machine load): `deepcopy` 530 k → 6.06 M calls
before, 56 k → 57 k after; total function calls 66 M → 198 M per 500 bars, a
2.99× late/early ratio against 3.0× before the candidate bound. The per-bar
curve was flatter, not flat, and the owners that remained were three more
"retain terminal state until capacity" patterns, now each given an explicit
retention:

- The Group 5 update carried every closed context path until
  `maximum_context_states` (256) evicted the oldest — 169 of 187 paths were
  closed at bar 3,000 — and re-validated its whole graph over them every bar
  (`_is_admitted` 696 k calls per 500 bars). A terminal path is now exposed
  in `terminal_context_retention_real_1m_bars` (1) completed outputs, the one
  that closed it, and then compacted from the update; the closing transition
  was delivered on that bar and stays in the log. The capacity rule still
  refuses to evict a live context, and a source whose context was compacted
  is not reported as cold (`configs/primitives_interaction.json`,
  `test_terminal_context_retention.py`).
- `CausalLiquidityTracker` kept every reaccepted/retired zone and every
  accepted/rejected pool until `retained_zones`/`retained_pools` (128) forced
  the oldest out — on 2022-01-09→ at bar 2,500, 90 of the 1m tracker's 128
  zones and 123 of its 128 pools were terminal, rebuilt into every snapshot
  and re-projected into every inventory (361 items, 330 of them consumed
  history). A terminal record is now kept for
  `terminal_state_retention_native_bars` (1) completed native bars, the one
  that made it terminal, then compacted; zones a live pool or live reference
  source cites are kept, and capacity still fails closed on live state
  (`configs/primitives_structure_liquidity.json`,
  `test_terminal_liquidity_retention.py`).
- `build_structural_legs` rebuilt a scale's legs from every retained swing
  whenever one swing resolved. The fold's only carried state is the last
  swing in pivot order, so the observer now resumes it from that anchor over
  the swings that resolved since, keeps the legs it already projected, and
  drops those whose start swing the structure tracker evicted — exactly the
  legs a rebuild would produce, which the test checks field for field
  (`test_incremental_structural_legs.py`). A confirmation that sorts before a
  cached swing or an eviction from the middle of the order still rebuilds.
- Group 3 (2026-09-14) kept every mitigated, invalidated, expired or failed
  FVG and order block until `maximum_fvg_states` / `maximum_order_block_states`
  (256 / 128 per scale) forced the oldest out, so the 5m frame handed Group 5
  285 sources per bar at bar 20,000 of 2022-02 against 9 at bar 500 and each
  terminal state held a live entity timeline. A terminal state is now
  exposed in the output of the bar that produced it and compacted at the
  start of the next completed native bar
  (`terminal_state_retention_native_bars` (1) in
  `configs/primitives_zones.json`, registered as
  `zone_terminal_state_retention`; `test_zone_terminal_retention.py`).

With every tracker's state bounded, a profile of bars 20,000–20,500 against
0–500 on the 2022-02 tape (`observe` 11.0 s → 47.2 s, 4.29×) separated what
still grew with the journal from what had merely reached a high plateau
([evidence/eye_growth_profile_2022-02_2026-09-14.md](evidence/eye_growth_profile_2022-02_2026-09-14.md)).
Four owners grew with the journal, each now bounded:

- `EventStore._indexed_eligible_bars` re-sorted a scale's whole committed bar
  index on every structural-leg validation (+5.9 s per 500 bars). The index
  is appended in `known_at` order with one normalized root per scale and
  clock, and a batch stages only later bars, so the view is the committed
  list followed by the staged tail (`test_event_store_eligible_bar_index.py`).
- `EventMemory` kept a swing's still-transitionable lifecycle prefix hot for
  ever: the structure tracker drops the oldest confirmed swing from its
  bounded deque and declines an ambiguous forming candidate without a fact,
  and nothing else can transition a swing, so `swing:confirmed` grew
  33 → 933 and `swing:forming` 33 → 770 over 20,000 bars while every minute
  walked them all (+5.7 s). A swing prefix now stays hot only while the
  tracker exposes the swing (`_OWNER_EXPOSED_TIMELINE_NAMESPACES`,
  `test_swing_timeline_cooling.py`).
- The emitter walked each scale's `(clock, event id)` bar-root list from the
  front for a confirmed swing's pivot bar, rebuilt an id→clock dictionary
  over all of it per swing window, and scanned it for the bar after a swing
  crossing and for exact clock roots (+1.9 s). The lists are clock-ordered
  with one root per clock, and every lookup is a bisection
  (`test_emitter_bar_root_lookup.py`).
- `append_batch` copied the whole open forward-reference set per batch
  (54,963 entries at bar 20,000; +0.9 s). A small overlay records the batch's
  own additions and removals (`test_event_provenance_contract.py`).

Measured after the four: late window 47.6 s → 35.5 s under profile
(4.23× → 3.18×), warm-up to bar 20,000 819 s → 598 s. What remained was not
journal growth but a plateau reached late in the month, and both halves were
then bounded: the per-scale `swing_hierarchy_hot_retention` working set
(2048; 1m saturates near bar 5,000, 5m is still filling at bar 20,000) was
re-validated by `TimeframeState.__post_init__` on every `replace` and
re-ranked per candidate projection (+5 s) — a hierarchy is now validated
once against its timeframe and clock and trusted until it changes, carries
its rank map, and the geometry settle re-views only the Swings that moved
(`ValidatedSwingHierarchy`, `test_swing_hierarchy_validation_cache.py`); and
Group 3's terminal-until-capacity retention (+2.4 s in Group 5's source scan)
became the exposure-based retention above. Measured after both: late window
35.5 s → 25.3 s (2.36× late/early against 4.23× at the start of the day),
warm-up to bar 20,000 598 s → 415 s.

## What stays in memory, and what goes to the journal

With per-bar *time* bounded, per-bar *memory* was not: the audit store held
every event object of the run in three maps (about 2 KB per event on the
heap, 26 MB per 1,000 bars, a gigabyte per month of 1m replay), and the
emitter and store kept a per-entity or per-bar memory for everything they
had ever spoken about. Four changes, in the order they depend on each other:

- **Checkpoint restore streams the prefix.** Restoring a reducer or publisher
  checkpoint cold-replays the committed prefix to prove the compact state;
  it did so by taking `events_since(0)` as one tuple and building a second
  full store from it. The prefix fingerprint and the journal length now
  read a raw digest sequence, and the verifier streams the prefix through
  the store `CHECKPOINT_REPLAY_CHUNK_EVENTS` (4,096) events at a time. The
  same verification had also failed for every checkpoint taken after the
  first settled Swing, as far back as `a52241f`: the publisher persists the
  geometry views and the candidate rank / range-membership projections into
  the reducer state, and those depend on *when* they were projected, so a
  cold fold never reproduced them. Verification now compares the fold-owned
  part of each state (`_fold_owned_state`: projections reset, armed
  inventory re-derived); the projections travel in the checkpoint bytes
  (`test_checkpoint_replay_streams_the_prefix.py`).
- **A journal file for cold events.** With `audit_journal_dir` set
  (`configs/model.json`: `outputs/eye_journal`, hot window
  `audit_hot_window_minutes` = 4,320), the observer spills events below the
  reducer cursor and older than the window to an append-only file of
  length-prefixed pickled records. The store still answers `get`,
  `iter_events` and `events_since` for them through the file, keeps 32 bytes
  of digest and one id→index entry per cold event, and its fingerprint is
  unchanged; a checkpoint carries the journal path, the cold offsets and the
  digest sequence, and restore proves every cold record against its digest
  before re-appending the hot events with full validation
  (`test_event_store_cold_journal.py`). A journal has one owner: a second
  store bound to the same file would interleave its appends. Nothing
  removes a journal — a checkpoint may still name it — so a month of 1m
  replay leaves 480 MB (1.39 KB per event) under `audit_journal_dir` per
  observer; a lifetime rule is an open item.
- **Bounded bar-keyed memories.** The emitter's close and price range per
  bar root and each scale's clock-ordered root list, and the store's
  eligible-bar index and same-clock reservations, are read within a window
  fixed when the reader forms: a Swing freezes the range of its own
  confirmation window, a crossing resolves at the next root of its scale,
  a leg reaches its start pivot (at most 256 retained swings back on its
  scale), a previous-week reference level cites the 1m root of its extreme
  (at most two trading weeks back). Each keeps its newest entries
  (`BoundedDict`, `eyes/core/bounded.py`; `BAR_ROOT_RETENTION_PER_SCALE`
  and `BAR_MEMORY_RETENTION` 32,768, `ELIGIBLE_BAR_INDEX_RETENTION_PER_SCALE`
  4,096, `RESERVATION_MEMORY_RETENTION` 8,192); a miss fails closed exactly
  as an absent entry always did, and `test_bounded_entity_memories.py`
  proves a two-session replay emits identical events under bounds a small
  fraction of these while the memories are actually evicting. The
  per-entity memories — the event that last spoke about a swing, level,
  BOS, zone or range — are deliberately *not* count-bounded: the emitter
  re-walks every retained entity on every frame, so such a memory is read
  for as long as the slowest scale retains the entity (256 4H swings is
  months), and a month replay with every memory instrumented measured
  lookups reaching back over the whole run (a first attempt at 16,384
  entries failed the month replay at bar 25,000 with "BOS post-break
  resolution lacks its canonical penetration event"). They cost about
  75 MB a month of 1m tape; bounding them means dropping an entry when no
  tracker retains its entity any more, which is the open item below.
- **Entity identities are not forward references.** Legacy-transport state
  events name entities in `source_ids` (`swing:…`, `pool:…`); each was an
  open forward reference no later event could resolve. An identity in a
  registered entity namespace is skipped; raw opaque identities keep their
  forward standing.

Measured on 1,500 bars of the real tape with a 60-minute hot window, heap
growth fell from 26 MB to 12 MB per 1,000 bars before the bar-keyed bounds;
the month replay in [evidence/eye_memory_2022-02_2026-09-14.md](evidence/eye_memory_2022-02_2026-09-14.md)
carries the final numbers, the per-memory reach table, and the proof that
none of the four changes altered an emitted event.

## What the runtime loads, and what was removed

`eyes/core/` is 74 k lines; the runtime path (`CausalObserver` and everything
it imports) is 21 modules of them. On 2026-09-13 the pieces the runtime never
loaded were removed: `eye_statistics.py` (an optional 4,268-line statistics
projection whose only consumer was the authority scan), the 2023 authority
scan and its case audit and the mature-range scan (all three imported the
validation-protocol loader retired with the typed Brain and could not run),
their three test modules, and the six `study_*.py` scripts (their receipts
live in [evidence/](evidence/)). The Task 0 lazy-import shims went with the
scripts.

Kept, and marked as the next removal: the Foundation v2.1 projection stack —
`semantic_foundation.py`, `semantic_lifecycle.py`, `semantic_zones.py`,
`structural_outcome.py` (≈9,300 lines). No replay has ever published a
`FOUNDATION_STATE_CHANGED` event, but the stack is wired into the semantic
selection (`configs/model.json` names the Foundation registry and its
identity, `eyes/core/semantics.py` validates it) and into the snapshot schema
(`FOUNDATION_VERSION` on five `market_state` dataclasses), so removing it is
a contract change, not a deletion.

## Scripts — `eyes/scripts/`

`replay_hash_stream.py` is the per-bar hash-stream and timing harness for
output-preserving cost changes; `scan_eye_event_statistics.py` is the event
census; `replay_coverage_stats.py` replays a window through the Eye alone
and writes one JSON under `outputs/eye_coverage/` counting what each 2026-09
repair changed (range coverage per scale, same-fact duplicates, formation
lag, target outcomes, the two event channels, per-scale state availability,
seconds per block and state sizes) — its 2022-02 reading is
`docs/evidence/eye_coverage_2022-02_2026-09-14.md`. Throwaway probes belong
here too.
