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
`f92b24c8…1f0c` and, after the later protocol and registry changes in this series, settled at `29cc2ba1…500a`, then at `edf27ca3…27c7` when the Group 5 terminal-context and liquidity terminal-state retention parameters were registered; `configs/model.json` pins the current value.

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

**The dealing range is not generalized.** Group 4 is not a definition applied
to a scale but a fusion of the 1H range with the 1m manipulation, reacceptance
and entry-path machinery, and its registered parameters are written in those
units (`candidate_real_h1_bars` 8–24, `range_manipulation_real_1m_bars`,
`…_completed_h1_closes`). Running it at 15m would be a new definition with
untested parameters, not the same one applied more widely, so
`{1m,5m,15m,4H}_range_*` stay NaN until a per-scale range definition is
registered.

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
offers. Equal-liquidity pools, mature range boundaries and previous-period
reference levels are bounded and retired by their own lifecycles and are left
to them. At bar 2,500 on 2022-01-09→ the 1m set holds 126 candidates.

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
census. Throwaway probes belong here too.
