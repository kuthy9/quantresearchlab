# Legacy composite entry-location and path-sequence protocol

The companion [`configs/primitives_entry.json`](../configs/primitives_entry.json)
is the frozen Group5 composite identity used by cold artifacts and registered
research runners. It is not the current hot Interaction contract, and current
runtime code must fail closed rather than drive physical Interaction semantics
with this legacy hash.

The hot Eye contract is
[`configs/primitives_interaction.json`](../configs/primitives_interaction.json)
(`3.2.0-interaction.1`). It publishes only physical zone interaction,
reclaim/hold, raw micro-break and ordered milestone facts. Path completeness,
directional alignment, setup qualification and trade construction belong only
to Brain. The remainder of this document describes the frozen
`3.2.0-group5.4` composite contract for historical interpretation; its
qualification vocabulary is not current Eye authority.

This file documents the retained v1.2 playbook-facing Group-5 lifecycle. Its
`first_pullback` is not the additive foundation's canonical geometric First
Retest research object. Foundation v2 requires an exact object generation,
strictly later observed departure, one native completed-bar return, and an
outcome-blind frozen information set; evidence from either definition cannot be
silently transferred to the other. See the
[`foundation specification`](canonical_semantic_foundation_v2.1.md).

Group 5 closes four legacy semantic gaps with one incremental reducer:

1. entry location is the completed 1m relationship to one exact frozen 5m
   FVG or order block;
2. first pullback is the first later visit to that exact zone;
3. qualified reacceptance proves leave, later reclaim and a later completed
   hold bar; and
4. path sequence is an ordered set of typed events, never swing progression.

Micro BOS is not detected again. Group 5 only binds the exact upstream M1
`BreakOfStructureState` to a causal context and anchor clock.

This is descriptive eye infrastructure. It never chooses a playbook, phase,
action, stop, target or risk result. Typed playbooks may consume its exact
states through their own causal gates; Group 5 has no independent action
authority.

## Minimal implementation boundary

The implementation adds one `CausalGroup5Reducer`, not separate controllers
for location, pullback, reacceptance, micro BOS and path sequence.

The reducer maintains two bounded context kinds:

- `zone_return`, keyed by one exact FVG or order-block identity; and
- `pool_reversal`, keyed by one exact Group 4 pool-sourced manipulation.

Overlapping FVG and order-block zones remain distinct. Their shared
`source_displacement_id` is retained so downstream evidence grouping can
avoid counting the same displacement twice.

For `pool_reversal`, the context milestone freezes only the manipulation
root, reverse displacement ID, displacement active clock and direction.  Its
`observed_at` remains the first completed-bar clock at which an eligible
linked zone made that displacement causally visible; `source_active_at`
preserves the displacement's earlier-or-equal active clock.  No zone ID is
stored in that context step.  Each linked FVG/OB continues to own a separate
`EntryLocationState` and `zone_return` path.

FAVR remains disabled. A forming or broken range midpoint is not value.
Mature ranges and range-boundary manipulations remain Group4/Brain context;
Group5 does not register a range as an entry zone. After Group4 reacceptance,
Group5 may observe only the reverse-displacement FVG/OB location and its
ordered first-pullback/trigger path.

## Frozen dependencies and prohibited proxies

The runtime binds the exact current structure/liquidity, zone and range
primitive files declared by `configs/model.json`, plus tick size `0.25`.
Their content hashes provide semantic identity without a product-version
selection branch.

The following legacy values have no Group 5 authority:

- `5m.pullback_depth` and `5m.pullback_completeness`;
- `5m.reacceptance_direction`;
- `1m.path_sequence`;
- `1m.trigger_hold_direction`;
- support/resistance `reaccepted`;
- a duplicate Group5 pool reclaim/hold reconstruction when Group4 already
  owns manipulation reacceptance;
- legacy dealing ranges or 4H range position; and
- any future path, PnL, MFE, MAE, action label or MBO outcome.

The typed reducer, Brain and visualization must neither read nor reproduce
them as Group 5 truth.

## Completed-bar clock

The reducer is called once for every newly completed 1m bar.

- Only `real_completed` candles change semantic state, age, strength, hold
  count or path steps.
- A new `open` FVG or `created` order block is registered after processing
  older states on the real 1m bar whose end equals source `confirmed_at`.
  `partial` and `untested` are continuation states only; they cannot first
  appear as a reconstructed new Group 5 source.
- That formation bar cannot visit its own new zone.
- A registered zone can first be visited only on a later bar satisfying
  `source.confirmed_at <= bar.start`.
- A zone first seen after its confirmation clock is cold and ineligible:
  without the missing prefix the reducer cannot prove that an earlier first
  visit did not occur.
- A pool context is registered only when a
  `formed_liquidity_pool` manipulation is first received as `swept` with
  `swept_at == current_bar.end`. A manipulation first seen at an older sweep
  clock or already terminal is cold and cannot be backfilled. After the pool
  origin and sweep-close step are appended, current-input M1 BOS records
  resolved exactly at `swept_at` are recorded immediately as sorted,
  record-only `micro_bos_simultaneous`; they are never backfilled later.
- Synthetic closure minutes preserve semantic state and do not count toward
  hold time or market duration.
- An event time is the completed-bar end at which the evidence becomes
  knowable, never an intrabar extreme time.

An exact retry returns the previous atomic result. A same-clock payload
change or an earlier clock is rejected before mutation. Hard epoch
boundaries censor live paths; they do not claim a market invalidation.

Every Group 5 distance normalization receives the current positive finite
`frames[M1].metrics["atr"]`. Its frozen upstream calculation scans backward
over real-completed M1 candles for at most fourteen finite true ranges.
True range is `max(high-low, abs(high-prior real close),
abs(low-prior real close))`; the first real candle uses `high-low`.
Zero ranges occupy a place in the fourteen-observation window, but the
returned mean contains positive values only. Synthetic candles are excluded.
The upstream fallback is `1.0` only when there is no real candle or no
positive finite true range. Group 5 rejects a non-finite or non-positive
passed ATR.

## Entry location and first pullback

One exact source creates one location. New locations may be registered only
from:

- an FVG in `open`; or
- an order block in `created`.

Once registered, an FVG may continue through `partial`/`mitigated` and an
order block through `untested`/`mitigated`; only typed
`invalidated`/`failed` is a source failure.

For a long zone:

- near edge is the upper bound;
- far edge and failure boundary are the lower bound; and
- delivery side is strictly above the near edge.

Short is the mirror image.

Location lifecycle is:

`approaching → in_zone|rejected|left`

`in_zone → rejected|left`

`rejected → left`

Rejection freezes the first reaction and remains a valid trigger milestone,
but the active path continues watching for a strictly-later M1 BOS or a real
failure. A later adverse close beyond the far edge or typed source failure may
therefore move the current location snapshot to `left`; the immutable
`wick_rejection`/`reacceptance_held` path step retains the earlier event clock.
`left` is terminal, and later visits cannot rewrite the frozen first visit.

A departure is the first real completed close strictly on the delivery side.
First pullback requires a strictly later real bar whose inclusive high-low
range intersects the frozen zone. Near-edge equality is a visit.

The immutable first-pullback record contains:

- exact source and location IDs;
- frozen bounds;
- departure and entry clocks;
- `crossed_near_edge` or `gap_opened_inside`;
- contact reference price, explicitly not a claim about the exact first
  traded tick;
- penetration fraction;
- same-bar rejection flag; and
- same-bar failure and ordering relation.

If the bar opens within the zone, the open is the contact reference. If it
opens strictly beyond the far edge after a delivery-side close, Group 5
records `gap_through_frozen_zone` and does not invent a zone trade.

First penetration is frozen as:

- long:
  `clamp((upper_bound - max(bar.low, lower_bound)) / width, 0, 1)`;
- short:
  `clamp((min(bar.high, upper_bound) - lower_bound) / width, 0, 1)`.

A first-pullback bar that closes strictly on the delivery side is
`rejected`. It is `same_bar_wick_rejection` when it opened on the delivery
side and `gap_inside_recovery` when it opened within the zone. A later
delivery-side close after an in-zone visit may also reject, provided
qualified reacceptance has not begun.

A strict completed close beyond the far edge, an FVG `invalidated`, or an
order block `failed` produces terminal `left`. FVG/OB `mitigated` is a source
visit result and cannot erase an already-registered 1m path.
When a typed source failure and price-derived leave share a clock, the typed
failure reason and source event identity take precedence.

Every current location view publishes:

- distance to the frozen zone;
- direction-signed distance to the frozen failure boundary; and
- the nearest compatible currently visible inventory ID and distance.

For long locations, compatible means a `visible`, `above` inventory item
of kind `swing`, `equal_highs` or `equal_lows`, confirmed no later than the
current clock and strictly above the current completed close. For short
locations it means the mirrored `visible`, `below` item strictly below the
close. Parked `range_boundary` inventory is excluded. Candidates sort by
absolute point distance and then `item_id`; no candidate produces two null
fields. This is descriptive only and never selects the target.

The full authoritative inventory remains top-level; this bounded view does
not select a draw.

Those are observations. The eye does not select a draw or planned entry.
The brain may later choose a price inside the frozen band that differs from
the current 1m close.

`EntryLocationState` has no aggregate strength. It exposes
`first_penetration_fraction` and `reaction_atr` as independent components;
each remains zero until its defining event.

## Qualified reacceptance

Reacceptance lifecycle is:

`left → reclaimed|failed|censored`

`reclaimed → held|failed|censored`

`held`, `failed` and `censored` are terminal.

The only allowed anchor is a first pullback to a frozen FVG/OB entry zone,
using its near edge as reference. Pool manipulation reacceptance is a Group4
state and never creates `QualifiedReacceptanceState`.

For long, `left` requires a close strictly below reference and reclaim
requires a later close strictly above. Short is mirrored. Equality is
neither leave nor reclaim.

The reclaim bar contributes zero hold bars. `held` requires at least one
later `real_completed` 1m close strictly on the reclaimed side. A later
close exactly at the reference neither adds a hold bar nor fails; state
remains `reclaimed`. A synthetic bar cannot prove hold.

Before `held`, a strict completed-close loss back to the adverse side, a
strict completed close beyond the frozen failure boundary, or typed entry-zone
source failure produces a specific immutable failure reason.
Wick-only excursions through the failure boundary are not this descriptive
failure and do not replace the risk engine's separate same-bar stop rules.

A wick that enters an entry zone but always closes on the delivery side is
rejection, not reacceptance. For pool reversal, Group4 `reaccepted` already
means that a later inside close was followed by the configured hold bar.
Group5 appends that exact event as the pool path's `reacceptance_held` step;
it does not manufacture leave, reclaim or hold clocks of its own.

Reclaim margin is frozen using causal ATR at `reclaimed_at`; hold margin is
frozen using causal ATR at `held_at`. Strength is their clipped minimum. It
is a description, not a probability or gate.

At a hard epoch boundary, a live `left` or `reclaimed` state becomes
`censored` at the boundary clock with reason `hard_boundary_censored`. It is
exposed once through the dedicated boundary output and then cleared with the
old epoch. Censoring is not failure. Reclaim margin remains zero before
reclaim; hold margin and strength remain zero before held.

## Micro BOS binding

The only permitted source is a typed confirmed
`Timeframe.M1 BreakOfStructureState`.

The anchor is:

- `first_entered_at` for `zone_return`; or
- the exact `opposite_displacement` path-step clock for `pool_reversal`.

The first confirmed M1 BOS strictly after the anchor is bound once. An
aligned direction is recorded as `micro_bos_confirmed`; an opposed first
break is recorded as contradiction. The same context is not rearmed using a
more convenient later BOS.

A BOS resolved at the anchor clock appends `micro_bos_simultaneous` in
ascending `bos_id` order with `same_clock_unknown`. It is record-only: it is
not a bound trigger, does not count as aligned or opposed, and never closes
either context.

`micro_bos_ambiguous` applies only when multiple confirmed BOS records share
the earliest eligible clock strictly after the anchor. Each reference is
preserved and one step per reference is appended in `bos_id` order. That
strictly-later ambiguity closes as contradiction
`micro_bos_ambiguous_same_clock`. Pending BOS, wick attempts, HTF BOS,
pre-anchor BOS and legacy trigger-hold scores are prohibited.

The reference preserves upstream BOS ID, target swing, scope, pending and
resolved clocks, direction and strength. Group 5 does not copy the swing
detector or calculate another break level.

## Path sequence

Every context owns one bounded append-only `PathSequenceState`.

Lifecycle is:

`active → closed|censored`

Each step contains:

- unique step ID and kind;
- completed knowledge clock;
- exact source event/entity identity;
- predecessor step IDs;
- same-clock relation;
- direction;
- individual descriptive strength; and
- reason.

No aggregate path score is produced.

Context identities are exact: `zone_return.context_id == location_id` and
`pool_reversal.context_id == manipulation_id`.

Step source identity is frozen as follows:

| Step | `source_event_id` | `source_entity_id` |
|---|---|---|
| `zone_visible` | source zone ID | source zone ID |
| `departure_confirmed`, `first_pullback`, `wick_rejection` | null | location ID |
| `location_left` | typed zone ID only for typed source failure; otherwise null | location ID |
| `pool_swept`, pool `reacceptance_held`, `accepted_outside` | manipulation ID | manipulation ID |
| `opposite_displacement` | source FVG/OB ID | displacement ID |
| entry-zone reacceptance leave/reclaim/held/failed | null | reacceptance ID |
| every micro-BOS form | BOS ID | target swing ID |

The first step has no predecessor. Every later step points to exactly the
immediately prior step. Its relation is `strictly_after` for a later clock,
`same_clock_known` when reducer priority proves order at the same clock, and
`same_clock_unknown` when intrabar order cannot be recovered.

Processing priority for one clock is:

1. hard epoch boundary;
2. advance already-registered location/reacceptance from the real 1m bar;
3. apply typed FVG/OB failures and Group 4 resolution;
4. append visit/rejection/reacceptance/failure steps;
5. bind eligible current-clock confirmed M1 BOS;
6. if failure and BOS share a clock, record both as
   `same_clock_unknown` and let failure determine the close reason;
7. close eligible existing paths;
8. register current-clock new zone and swept-pool contexts without
   self-visit, resolution or ordered BOS binding; a new pool immediately
   records current-input anchor-clock BOS as sorted, record-only
   `micro_bos_simultaneous`, preventing later backfill; then
9. compact only previously exposed terminal contexts.

Zone paths may contain:

`zone_visible → departure_confirmed → first_pullback →
wick_rejection|reference_left → reference_reclaimed →
reacceptance_held|reacceptance_failed`

and a single bound `micro_bos_confirmed|micro_bos_opposed`, plus
`micro_bos_ambiguous` when the earliest eligible later clock has multiple
records, record-only `micro_bos_simultaneous`, and `location_left` when
applicable.

Pool paths may contain:

`pool_swept → reacceptance_held → opposite_displacement →
micro_bos_confirmed|micro_bos_opposed`

They may instead terminate on `accepted_outside`, the Group4 resolution
deadline, earliest-clock opposite-displacement ambiguity, or strictly-later
micro-BOS ambiguity. `micro_bos_simultaneous` remains record-only.

Steps are ordered only when the completed data proves order. Events at the
same completed clock carry `same_clock_known` or `same_clock_unknown`.
Changing scalar swing progression without a typed event appends nothing.

Zone rejection and qualified reacceptance held are nonterminal path
milestones. They leave the path `active`, with transition reason
`zone_rejection_observed` or `qualified_reacceptance_held`, so a playbook that
requires a strictly-later micro BOS can receive it. A zone path closes only on
the earliest strictly-later M1 BOS clock (including ambiguity),
location/reacceptance failure, or typed source failure. A simultaneous
anchor-clock BOS never closes. Qualified reacceptance held moves an `in_zone`
location to `rejected` at `held_at`. If BOS closes while the location remains
`in_zone`, the terminal atomic context freezes that exact `in_zone` state and
its view, age and state duration; it is not relabelled `left`. If a live
reacceptance has not held, it becomes `failed` at the closure clock with reason
`context_closed_before_hold`, while the location remains frozen `in_zone`.

The same freeze rule applies when reacceptance itself fails while the price
has not separately closed beyond the far edge and the typed source has not
failed. A separately satisfied far-edge or source failure still moves the
location to `left`.

When several zone terminal facts share a completed clock, the immutable
close-reason priority is: `location_left`, `reacceptance_failed`,
`micro_bos_ambiguous_same_clock`, `micro_bos_opposed`, then
`micro_bos_aligned`. `qualified_reacceptance_held` and
`zone_rejection_observed` are milestone reasons, not close reasons.

Pool reacceptance does not close by itself. Group5 first waits for one
strictly-later opposite displacement evidenced by a displacement-linked FVG
or qualified order block, then binds a strictly-later M1 BOS. Multiple
eligible displacement identities on the same earliest clock fail closed.
`accepted_outside` and the Group4 resolution deadline close immediately.
FVG/OB mitigation alone never closes a continuously registered context.

A simultaneous anchor-clock BOS never closes a pool path. At a hard boundary,
the active path and any live entry-zone reacceptance are censored, exposed
once on the dedicated boundary channel, and then removed with the old epoch.

Censored states retain the old epoch `symbol` and `instrument_id`. On
`contract_change_reset` that pair must differ from the boundary
observation's new pair. On `data_gap_reset`, `data_anomaly`, and
`tick_size_mismatch` it must remain the same pair. A censored
reacceptance must match the identity of its censored path context.

The complete bounded step tuple remains in the current observation. New
steps are also projected to EventMemory with the same identities and clocks.
The retained `entry_path` timeline records `active → closed|censored`, so
bounded `recent_events` cannot erase the path origin.

## Identity, state duration and retention

All IDs are lowercase SHA-256 of UTF-8 fields joined with a literal `|`;
timestamps use timezone-aware ISO-8601.

- location:
  `group5-location-v1|protocol_hash|symbol|instrument_id|kind|source_id`;
- reacceptance:
  `group5-reacceptance-v1|protocol_hash|context_kind|context_id|`
  `reference_price_decimal_10|left_at`, where the frozen descriptive
  reference uses exactly 10 decimal places and may lie between executable
  ticks;
- path:
  `group5-path-v1|protocol_hash|symbol|instrument_id|context_kind|`
  `context_id`;
- step:
  `group5-step-v1|protocol_hash|sequence_id|ordinal|kind|observed_at|`
  `source_entity_id`, plus `source_active_at` only for an
  `opposite_displacement` step.

The reacceptance identity's `context_kind` is exactly `entry_zone`; it never
encodes the path aliases `zone_return` or `pool_reversal`.

Every state outputs identity, full provenance, lifecycle, formation,
state-start, last-update and terminal clocks, real-1m age, real-1m state
duration and exact transition/failure reason. Location exposes penetration
and reaction components; reacceptance exposes reclaim/hold margins and held
strength; a path has no aggregate strength and its steps keep their own
strength. Age increments on each later real 1m update while retained. State
duration resets to zero on a lifecycle change, increments on each later real
bar in that lifecycle, and freezes at the next change.

There is one atomic cap of 256 contexts. A context owns its location,
reacceptance, micro-BOS references and path. Only a closed/censored context
already exposed in a prior successful observation may be evicted, ordered
by `ended_at` then path ID. A same-bar terminal remains visible for that
output. If capacity is needed and no eligible terminal exists, the entire
bar fails before commit.

## Stopping rule

Formalization, one incremental implementation, static review and the unified
synthetic/edge/causal suite, finite descriptive OHLCV replay and small
stratified outcome-blind review were completed for `3.2.0-group5.3`.  Protocol
`3.2.0-group5.4` changes the pool-reversal displacement wire identity and is
therefore intentionally incompatible with those checkpoints.  Its code-level
contract must pass the pre-registered five-session LSR multi-zone diagnostic
before the new evidence is described as replay-validated.  This does not
validate the missing natural FAVR chain.

At most one concept-level repair is allowed. A second systematic semantic
failure parks the affected concept instead of starting threshold search.

Primitive PnL, rolling OOF, MBO stability and sealed holdout are explicitly
out of scope. Typed state may feed the Brain during development, but Group5
never owns an independent action and FAVR remains disabled.

Implementation naming changed without changing this frozen protocol identity:
`InteractionSemantics` is the sole production reducer and `group5.py` is a
legacy import shim. The Eye-side schema exposes raw `MicroBreakFact` and
physical milestone paths only. Aligned/opposed classification, path success,
setup qualification, entry method, SL/TP, and TradeIntent are Brain-owned.
Historical `MicroBOSReference` and interpreted path names exist only in the
explicit `group5.py` cold-reader adapter and are not serialized by the current
`MarketObservation` contract. Schema 3 removed the former Group-5 availability,
current-state, transition, and boundary fields from the canonical dataclass and
exact pickle shape, and the current schema-5 contract still excludes them; all
current consumers read its sole `interaction_update`.
`InteractionProtocol`, `InteractionSemantics`, and `InteractionUpdate` remain
explicit internal-module imports rather than package-root public exports.
`InteractionUpdate` has an exact schema-1 pickle and artifact contract. Its
nested locations, reacceptances, paths, steps, and MicroBreak facts reject
missing or extra fields; cross-record custody uses bounded maps over the
current update rather than retained history.
