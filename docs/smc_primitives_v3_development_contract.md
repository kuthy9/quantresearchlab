# SMC primitives v3 development contract

Version: `3.1.0-group12.4`

This is the single development definition for concept groups 1–2. It is
descriptive and must not be tuned against PnL.

## Common clock

- Every primitive consumes only completed candles from its own timeframe.
- Candles containing synthetic minutes advance the market clock but do not
  change swing, structure, zone, pool, ATR, or semantic age.
- Frame metrics, readiness, bar ages, bootstrap ATR and event-memory
  durations use only `real_completed` candles. The raw completed tail may
  advance the frame cutoff and expose candle provenance, but cannot change
  semantic state, reference price, or event clock.
- Exact retry of one failed integrated observation is idempotent for the
  event-memory minute clock and the isolated displacement projection.
- A cold observer may age retained events only when its completed-1m prefix
  covers every retained or subsequently emitted event age/duration origin.
  The coverage start is persistent and checked at each event append.
  Otherwise it fails closed and requires an earlier checkpoint replay.
- A 1m decision observation may project a sweep onto an already-visible HTF
  inventory item, but it must not rewrite the HTF frame cutoff or frozen zone.
- Contract changes and unknown data gaps terminalize pending BOS, then reset
  all group 1–2 state. No relation crosses the boundary.

## Candle structure

- One completed bar emits range, absolute body, upper/lower wick, their range
  ratios, close location, direction, event time, and real/synthetic provenance.
- A zero-range bar emits zero component ratios, close location `0.5`, direction
  `0`, and `zero_range=true`.
- Non-finite OHLCV, invalid OHLC geometry, negative volume, or invalid contract
  identity is rejected before observation.
- Candle structure is a single-bar description, never a strategy.

## Confirmed swing and directional sequence

- Swing pivots use two completed real bars on the left and two on the right.
- A candidate remains `forming` until the second registered right bar. At that
  clock it becomes `confirmed` or `formation_failed`; confirmation is never
  backfilled to the pivot.
- Confirmed same-side swings are compared in integer ticks to produce
  HH/LH/EH or HL/LL/EL. Each swing records the prior same-side ID, delta,
  ATR-relative magnitude, age, and close-break clock/reason.
- A directional structure is `forming`, `formation_failed`, `confirmed`, or
  `broken`. Its identity is based on direction and original formation clock
  and remains unchanged through its lifecycle. Formation failure and break
  have separate frozen clocks/reasons. The protected swing may tighten but
  cannot be retrospectively loosened.
- Swing progression remains a structure descriptor. It is not `path_sequence`.

## BOS

- A pending BOS is created from an already-confirmed structural swing.
- Only a later same-timeframe completed real candle close strictly beyond the
  target confirms BOS. Wick penetration records an attempt but does not
  confirm.
- Lifecycle is `pending → confirmed` or `pending → failed`; failures have a
  registered boundary/supersession/structure reason.
- One BOS identity represents one pending attempt and becomes terminal after
  `confirmed` or `failed`. If a later confirmed-evidence refresh re-arms the
  same target, it creates a new identity whose `pending_at` is that refresh
  candle's completion clock; it never reuses the target's older confirmation
  clock or the terminal identity.
- Strength is the descriptive close distance beyond the frozen target,
  normalized by causal ATR. It does not change the confirmation rule.

## Support/resistance

- The first confirmed swing freezes a zone centered on its price.
- Frozen tolerance is `max(1 tick, 0.10 × causal ATR at formation)`.
- A later confirmed same-side swing inside the frozen zone is another touch.
  Formation touches plus the newest bounded touch history are retained, while
  `total_touch_count` remains exact. Touch clocks, intervals (derivable from
  clocks), source swing IDs, and confirmation-time reaction magnitudes are
  exposed.
- Lifecycle is `active → tested → broken → reaccepted`, with `retired` as
  a separate terminal realtime-visibility outcome from `active`, `tested`, or
  `broken`.
- Only a same-timeframe completed close beyond the frozen zone breaks it.
  A later completed close back into the frozen zone reaccepts it.
- `reaccepted` here means only a descriptive zone recapture. It is not the
  qualified entry-trigger reacceptance, whose leave/return/hold/failure
  sequence belongs to the later entry-location concept group.
- A zone is retired only after none of its retained source swing IDs remains
  in the current authoritative structure snapshot. A `formed` or `swept`
  equal-liquidity pool pins its source zone until that pool resolves.
  A newly confirmed same-bar swing that matches the frozen zone is admitted
  as replacement evidence before the retirement decision.
  Retirement is timestamped on a later completed bar with reason
  `source_evidence_retired`; event memory keeps that terminal transition.
- Capacity never evicts a live or newly terminal zone. Only a terminal zone
  already exposed on an earlier completed bar may leave the realtime workset;
  exhaustion without either such a candidate or a same-bar terminal transition
  fails closed. When a newly retired zone and a new unmatched swing occur at
  full capacity on the same bar, the terminal transition is retained in a
  one-real-bar exposure buffer, so the snapshot may temporarily exceed the
  realtime workset cap, but never by more than the number of same-bar terminal
  zones. The next real completed bar compacts only those
  already-observed terminal states before semantic updates; synthetic bars do
  not compact them.
- Bounds are never recentered using later swings.

## Equal highs/lows and liquidity inventory

- The second confirmed same-side swing in a frozen zone forms an equal-high or
  equal-low pool; later in-zone swings increase the double/triple/multiple
  membership without changing bounds.
- Lifecycle is `formed → swept → accepted|rejected`.
- A wick strictly beyond the frozen pool marks `swept`. Resolution occurs no
  earlier than the next completed bar: a close held outside is `accepted`; a
  close returned inside is `rejected`.
- Pools use the same one-real-bar terminal exposure rule: a newly
  `accepted/rejected` pool remains visible when another pool forms at full
  capacity on that bar. Overflow is bounded by the same-bar terminal pool
  count, then compacted before the next real semantic update.
- Inventory contains typed confirmed swings and equal pools. The eye emits only
  `visible` or `consumed`; `targeted` is a downstream belief/plan overlay.
- Typed liquidity is enabled only by an explicit `observer.liquidity_protocol`.
  Its file SHA is part of replay bindings. An authoritative inventory remains
  authoritative when empty and must never fall back to legacy proxy levels.
- Current 1m sweeps update the top-level observation inventory and event
  sequence while preserving each source timeframe cutoff.
- Frame pools remain frozen formation/source views. Top-level
  `liquidity_pool_states` is the stable current-state authority for
  `formed/swept/accepted/rejected`; inventory is the draw-availability view,
  and event memory is the ordered transition log. All three identities and
  frozen source fields must agree.
- If an observer first attaches after a native higher-timeframe pool has
  transitioned, it reconstructs the first crossing and next-real-1m
  resolution from the retained completed 1m prefix. It fails closed when that
  prefix starts too late to recover the first transition, lacks one
  predecessor plus the registered ATR window, or proves that the coarse
  native transition swallowed a post-resolution generation touch.
- Cold attach applies the same first-crossing reconstruction to every visible
  or consumed swing draw. A retained prefix beginning after any draw
  confirmation cannot prove that an earlier sweep did not occur and therefore
  fails closed, even when the native higher-timeframe state is still visible.
- After a terminal pool resolution, two later confirmed swings are required
  before a new equal-liquidity generation is formed.
- Active/tested zones and formed/swept pools are semantic state and cannot be
  evicted merely to satisfy a cache limit. Only previously observed terminal
  state may be removed; exhausted capacity otherwise fails closed instead of
  changing identity or growing without bound. A rejected capacity update is
  transactional: clock, lifecycle, ATR, touch history, and known-swing state
  remain at the last accepted completed bar.
- If a paired structure/liquidity update raises after one tracker has
  advanced, the observer becomes terminal and must be discarded in favor of
  the last replay checkpoint; a partially advanced observer cannot be reused.
- The same terminal rule applies when multi-pool bootstrap, sweep projection,
  or resolution fails after another pool may already have advanced.
- The brain and risk engine must consume this same top-level inventory. Legacy
  frame liquidity is only a compatibility projection.

## Group 1–2 development freeze

Status: frozen for downstream development as
`3.1.0-group12.4`. This is a primitive-level freeze, not a profitability or
strategy-validation claim.

- Protocol SHA-256:
  `189b6af3bff631c3985fa37bcf9f5f82528296800886d9c9bd4cbe123ea4c701`.
- Model configuration SHA-256:
  `cca13a4dbb70c215dc624c7db061034efaf78aa91537765cdcb9f1bf6755bdc4`.
- The unified default development suite passed with 421 selected tests; the 28
  historical-frozen tests remained excluded by the registered default marker.
  Focused BOS, Group 1–2, event-timeline, observer equivalence, and checkpoint
  resume tests also passed.
- The finite real replay used 12,279 rows from the registered causal front
  (source SHA-256
  `5057fe574b82b26e3fe8a7798607a177b847876fee7f5ef41a938c7c87499bfc`)
  and produced 5,460 decision-time observations. Its completion artifact is
  `outputs/development/group12_3_event_timeline_real_replay_r2_20190311_20190316/COMPLETED.json`
  with SHA-256
  `f6cd01a24c0b6ebf0a740b67f2d7d1b7df26c466ccb42d1d93b5ec4932ebabac`.
- Final replay acceptance found 1,228 retained entity timelines exactly
  matching the current four-timeframe typed snapshot, zero incomplete
  timelines, zero decision-time incomplete-clock anomalies, and 29 current
  retired zones with complete `active → ... → retired` histories.
- The prior stratified blind review passed 22/22 cases: 11 structure/BOS cases
  and 11 support/resistance or liquidity cases, with no systematic semantic
  misread. The real-replay BOS failure then consumed the one allowed BOS
  concept repair: a terminal attempt can no longer be re-armed under the same
  identity. The repaired case has a deterministic candle regression and the
  complete finite replay passed afterward.
- No PnL search, threshold search, MBO authority, or future-path input was used.
  All 5,460 actions remained `abstain`; this is expected before brain and
  decision calibration and is not evidence for or against profitability.

Any second systematic BOS semantic failure parks BOS instead of triggering
another threshold or concept rewrite. Downstream code may consume the frozen
typed snapshots and retained timelines without expanding `recent_events`.
