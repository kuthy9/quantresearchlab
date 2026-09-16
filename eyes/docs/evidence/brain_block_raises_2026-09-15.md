# Two raises from the Brain's block build, reproduced and repaired — 2026-09-15

The Brain's Setup gate (run 2, `brain/docs/evidence/2026-09-15_setup_gate_run2_eye.md`)
built 19 of 23 Globex-week blocks on the Eye at `d8773a2`; four raised inside
the Eye's own consistency checks. Both raises reproduce with the registered Eye
(`build_eye`, `configs/model.json`) driven over the block's own window
(seven-day warm-up, `globex_week_warmup_7d`), on the real NQ tape, with
nothing the Brain passes in. Neither was reachable in the 2022 Q1 evidence:
the first needs an entity-memory pass to fall inside a displacement's short
window, the second needs a settled balance claim, of which Q1 had none on any
scale.

## A. A pruned base-origin core (blocks 2022-04-11, 2022-05-23)

`qualified origin zone lacks the base origin core its impulse published`
(`_record_group3_events`), at bar 10,270 of the 2022-04-11 block
(2022-04-13 04:10 New York) — the first liveness pass over the entity
memories ran four bars earlier and kept 1 of the 320 cores published so far.

Root cause: a base-origin core is published once, when its impulse locks the
anchor cluster; until a break qualifies the order block, the only thing that
names it is the zone tracker's pending candidate (`_ob_candidates`), which is
not a published state. The liveness harvest read only published states, so
the core's memory died while its candidate was still able to qualify. The
candidate lives exactly as long as its displacement is active, which is as
long as a break can still use it.

Repair: `CausalZoneTracker.pending_base_origin_core_ids()` names those cores
and the observer's harvest includes them (`_retire_entity_memories`).
`test_pruning_on_every_bar_changes_no_event` runs the pass on every bar of a
two-session replay and requires the same events as never pruning; before the
repair it raised at bar 244, exactly as the tape did.

## B. The first settled balance claim (blocks 2022-05-30, 2022-06-06)

Range `6d1bb2f1…` on 15m, formed 2022-05-30 09:45, settled its balance claim
at 18:30 New York with two price-test generations on each boundary, while
its upper source zone had one structural touch and no tested clock.

Two stale contracts, both from before `balance_range_v1.2` (2026-09-06)
moved the claim from the source zones' structural touches onto price tests:

1. `DealingRangeState` rejected the settled claim (`mature dealing-range
   evidence is invalid`) for `upper_touch_count < 2` and
   `upper_source_tested_at is None`. It now requires what the reducer gates
   on — two generations per boundary — and leaves the structural touches
   as the separate fact they are (`test_balance_range_split.py`).
2. With the claim accepted, the range's break 75 minutes later
   (`close_beyond_frozen_range`, 20:45) failed the store's parent contract:
   `expected … balance_range_matured … actual acceptance_confirmed,
   bar_completed, dealing_range_created`. `BALANCE_RANGE_MATURED` has been
   reserved-not-emitted since the lifecycle dropped its grades, and the
   emitter cites the creation event the promoted boundaries descend from.
   `_RANGE_INVALIDATION_SOURCE_KINDS` now names exactly that
   (`test_event_provenance_contract.py`, whose chain fixture had still been
   minting the retired event).

Neither the 1H nor the 4H scale had "coincided": the 288 ranges of 2022 Q1
settled no claim on any scale, so the two contracts had never been reached.

## After

All four blocks build (`repro_block.py`, scratch): 2022-04-11 12,540 bars /
164,124 events; 2022-05-23 13,920 / 186,205; 2022-05-30 13,680 / 181,369;
2022-06-06 13,680 / 180,772 — one liveness pass each, no raise. On the tape
the 15m range's chain is `balance_range_observed` (generations 2/2) →
`acceptance_confirmed` (`mature_range_boundary`) → `dealing_range_invalidated`
with parents {creation, BAR, Acceptance}.

Still growing, and not part of this: the zone tracker's own
`_base_origin_cores` map keeps every core it ever locked (about 320 per
10,000 1m bars) to tell new cores from known ones.
