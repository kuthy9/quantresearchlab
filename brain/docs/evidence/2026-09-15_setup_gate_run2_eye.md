> **Retired 2026-09-16.** This receipt measured the mechanical (kNN) Brain, which
> was replaced by the LLM Brain ([spec](../specs/2026-09-16-llm-brain-design.md)).
> The code it describes no longer exists; the numbers stand as the record of why.

# Setup first-passage gate, run 2 against the repaired Eye — status, 2026-09-15

Spec: [2026-09-13-setup-first-passage-gate-design.md](../specs/2026-09-13-setup-first-passage-gate-design.md)
(§11). Run 1's receipt and verdict: [2026-09-14_setup_gate.md](2026-09-14_setup_gate.md).

## Status: no verdict — four of the twenty-three weeks do not build

Run 2 (`4e63cf0065dacb06`, recorder `paths_v3`, Eye at `3fa40ab` = `main`
`d8773a2` merged into `brain`) built 19 of its 23 Globex-week blocks. The
Eye raises on the other four, on the real tape, inside its own consistency
checks (below). With 96 emitted sessions the primary fold cannot be formed
(110 needed), so no cell can be judged and the gate is not run. The
verdict of record remains run 1's FAIL until the four weeks build.

## What the Eye repairs changed for the Setup build

Measured on the 19 common weeks (96 sessions), both runs' path logs
relabelled with the current labeller (`compare_runs`, first occurrence per
path and clock; shortfall = realised `hit_target` − driftless ratio
d_f / (d_t + d_f), session-block bootstrap 5–95 %).

**Group 5 is bit-for-bit the same.** 31,772 new-step rows and 8,878 paths
in both runs; every step-kind count, every cell's first-occurrence count,
every `past_failure` share and every failure-distance median are identical.
The seven repairs (one formation clock, 1m as microstructure, no scale
hard-coding, one crossing once, target outcomes, candidate retirement,
bounded memories) touch what Group 5 reads only through the 5m FVG / OB
frame and the formed-pool manipulations, and the Eye's own commits say the
5m facts are unchanged; the milestones, boundaries and directions the
run-1 diagnosis pointed at are therefore exactly as they were.

**What did move is the target inventory.** Candidate retirement (480
native bars / 20 ATR) reshapes the unswept BSL / SSL views the labeller
draws targets from: `no_target` falls (K0:zone 26.7 % → 23.5 %, K2:pool
28.6 % → 26.1 %), target distances grow a little (K0:zone d_t 0.71 → 0.74
ATR₆₀, K0:pool 0.36 → 0.45), the driftless ratio and the realised hit rate
fall with them. The pool scale is now read from the state it belongs to:
1m 19,466 rows, 5m 837, 15m 173, 1H 60, 4H 8 (run 1's recorder wrote `1m`
on every row).

| cell | first / labelled (run 1 → run 2) | hit (1 → 2) | shortfall run 1 | shortfall run 2 | d_f ATR₁ₘ | ≤ 1 min |
| --- | --- | --- | --- | --- | --- | --- |
| K0:zone_return | 1,951 / 1,428 → 1,491 | 0.258 → 0.255 | −0.044 [−0.068, −0.019] | −0.040 [−0.064, −0.014] | 2.59 → 2.61 | 5 % |
| K0:pool_reversal | 6,909 / 6,458 → 6,486 | 0.161 → 0.139 | −0.012 [−0.020, −0.005] | −0.006 [−0.012, 0.000] | 0.39 → 0.39 | 56 % → 54 % |
| K1:zone_return | 247 / 216 → 225 (87 sessions) | 0.278 → 0.284 | −0.016 [−0.063, +0.038] | −0.014 [−0.063, +0.039] | 1.98 → 1.96 | 6 % |
| K1:pool_reversal | 3,106 / 2,967 → 3,002 | 0.251 → 0.241 | −0.010 [−0.021, +0.001] | −0.008 [−0.020, +0.003] | 1.11 → 1.12 | 19 % → 18 % |
| K2:zone_return | 1,597 / 1,186 → 1,213 | 0.242 → 0.231 | −0.027 [−0.048, −0.006] | −0.026 [−0.047, −0.004] | 1.52 → 1.52 | 21 % → 20 % |
| K2:pool_reversal | 3,103 / 1,152 → 1,232 | 0.212 → 0.212 | −0.100 [−0.130, −0.069] | −0.099 [−0.129, −0.065] | 4.02 → 4.21 | 8 % → 7 % |

Every finding of the run-1 diagnosis survives the repairs: the realised
hit rate sits below the driftless ratio in every cell, worst where the
failure boundary is farthest (K2:pool), K0:pool_reversal still resolves
within a minute on 54 % of instances against a 0.39 ATR₁ₘ boundary, and
K1:zone_return still occurs on too few sessions (87 of 96) to be judged.
Nothing here is a verdict; it is the same exploratory view as the run-1
diagnosis, on a Setup population the Eye repairs did not change.

**What the repairs did change for the Brain's pass.** A block that took
63–91 minutes under the old Eye takes 9–10 minutes (six workers, 23 blocks
in under 40 minutes instead of 5.5 hours) with workers at 220–330 MB. Two
consequences reached the recorder and are fixed in `3941c0a` (spec §11):
a resolved manipulation is compacted a bar after it resolves, so a pool
path's later steps found no context (74.7 % `context_found` before the
fix, 100.0 % after, on the same blocks); and the Eye's cold-event journal,
written where `configs/model.json` points, grows by about a gigabyte per
block and is never emptied, so each block now journals into a temporary
directory removed with it.

## Is the FAIL the Eye's? Three tests on the 19 blocks (exploratory)

The gate asks whether the Setup adds to geometry. Three simpler questions
say where the absence comes from; all on run 2's 19 blocks (96 sessions),
first occurrence per path and clock, session-block bootstrap 5–95 %.

**1. Direction alone.** Ignore targets and boundaries: from the Setup's
close, does price travel +R in the Setup's direction before −R? Driftless
is 0.5 exactly. At R = 1 ATR₁ₘ and R = 3 ATR₁ₘ every cell is a coin flip:
K0:zone 0.493 [0.471, 0.515] / 0.506 [0.484, 0.530]; K0:pool 0.509 [0.500,
0.517] / 0.494 [0.485, 0.504]; K1:pool 0.507 / 0.483 [0.469, 0.497];
K2:zone 0.517 / 0.519; K2:pool 0.495 / 0.491; K1:zone (n 247) 0.522 / 0.556
[0.510, 0.602]. At R = the Eye's own failure distance the same, with 27 %
of K0:pool instances touching both ±R on the sweep bar itself. By pool
scale (1m / 5m / 15m), zone kind (FVG / OB), entry mode and session hour
(RTH / overnight) no subgroup leaves 0.5 beyond what fifty unadjusted
comparisons produce; 5m-pool sweeps at K0 lean *against* the direction at
3 ATR₁ₘ (0.428 [0.380, 0.478], n 284).

**2. The Eye's own target.** For zone paths, the Eye publishes its own draw
(`nearest_visible_draw_distance_points`). P(draw before failure boundary)
against d_f / (d_draw + d_f): K0 0.660 vs 0.662 (−0.002 [−0.023, +0.016],
n 1,835); K1 0.593 vs 0.556 (+0.037 [−0.009, +0.085], n 241); K2 0.461 vs
0.455 (+0.006 [−0.013, +0.025], n 1,334). With the Eye's own target the
Setups pay exactly what a driftless walk pays. The shortfall reported
under the Brain's 1R rule is therefore a property of that rule (levels
selected ≥ 1R away are reached less often than the ratio says), not
evidence about the Eye's direction; it does not bias the verdict, which
compares M₁ and M₀ on the same labels.

**3. Survivors.** Instances still unresolved after 5 / 15 / 30 minutes,
against the driftless ratio re-anchored at that minute's close (the
time-0 ratio is stale for a survivor, whose price has by construction
moved away from the boundary — against the stale ratio K0:pool shows a
spurious +0.19). Re-anchored, pools stay driftless at every checkpoint
(K0:pool after 30 min 0.440 vs 0.445; K1:pool 0.380 vs 0.403) and zones
and K2:pool fall further behind the longer they live (K0:zone after 30
min 0.288 vs 0.359, −0.072 [−0.110, −0.036]; K2:pool −0.149 [−0.188,
−0.113]). Nothing the Setup earns by surviving is information; what it
loses is.

**Reading.** The FAIL is not a Brain-side artefact: with the Eye's own
boundary, the Eye's own target and no target at all, price after the
Eye's milestones, in the Eye's direction, is a driftless walk or slightly
worse. No label, feature set or model downstream can recover an edge from
a direction that is 0.5 and a target reached at the driftless rate. Nor is
it an Eye *implementation* bug in the sense of the seven repairs, which
left the Group-5 population untouched. It is the Eye's Setup
*definitions* — where the milestones fire (a 0.39 ATR₁ₘ boundary at the
sweep; a micro-break that in 34 % of pool paths fires past the boundary),
which direction they assert, and how rarely the held-reacceptance clock
fires on zones — that on this tape carry no directional information. The
alternative reading, that these event types carry no edge on NQ at all,
cannot be separated from this one without a different, pre-registered
Setup definition; the ICT claim the Eye encodes (sweep → reversal) is not
borne out by its own sweeps.

## The two Eye failures, reproduced

Both reproduce with a fresh Eye (`build_eye`, `configs/model.json`) driven
bar by bar over the block's own window (`globex_weeks` → `warmup_start`
… `end`, seven-day warm-up); both are inside the Eye, on the real tape,
and are not raised by anything the Brain passes in.

**A. An order block qualifies after its base-origin core was pruned from
the emitter's memory** — weeks 2022-04-11 and 2022-05-23.
`eyes/core/semantic_event_emitter.py:3855` (`_record_group3_events`)
raises `qualified origin zone lacks the base origin core its impulse
published`. On the 2022-04-11 block the first entity-memory prune fires at
bar 10,266 (2022-04-13 04:05 New York) and keeps 1 of the 320 base-origin
cores published so far; four bars later (04:09) a 5m order block reaches
`untested` citing a pruned core. On 2022-05-23 the same raise comes at bar
10,005 (2022-05-24 23:44) with 2 cores left. `_retire_entity_memories`
(`eyes/core/observation.py`) harvests identifiers from the *published*
states; a base-origin core is published once and then compacted, and
until an order block qualifies the only holder of its
`base_origin_core_id` is the zone tracker's unpublished candidate, so the
harvest sees nothing live and `_base_origin_core_event_ids` loses it. The
raise is the emitter's refusal to backdate a core, which is the right
refusal; the prune is what is wrong. The liveness sources need the zone
trackers' pending cores (or the memory must stay unprunable while the
core's locating displacement is retained).

**B. A 15m dealing range confirms balance with one upper touch** — weeks
2022-05-30 and 2022-06-06 (in the latter's warm-up).
`eyes/core/range_auction.py:1029` (`_advance_live_range`) replaces the
state with `balance_confirmed_at` and `contract/eye/entities.py:1767`
raises `mature dealing-range evidence is invalid`. The range is
`6d1bb2f1…` on `Timeframe.M15`, formed 2022-05-30 09:45, confirmed at
19:30 on bar 1,230 of the 2022-06-06 block (2022-05-30 19:29, the
Memorial-Day evening session): `lower_touch_count` 3 with
`lower_source_tested_at` 10:15, but `upper_touch_count` 1 and
`upper_source_tested_at` None, while the reducer's own gate — which counts
`balance_upper_test_generations` — passed. The reducer's generation count
and the contract's touch count disagree on the 15m scale; on the 1H-only
range they did not.

## Files

- `outputs/setup_gate/4e63cf0065dacb06/` (ignored): `run.json`
  (`eye_revision` `5e9788b542b0`; the Eye's code is identical from
  `3fa40ab` through `3941c0a`), 19 blocks, no gate tables.
- `outputs/setup_gate/blocks_run2.log`: the build log with the raise.

## Commits

`3fa40ab` merge of `main` (the Eye repairs) · `5e9788b` schema-6
follow-up (every scale in the event log, 51 kinds, Eye revision in
`run.json`) · `3941c0a` pool context memory, temporary journal per block
· spec §11.
