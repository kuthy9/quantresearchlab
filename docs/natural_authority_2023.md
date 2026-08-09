# 2023 Group 1–5 natural-authority review

## Evidence layers

This document retains three distinct outcome-blind evidence layers:

1. the current full-eye 2023 annual scan at commit `f64427f31bb61066fe69dc6cd25349cf721d8b15`;
2. the earlier focused Group4-only annual scan at commit `d1a2671057f7dc8cd805904c90762646b23f8316`;
3. the earlier fixed-week Group 1-5 review at commit `7ac42735bf42b43e7731493f765342ae458a90e7`.

The full-eye aggregate supersedes only comparable aggregate coverage claims. It
does not rewrite the earlier artifacts, commands, identities or case observations.

## Current full-eye 2023 annual scan

The production Reader -> Observer path ran every completed pre-registered Eye
module (Group 1-5 plus Displacement) and a lightweight statistics consumer. The
scan identity is
`effe9ab4c0045ad41303cce64c9d5a2672ac92f760f61ad7ecac713f828a221b`;
the causal source SHA-256 is
`84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b`.
Protocol versions and run-bound hashes are recorded once in the permanent
[`summary JSON`](evidence/eye_group1_5_natural_authority_2023_summary.json), and
the 40 future-hidden clocks are frozen in the separate
[`case index`](evidence/eye_group1_5_natural_authority_2023_cases.json).

The scan retained 353,445 in-window observations. Its principal funnels were:

| Component | Full-eye annual result |
|---|---|
| Group 1-2 | 155,653 confirmed swings; 189,002 BOS identities, of which 53,592 confirmed and 135,404 failed; 25,690 formed pools and 25,412 swept pools; 182,559 inventory items and 170,491 consumed items |
| Displacement | 15,702 started -> 2,836 ever active -> 15,498 exhausted; 4,024 same-bar terminal -> restart transitions |
| Group 3 | 13,693 FVG geometries, including 4,904 displacement-linked; 531 order blocks created, 494 mitigated and 36 failed |
| MatureBalanceRange | 225,102 geometry-valid pair evaluations -> 15,640 eligible -> 448 forming -> 2 mature; 447 later broke and 1 was right-censored |
| Manipulation | 25,719 visible eligible sources -> 25,445 crossed source IDs -> 19,465 created episodes -> 8,840 reaccepted / 10,549 accepted outside / 76 deadline-censored |
| Group 5 | 5,435 qualified zones -> exactly 5,435 EntryLocation/path identities; 5,325 first pullbacks, 3,710 with trigger; 6,468 complete and 18,413 interrupted paths; 0 sequence-order errors |
| FAVR | No complete natural identity-bound chain; FAVR remains parked |

The 5,980-source gap between crossed identities and created episodes is fully
accounted for: 2,311 same-side secondaries, 749 coincident secondaries, 1,641
blocked by a live episode resolved on the same bar, 1,132 blocked by an existing
live episode, 145 ambiguous dual-side sources and 2 missing/stale sources. Thus
both source disposition and episode outcome conservation hold exactly.

Brain, Decision, Risk, execution/MBO, PnL, future paths, full-year Scene Graph,
images and per-minute snapshots were disabled. Internal reducer identity,
lifecycle and ordering remained enabled. `TARGETED` liquidity and execution
reality are therefore `not_evaluated`, not Eye failures.

Only two mature episodes occurred, so the current definition was sparse for the
registered 2023 source and window. The completed blinded review found both natural
mature ranges visually credible and no repeated same-gate miss among the selected
near-mature and multi-gate cases. It is therefore treated as a rare
MatureBalanceRange context rather than a generic H1 dealing range. The zero
complete FAVR chain keeps FAVR parked. This Eye-only run cannot justify merging
FAVR into LSR because it did not construct comparable invalidations or targets;
optional LSR `range_context` remains non-gating.

### Current sampled transport result

Thirty unique frozen clocks were replayed through EventMemory, the public Scene
Graph and the Eye renderer. Twenty-nine distinct images were produced and 28 were
reviewed with future price, PnL and later extrema hidden. The reviewed range,
reset, manipulation and Group5 categories matched their frozen classifications;
no systematic range miss, path-order error or identity mixing was found.

The strict nine-case subset produced nine observations and nine images. Seven
cases were exactly evaluable and all seven passed; this includes hard-boundary
terminal/no-revival handling, an exact November mature-range manipulation, and
complete/interrupted Group5 paths with 8/8 and 2/2 typed steps transported through
EventMemory and Scene Graph.

Two dependent July cases were not evaluable for exact annual identity under an
isolated seven-calendar-day prefix. The local replay used the same S/R source pair
and geometry but formed the sibling range one hour earlier, cascading into new
range-boundary and manipulation IDs. These cases remain `prefix_censored`: they
are not passes, but they are also not evidence of a reducer semantic failure.
Reproducing those annual IDs requires an annual reducer checkpoint or an earlier
causal prefix; it does not authorize changing a primitive or threshold.

The earlier Group4-only artifact reported 25,724 visible eligible sources,
whereas the production full-eye path reports 25,719. Both report 25,445 crossed
sources and 19,465 created episodes. The five-source difference remains bound to
the earlier artifact's own commit/profile and is not rewritten.

## Earlier fixed-week review: scope and frozen identity

This review is descriptive and outcome-blind. It did not inspect PnL, reveal
future paths, load MBO, or search primitive thresholds. The frozen pre-replay
code commit was `7ac42735bf42b43e7731493f765342ae458a90e7`; the tested protocol files were:

| Group | Version | SHA-256 |
|---|---|---|
| Group 1–2 | `3.2.0-group12.7` | `ab3d2cde247f222900fb74c1b61f88c51b87b5056c15b51f003878042e18c457` |
| Displacement | `3.2.0-displacement-episode.3` | `b7ebe0205af7023ad81d71c15985d1e7755c704acda983508f5d16280e09efc4` |
| Group 3 | `3.2.0-group3.4` | `1fb83f45c2b00a1c35c2644776e97669dfa614ef9ac9daf688b03523f4e19f64` |
| Group 4 | `3.2.0-group4.1` | `af04fc78133d43c4a643e26898b026ece9f0aafc5c51da4719437d9cb06d1d5b` |
| Group 5 | `3.2.0-group5.3` | `c51ead7e909b1db91148e4c9cd36aa22c9ef48cb9217264cb3b2b6bdc68ce105` |

The causal OHLCV source was the registered previous-session front at SHA-256
`84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b`.

### Earlier focused full-year Group4-only scan

The fixed-week Group 1–5 replay and image review above remain bound to commit
`7ac4273`. The full-year Group 4 coverage scan was rerun separately at commit
`d1a2671057f7dc8cd805904c90762646b23f8316`, with scan identity
`41dec91ef5a45bf5827ae48f1912b6609af802c8afa6480ae9e8be49b9279a4e`.
Its exact source, model, data-split and five protocol identities are recorded
once in the permanent lightweight result:
[`group4_natural_authority_2023.json`](evidence/group4_natural_authority_2023.json).

The registered profile fixes 2023-01-01 00:00 EST through 2024-01-01 00:00
EST, seven New York calendar days of warmup, the causal OHLCV source, and all
five enabled pool-source timeframes. It can be reproduced with:

```bash
python3 scripts/scan_mature_ranges.py \
  --profile group4_natural_authority_2023_full_year \
  --force
```

Only Group 1–2 and Group 4 execute in this scan. Displacement, Group 3 and
Group 5 hashes are context identities, not claims that those reducers were
rerun for the full year. The scanner keeps the production completed-bar
reader, all-scale pool formation, prior eligible liquidity inventory, H1 S/R
range sources, Group 4 reducer and boundary handling. It omits fields Group 4
does not consume, EventMemory payload views and Scene Graph projection. A
3,000-bar real-data A/B check found exact equality with the full Observer for
Group 4 inventory, pool, H1 S/R, range, manipulation, transition and anomaly
outputs.

## Replay design

The two fixed non-adjacent review weeks were:

- 2023-03-26 18:00 EDT through 2023-03-31 17:01 EDT, with the prior local
  calendar week as warmup;
- 2023-11-05 18:00 EST through 2023-11-10 17:01 EST, with warmup beginning
  2023-10-29 18:00 EDT.

Each review week contained 6,900 real completed 1m bars and no synthetic bars.
The autumn replay initially exposed a fixed-168-hour warmup bug across DST. The
runner now subtracts New York calendar days; the corrected replay loaded 6,900
warmup plus 6,900 review bars. The correction changed only a few carried 1m
S/R/liquidity identities and did not change any review-week Group 1–5 lifecycle
counts or conclusions.

Because neither fixed week produced a mature range, the registered 2023
upper-bound scan was run through the lightweight Group 1–2 + Group 4 path. It
used no Brain, Decision, Risk, Scene Graph, MBO, images, PnL, outcomes or
threshold search. One natural range case was then replayed through the
production Reader, Observer, EventMemory and public Scene Graph update path
with a complete local-calendar warmup.

Thirty-six completed-data-only images were reviewed: 12 category-stratified
cases from each fixed week and 12 fixed clocks around the natural mature-range
episode. The first two category samples were concentrated near each Sunday
open, so they support bounded semantic review rather than a claim of broad
regime stability. The third set explicitly covered the rare Group 4 sequence.

## Results

| Component | Natural result | Authority boundary |
|---|---|---|
| Group 1–2 structure, BOS, S/R and liquidity | All principal lifecycles and BOS scopes occurred on both directions and multiple scales. No causal-clock, identity, or systematic visual error was detected. | Bounded natural reachability and lifecycle transfer passed. This is not a trading-edge result. |
| Displacement episode `.3` | Both directions produced `started`, `active`, and `exhausted`; the two fixed weeks contained 605 starts and 107 active transitions. One visually late start remains a non-systematic review item. | Bounded natural reachability passed; no PnL tuning or broad-regime stability claim. |
| Raw/linked FVG and Order Block | Raw and displacement-linked FVGs covered open/partial/mitigated/invalidated. The two weeks produced 19 OBs, 18 mitigations and one failure. | FVG natural lifecycle passed. OB failure coverage remains sparse. |
| Group 4 range | The fixed weeks produced forming/broken ranges but no mature range. The registered annual scan recorded 448 in-window formation events, two mature ranges, one in-window forming range right-censored at year end, and 448 break events; one break belonged to a warmup-carried candidate. | Natural mature-range observability is demonstrated, but only two cases exist; broad stability remains pending. |
| Group 4 manipulation | The annual all-scale scan found 25,724 unique visible eligible sources, 25,445 crossed sources and 19,465 typed manipulations: 8,840 `reaccepted`, 10,549 `accepted_outside` and 76 deadline-censored. Four mature-range boundary items produced two manipulations, split one `reaccepted` and one `accepted_outside`. | Pool lifecycle reachability exists on every enabled timeframe and the range-boundary lifecycle is reachable. This is descriptive coverage, not FAVR or economic authority. |
| Group 5 typed states | Entry location, first pullback, entry-zone reacceptance, micro-BOS and path lifecycles occurred naturally. Path-step kind is now preserved in Scene Graph metadata; the corrected week and target window contained no `unknown` path kinds. | Typed transport and general lifecycle reachability passed. Independent action authority remains false. |
| FAVR | No exact identity-bound complete chain occurred in the fixed weeks or the natural target case. | `favr_enabled=false`; FAVR remains parked. |

The reproducible 2023 Group 4 upper-bound funnel was:

- 9,450 unique geometrically valid H1 source pairs;
- 448 range formation events inside the registered year;
- 2 mature candidates;
- 448 break events, including one warmup-carried candidate, and 1 in-window
  forming state right-censored at year end;
- forming terminal causes: 353 close breaks, 90 maturity deadlines, 1 source
  invalidation and 2 contract resets;
- unmet gate counts: bilateral touches 436, compression 367, midpoint crossing
  321, width 253, duration 190 and inside-close fraction 129;
- 4 range-boundary inventory items and 2 natural range-boundary manipulations,
  split one reaccepted and one accepted outside;
- 25,724 unique visible eligible manipulation sources, 25,445 crossed sources
  and 19,465 created states;
- created-state outcomes: 8,840 reaccepted, 10,549 accepted outside, 76
  deadline-censored, 0 hard-boundary-censored and 0 right-censored;
- 145 ambiguous dual-side source crossings and 0 ATR-unready sources; neither
  category created a manipulation.

All five enabled pool-source timeframes were naturally represented:

| Pool timeframe | Visible | Crossed | Created | Reaccepted | Accepted outside | Deadline-censored |
|---|---:|---:|---:|---:|---:|---:|
| 4H | 42 | 35 | 12 | 6 | 6 | 0 |
| 1H | 161 | 150 | 44 | 17 | 27 | 0 |
| 15m | 741 | 714 | 162 | 65 | 96 | 1 |
| 5m | 2,174 | 2,125 | 738 | 322 | 414 | 2 |
| 1m | 22,602 | 22,417 | 18,507 | 8,429 | 10,005 | 73 |

For both the global cohort and every source-kind/timeframe/side row,
`swept_created = reaccepted + accepted_outside + deadline_censored +
hard_boundary_censored + right_censored`. The permanent JSON passed this
conservation check and separately proved that ambiguous and ATR-unready source
identities never entered the created cohort.

## Natural FAVR fail-closed trace

The natural target retained one exact range identity throughout the following
sequence:

1. `04:00`: range `d4ce7a3b…` became mature with frozen bounds
   `[15827.52, 15920.53]`.
2. `04:49`: its upper-boundary liquidity was consumed and manipulation
   `df037012…` entered `swept`.
3. `04:52`: the same manipulation entered `reaccepted` after the required
   multi-bar inside hold.
4. The active displacement through this interval was long, in the same
   direction as the upper sweep. It was not the required opposite short return
   displacement.
5. `05:00`: the completed H1 bar closed at `15928.50`, beyond the frozen upper
   bound, so the original range correctly became terminal `broken`.
6. A short displacement/FVG appeared at `05:30`, after the range had broken.
   The graph correctly refused to attach it retrospectively to the old range
   and manipulation episode.

The exact graph-chain count of zero is therefore a correct fail-closed result,
not a missing Scene Graph relation. The public synthetic graph tests separately
cover a complete FAVR chain, a missing-middle-edge rejection, and a crossed-ID
rejection without private node or edge insertion.

## Release and implementation status

The Engine factory now requires an explicit `development` or `live` runtime
mode. Live construction is centralized and fail-closed on model natural
authority, economic validation, rolling OOF, MBO stability, live permission,
and DFP/LSR Group5 input authority; direct live construction cannot bypass the
factory check. DFP/LSR input authority is now enabled, while FAVR authority is
separately false. The model-level release-readiness flags remain false, so
development evaluation can continue while real execution remains blocked.

No protocol threshold or playbook rule was changed by this review. The only
post-replay corrections were the local-calendar warmup calculation and the
descriptive `path_step_kind` Scene Graph metadata needed to audit an already
existing typed sequence.
