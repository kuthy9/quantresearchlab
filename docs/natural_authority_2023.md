# 2023 Group 1–5 natural-authority review

## Scope and frozen identity

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

Because neither fixed week produced a mature range, the preregistered 2023
upper-bound scan was run through the lightweight Group 4 path only. It used no
Brain, Decision, Risk, Scene Graph, MBO, images, or PnL. One natural range case
was then replayed through the production Reader, Observer, EventMemory and
public Scene Graph update path with a complete local-calendar warmup.

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
| Group 4 range | The fixed weeks produced forming/broken ranges but no mature range. The 2023 scan produced 449 candidates, two mature ranges and one right-censored forming range. | Natural mature-range observability is demonstrated, but only two cases exist; broad stability remains pending. |
| Group 4 manipulation | The two fixed weeks' pool paths obeyed multi-bar resolution. The annual mature-range inventory produced two boundary sweeps: one `reaccepted`, one `accepted_outside`. | Pool and range-boundary lifecycle reachability passed. The rare range sample count remains explicit. |
| Group 5 typed states | Entry location, first pullback, entry-zone reacceptance, micro-BOS and path lifecycles occurred naturally. Path-step kind is now preserved in Scene Graph metadata; the corrected week and target window contained no `unknown` path kinds. | Typed transport and general lifecycle reachability passed. Independent action authority remains false. |
| FAVR | No exact identity-bound complete chain occurred in the fixed weeks or the natural target case. | `favr_enabled=false`; FAVR remains parked. |

The 2023 Group 4 upper-bound funnel was:

- 9,450 unique geometrically valid H1 source pairs;
- 449 forming candidates;
- 2 mature candidates;
- 449 terminal broken states and 1 right-censored forming state;
- forming terminal causes: 353 close breaks, 90 maturity deadlines, 2 source
  invalidations and 2 contract resets;
- unmet gate counts: bilateral touches 437, compression 369, midpoint crossing
  322, width 254, duration 191 and inside-close fraction 130;
- 4 range-boundary inventory items and 2 natural range-boundary manipulations,
  split one reaccepted and one accepted outside.

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
