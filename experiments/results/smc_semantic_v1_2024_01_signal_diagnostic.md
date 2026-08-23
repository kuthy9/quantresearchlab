# SMC Semantic v1.0 — 2024-01 Signal Research Diagnostic

> Diagnostic only. This window is not OOS, cannot fit a Brain artifact, and cannot authorize trading.

- Replayed real 1m rows: 30,477
- Canonical atomic events: 42,150
- Audit events retained: 671,361
- Deterministic audit fingerprint: `9fabceb0d263e77be552af41ea299b7eef959f4142683ff59cb1e28933114bea`

## Nested event chain

| Stage | Signals | Resolved | Laplace rate | Δ vs prior | Min n=30 |
|---|---:|---:|---:|---:|---:|
| C0_matched_control | 9219 | 9163 | 0.491 | — | yes |
| E1_level_touch | 16706 | 16525 | 0.480 | -0.012 | yes |
| E2_sweep | 8566 | 8480 | 0.491 | +0.011 | yes |
| E3_sweep_displacement | 221 | 221 | 0.516 | +0.025 | yes |
| E4_sweep_displacement_mss | 18 | 17 | 0.421 | -0.095 | no |
| E5_plus_fvg | 13 | 13 | 0.467 | +0.046 | no |
| E6_plus_parent_alignment | 4 | 4 | 0.333 | -0.133 | no |

## Atomic event studies

| Event | Signals | Resolved | Laplace rate |
|---|---:|---:|---:|
| acceptance_confirmed | 5233 | 5195 | 0.500 |
| displacement_observed | 240 | 240 | 0.512 |
| fvg_created | 1297 | 1284 | 0.522 |
| fvg_touched | 1254 | 1238 | 0.501 |
| level_touched | 17017 | 16834 | 0.481 |
| mss_core_confirmed | 1765 | 1752 | 0.511 |
| qualified_bos | 1988 | 1971 | 0.514 |
| raw_boundary_break | 4579 | 4548 | 0.514 |
| sweep_confirmed | 8777 | 8690 | 0.493 |

## Non-nested comparisons

| Comparison | Population | Signals | Resolved | Laplace rate | Min n=30 |
|---|---|---:|---:|---:|---:|
| mss_prior_sweep_30m | with | 193 | 192 | 0.459 | yes |
| mss_prior_sweep_30m | without | 34 | 34 | 0.611 | yes |
| displacement_prior_mss_30m | with | 29 | 29 | 0.452 | no |
| displacement_prior_mss_30m | without | 211 | 211 | 0.521 | yes |
| fvg_prior_displacement_15m | with | 234 | 232 | 0.521 | yes |
| fvg_prior_displacement_15m | without | 1063 | 1052 | 0.522 | yes |
| parent_relation | aligned_with_parent | 20397 | 20249 | 0.498 | yes |
| parent_relation | against_parent_but_parent_intact | 21386 | 21139 | 0.491 | yes |
| parent_relation | parent_neutral | 78 | 75 | 0.481 | yes |
| parent_relation | after_parent_invalidation | 86 | 86 | 0.523 | yes |
| parent_relation | parent_unresolved | 203 | 203 | 0.459 | yes |

## Matched controls

- Requested: 16,706
- Matched: 9,219
- Unmatched: 7,487

## Interpretation guardrails

- Rates are structural one-ATR diagnostics before execution costs, fills, stops, or P&L.
- Sparse cells below the preregistered minimum n=30 are retained and marked; they are not evidence of absence or permission to retune.
- Nested stages occur at later confirmation clocks, so deltas measure registered conditional populations, not a causal treatment effect.
- No pseudo-level or time-shifted control was silently added; those remain explicit follow-up studies.

