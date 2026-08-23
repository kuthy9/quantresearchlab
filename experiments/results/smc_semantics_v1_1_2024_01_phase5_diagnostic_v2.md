# smc_semantics_v1.1 — 2024-01 Signal Research Diagnostic

> Diagnostic only. This window is not OOS, cannot fit a Brain artifact, and cannot authorize trading.

- Artifact status: `frozen_development_diagnostic_not_oos_not_trading_authority`
- Snapshot authority: `atomic_event_reducer`
- Replayed real 1m rows: 30,477
- Canonical atomic events: 157,802
- Audit events retained: 471,045
- Deterministic audit fingerprint: `a4e412d60eb518bc93bcfdf1e39cbc15b6b5be25ab7fde131f78c3e5e8ceb4ec`

## Nested event chain

| Stage | Signals | Resolved | Laplace rate | Δ vs prior | Min n=30 |
|---|---:|---:|---:|---:|---:|
| C0_matched_control | 6991 | 6953 | 0.495 | — | yes |
| E1_level_touch | 6991 | 6931 | 0.465 | -0.029 | yes |
| E2_sweep | 2942 | 2916 | 0.505 | +0.040 | yes |
| E3_sweep_displacement | 0 | 0 | 0.500 | -0.005 | no |
| E4_sweep_displacement_mss | 0 | 0 | 0.500 | +0.000 | no |
| E5_plus_fvg | 0 | 0 | 0.500 | +0.000 | no |
| E6_plus_parent_alignment | 0 | 0 | 0.500 | +0.000 | no |

## Atomic event studies

| Event | Signals | Resolved | Laplace rate |
|---|---:|---:|---:|
| acceptance_confirmed | 17101 | 16975 | 0.510 |
| dealing_range_activated | 0 | 0 | 0.500 |
| dealing_range_created | 41 | 0 | 0.500 |
| dealing_range_invalidated:forming_close_before_activation | 35 | 0 | 0.500 |
| dealing_range_invalidated:forming_source_invalidated | 0 | 0 | 0.500 |
| dealing_range_invalidated:forming_maturity_deadline_elapsed | 6 | 0 | 0.500 |
| dealing_range_invalidated:active_acceptance | 0 | 0 | 0.500 |
| dealing_range_replaced | 41 | 0 | 0.500 |
| displacement_observed | 240 | 240 | 0.512 |
| fvg_created | 1297 | 1284 | 0.522 |
| fvg_fully_filled | 675 | 663 | 0.489 |
| fvg_invalidated | 587 | 580 | 0.485 |
| fvg_midpoint_touched | 396 | 395 | 0.509 |
| fvg_partially_filled | 357 | 354 | 0.525 |
| level_penetrated | 29993 | 29676 | 0.519 |
| level_touched | 32923 | 32588 | 0.481 |
| liquidity_level_created | 25937 | 0 | 0.500 |
| mss_core_confirmed | 1765 | 1752 | 0.511 |
| origin_zone_created | 46 | 46 | 0.562 |
| origin_zone_invalidated | 2 | 2 | 0.500 |
| origin_zone_mitigated | 43 | 42 | 0.545 |
| protected_swing_assigned | 1507 | 1495 | 0.508 |
| qualified_bos | 1988 | 1971 | 0.514 |
| raw_boundary_break | 4579 | 4548 | 0.514 |
| structural_leg_created | 9966 | 9895 | 0.502 |
| structure_direction_confirmed | 2264 | 2250 | 0.478 |
| sweep_confirmed | 12887 | 12764 | 0.506 |
| swing_confirmed | 13126 | 12659 | 0.496 |

## Non-nested comparisons

| Comparison | Population | Signals | Resolved | Laplace rate | Min n=30 |
|---|---|---:|---:|---:|---:|
| mss_prior_sweep_source_linked | with | 32 | 32 | 0.382 | yes |
| mss_prior_sweep_source_linked | without | 195 | 194 | 0.500 | yes |
| mss_prior_displacement_source_linked | with | 20 | 20 | 0.409 | no |
| mss_prior_displacement_source_linked | without | 207 | 206 | 0.490 | yes |
| fvg_prior_displacement_source_linked | with | 234 | 232 | 0.521 | yes |
| fvg_prior_displacement_source_linked | without | 1063 | 1052 | 0.522 | yes |

## Matched controls

- Requested: 29,077
- Matched: 6,991
- Unmatched: 22,086

## Interpretation guardrails

- Rates are structural one-ATR diagnostics before execution costs, fills, stops, or P&L.
- Sparse cells below the preregistered minimum n=30 are retained and marked; they are not evidence of absence or permission to retune.
- Nested stages occur at later confirmation clocks, so deltas measure registered conditional populations, not a causal treatment effect.
- No pseudo-level or time-shifted control was silently added; those remain explicit follow-up studies.

