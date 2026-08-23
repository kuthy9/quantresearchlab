# smc_semantics_v1.2 — Signal Research protocol v3

> Development diagnostic only: unvalidated, not OOS, not a model, and not trading authority.

- Status: `frozen_development_diagnostic_not_oos_not_trading_authority`
- Complete registered window: `True`
- Canonical M5 touch episodes: 1124

## Registered M5 evidence chain

| Stage | Episodes | Resolved | Laplace rate | Δ vs prior | Min sample met |
|---|---:|---:|---:|---:|---:|
| E1_level_touch | 1124 | 1114 | 0.498 | — | yes |
| E2_sweep | 317 | 316 | 0.547 | +0.049 | yes |
| E3_sweep_displacement | 17 | 17 | 0.579 | +0.032 | no |
| E4_sweep_displacement_mss | 1 | 1 | 0.667 | +0.088 | no |
| E5_plus_fvg | 1 | 1 | 0.333 | -0.333 | no |
| E6_plus_parent_alignment | 0 | 0 | 0.500 | — | no |

## Non-nested typed comparisons

| Comparison | Population | Episodes | Resolved | Laplace rate |
|---|---|---:|---:|---:|
| mss_prior_sweep_composition_linked | with | 52 | 52 | 0.481 |
| mss_prior_sweep_composition_linked | without | 175 | 174 | 0.483 |
| mss_prior_displacement_composition_linked | with | 11 | 11 | 0.462 |
| mss_prior_displacement_composition_linked | without | 216 | 215 | 0.484 |
| fvg_prior_displacement_composition_linked | with | 206 | 205 | 0.527 |
| fvg_prior_displacement_composition_linked | without | 1091 | 1079 | 0.521 |

## Separate control families

| Control | Requested | Matched | Exact McNemar p | Holm p |
|---|---:|---:|---:|---:|
| quiet_zero_event | 1124 | 372 | 0.8862 | 1.0000 |
| same_session_non_sweep_touch | 1124 | 62 | 0.7111 | 1.0000 |
| pseudo_level_touch | 1124 | 0 | 1.0000 | 1.0000 |
| forward_time_shift | 1124 | 0 | 1.0000 | 1.0000 |

## Guardrails

- E1→E2 requires semantic-event ancestry; later edges require a shared exact canonical M5 BAR and are composition evidence, not causal ancestry.
- Quiet, non-sweep-touch, pseudo-level, and forward-time-shift controls are separate populations and are never pooled.
- Missing or underpowered fixed-family tests enter Holm with p=1.
- Cross-pair outcome windows may overlap; McNemar/Holm p-values remain descriptive and unvalidated.
- Any `--max-bars` run remains an incomplete smoke artifact permanently.

