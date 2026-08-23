# Phase 6 — MBO mechanism validation

> Development association only; not causal, OOS, model-fit, or trading authority.
> Display classification: `non_authoritative_derived_display`.

- Engineering/data status: `pass`
- Experiment: `smc_semantics_v1_2_phase6_mbo_2024_06_week1_v5`
- Study mode: `primary_week_only`
- Manifest: `experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.yaml` / `77e5c43a157fc9aa5ac27a1f209daa869e07d649284630a186b3a50b71c689c4`
- Raw partition hashes verified: `True`
- MBO feature artifact: `data/processed/mbo_mechanism_phase6_20240602_20240607_v1.parquet` / `9acbee73784cd260b02b30edbcbdd60aad3ec3faa1ae5de9b52a38fbc03d03c7`
- Result identity: `64ecd3067dc9bd41abbdbe01296af1e1ded8b15e86b020ad66ac02ef74b1fb1d`
- MBO response rows / real OHLCV source rows / registered synthetic no-trade rows: 6,900 / 6,899 / 1
- Synthetic semantic exception gate allowed/rejected: 1 / 0
- Week-2 extension required: `True`
- Registered extension consumed: `False`; further extension authorized: `False`

| Fixed hypothesis | Status | Matched n | Mean effect | Holm p | Stable |
|---|---|---:|---:|---:|---:|
| sweep_rejection | underpowered | 13 | 0.200116 | 1.000000 | False |
| acceptance_continuation | underpowered | 20 | 0.138036 | 1.000000 | True |
| displacement_impact | supported | 37 | 0.017521 | 0.012816 | True |
| mss_flow_shift | underpowered | 2 | 0.005811 | 1.000000 | False |
| fvg_retest_response | underpowered | 17 | -0.045374 | 1.000000 | False |

## FVG pseudo-zone descriptive sensitivity

This comparison is independently matched and descriptive only. It is excluded from Holm correction and cannot enter the Phase-7 allowlist.

- Packed n: 1; mean effect: 0.1260420554933433; descriptive CI: [0.1260420554933433, 0.1260420554933433]

## Compact audit ledgers

- episodes: `experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.episodes.jsonl` — 2,205 rows — `1bf9b1ddc1d73b0362a967a7e02194108c87316bb80f1e4980bcf4911d47f54a`
- matched_pairs: `experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.matched_pairs.jsonl` — 90 rows — `483a239478e2d4d8a1f6e608f8f59c1b32e3eca69a84458704d9f35f4fcb6759`
- unmatched: `experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.unmatched.jsonl` — 1,299 rows — `8630928c03ae1b3967ed441ea0784c31251610ed7622228944a4a498b0ca496e`

## Phase-7 evidence admission

Only statistically supported mechanisms are admitted; raw post-event window values remain retrospective and are never live evidence.

Allowlist: `displacement_impact`

The displayed-defense net-add metric is an all-book A−C−passive-F proxy. It is not proof of same-level queue replenishment or absorption.

## Limitations

- development association only; no causal claim
- post-event windows are retrospective mechanism validation and are not live evidence
- registered OHLCV synthetic no-trade decision clocks have real MBO/BBO rows with zero trade/fill flow and may appear in retrospective MBO response windows; they cannot become M5 sources/control bases or emit any sample-eligible semantic event
- the sole registered synthetic-clock semantic exception is a displacement CENSORED/synthetic_interruption terminal whose current clock-only M1 root is context-only; it is audit-recorded and excluded from every analysis sample
- FVG pseudo-zone results are descriptive sensitivity only and are excluded from the five-test Holm family and Phase-7 evidence admission
- primary inference uses deterministic earliest-first hypothesis-local packing; formation/post MBO clocks are not reused within a hypothesis and the packing is not maximum-cardinality
- displayed-defense net-add is an all-book A/C/F proxy, not same-level queue replenishment or absorption
- unsupported and underpowered mechanisms are excluded from Phase 7
