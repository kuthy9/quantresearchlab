# Phase 6 — MBO mechanism validation

> Development association only; not causal, OOS, model-fit, or trading authority.
> Display classification: `non_authoritative_derived_display`.

- Engineering/data status: `pass`
- Experiment: `smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3`
- Study mode: `primary_plus_registered_underpowered_extension`
- Manifest: `experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml` / `99901b1893cdea71615239fd9c536a318ec3c8088f0c491e48cb710b9d0eeafc`
- Raw partition hashes verified: `True`
- MBO feature artifact: `data/processed/mbo_mechanism_phase6_20240609_20240614_v1.parquet` / `31a9b9b7d22d76695af2da648bfc672cf7ae7c5206533df914e3dd3baf85fe43`
- Result identity: `8e69ca54f4a6c8c1ae11878a9a0552c037910c54bfa3478aef443b1a27a6bc77`
- MBO response rows / real OHLCV source rows / registered synthetic no-trade rows: 6,900 / 6,899 / 1
- Synthetic semantic exception gate allowed/rejected: 2 / 0
- Synthetic exception clock scopes: `{"active_registered_window": 1, "hash_bound_prior_week_warmup": 1}`
- Week-2 extension required: `True`
- Registered extension consumed: `True`; further extension authorized: `False`
- Displacement monotonicity reporting: `{"combined": null, "combined_unavailable_reason": "registered compact week-1 ledgers do not bind every displacement score/response observation", "current_week2": {"n": 48, "spearman_rho": 0.02405916546404182, "threshold_selected": false}, "holm_included": false, "phase7_evidence_admission": false, "policy": "report_hash_bound_prior_week1_and_current_week2_separately;combined_null_because_compact_ledgers_do_not_bind_all_episode_responses;excluded_from_holm_and_phase7", "prior_week1": {"n": 53, "spearman_rho": 0.02496079694770474, "threshold_selected": false}, "threshold_selected": false}`

| Fixed hypothesis | Status | Matched n | Mean effect | Holm p | Stable |
|---|---|---:|---:|---:|---:|
| sweep_rejection | underpowered | 24 | 0.171092 | 1.000000 | True |
| acceptance_continuation | supported | 39 | 0.145220 | 0.000000 | True |
| displacement_impact | supported | 75 | 0.019649 | 0.000002 | True |
| mss_flow_shift | underpowered | 2 | 0.005811 | 1.000000 | False |
| fvg_retest_response | unsupported | 31 | -0.000116 | 1.000000 | False |

## FVG pseudo-zone descriptive sensitivity

This comparison is independently matched and descriptive only. It is excluded from Holm correction and cannot enter the Phase-7 allowlist.

- Packed n: 5; mean effect: -0.05419648760907079; descriptive CI: [-0.1814238743449298, 0.06586995169279206]

## Compact audit ledgers

- episodes: `experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.episodes.jsonl` — 4,390 rows — `75880c29cfcf7d8452d314a1cf615dd3b278333792edb2caca1d3c23276b81d3`
- matched_pairs: `experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.matched_pairs.jsonl` — 176 rows — `f312e21f33df2bf1f72eac6d06ae33468ff07e32e7376a97ded257905969e73a`
- unmatched: `experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.unmatched.jsonl` — 2,815 rows — `39d1e00389750ab84c40aec49afc7a612cce5fc8b97a8f41ad54193ae6175742`

## Phase-7 evidence admission

Only statistically supported mechanisms are admitted; raw post-event window values remain retrospective and are never live evidence.

Allowlist: `acceptance_continuation, displacement_impact`

The displayed-defense net-add metric is an all-book A−C−passive-F proxy. It is not proof of same-level queue replenishment or absorption.

## Limitations

- development association only; no causal claim
- post-event windows are retrospective mechanism validation and are not live evidence
- registered OHLCV synthetic no-trade decision clocks have real MBO/BBO rows with zero trade/fill flow and may appear in retrospective MBO response windows; they cannot become M5 sources/control bases or emit any sample-eligible semantic event
- the sole registered synthetic-context semantic exception is a displacement CENSORED/synthetic_interruption terminal; every exact clock-only M1 root in its M5 constituent interval is context-only, the terminal may settle on the later M5 boundary, and it plus all descendants are excluded from every analysis sample
- FVG pseudo-zone results are descriptive sensitivity only and are excluded from the five-test Holm family and Phase-7 evidence admission
- primary inference uses deterministic earliest-first hypothesis-local packing; formation/post MBO clocks are not reused within a hypothesis and the packing is not maximum-cardinality
- displayed-defense net-add is an all-book A/C/F proxy, not same-level queue replenishment or absorption
- unsupported and underpowered mechanisms are excluded from Phase 7
- in an extension result displacement score monotonicity is reported separately for hash-bound week 1 and current week 2; no combined value is reconstructed from compact ledgers
