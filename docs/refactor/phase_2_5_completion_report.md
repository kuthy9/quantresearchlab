# Phase 2–5 Refactor Completion Report

Report date: 2026-08-20
Last documentation and repository-hygiene review: 2026-08-21

> Historical v1.1 report. This file preserves the exact January 2024
> protocol-v2 artifact boundary, counts, hashes, limitations, and acceptance
> decision as recorded. It is not the current runtime implementation status.
> v1.2 Swing hierarchy/IRL-ERL, research-protocol-v3 tooling, the shadow
> path/DOL Brain, and the additive canonical foundation v2 were added later. Use
> [Current SMC Refactor Implementation Status](current_implementation_status.md)
> for the current 1–20 and phase matrices. No statement below is retroactively
> rewritten as v1.2 evidence.

## Verdict and Authority Boundary

The executable core of Phase 2, Phase 3, and Phase 4 is complete and is wired
into the production `CausalObserver` path. Phase 5 has also completed one full
January 2024 OHLCV diagnostic under a frozen v1.1 manifest.

This verdict is deliberately narrower than “the trading system is complete”:

- The January window is registered as `brain_calibration_trial`. It is not an
  out-of-sample test, cannot fit or admit an artifact, and has no trading
  authority.
- The current Brain has reusable multi-candidate beliefs, but it is not the
  requested normalized Bayesian manager of competing path hypotheses.
- MBO mechanism validation, Execution Research, an explicit order FSM, Shadow
  Live, and final OOS testing remain later phases.
- Several v1.1 semantics are explicitly reserved or only partially executable;
  those limits are listed below rather than being treated as completed.

The authoritative frozen run is:

- Manifest: `experiments/manifests/smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.yaml`
- Manifest SHA-256: `295daddb13f51fb6daa219cfecb78a625dab7e4f0373dd7fd93a437da3f1483b`
- Semantic version: `smc_semantics_v1.1`
- Semantic definition identity: `bc53735c98a14f4f1770687302c4be80686564193c9d85e39bee1e5eb7b0bba5`
- Result: `experiments/results/smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.json`
- Result identity: `2f0193258ce9379f9ba26e9708a63882ad5ee2ad3450661ca74e1a33a9ec1097`
- Result-file SHA-256: `5f05bf1d01b60452c6b1aa053299646978d6cc076f0e73c3ab02ac530b1aa784`

The earlier v1 manifest remains immutable. Its full run failed before artifact
creation when the second `synthetic_no_trade` minute exposed a mixed real/bar
clock defect in Swing ancestry. The v2 manifest binds the repaired Observer
hash and was frozen before the successful rerun.

## Delivered Architecture

The incremental Eye and replay path is now:

```text
normalized completed M1 facts
  -> immutable normalized BAR_COMPLETED roots
  -> immutable SEMANTIC_ATOMIC events
  -> definition-bound EventStore
  -> TimeframeEventReducer for each timeframe
  -> independent RelationResolver and SessionStateReducer
  -> MarketSnapshot with atomic_event_reducer authority
```

`STATE_PROJECTION` remains an optional transport and is not an authoritative
input. The Phase 5 diagnostic disables projection persistence.

The repository was evolved in place instead of mechanically duplicating all
recommended top-level directories. `semantics/`, `experiments/`, the event
store, hierarchical state, research runner, and tests were added around the
existing `smc_trader` implementation. This follows the reuse-first constraint
and avoids maintaining a second Eye/Brain/Execution stack.

## Implementation Plan Status

Status is measured against the requested architecture, not against whether a
class or filename merely exists. “Complete within v1.1” means the registered
executable boundary is wired into production and replay; it does not imply OOS
value or trading authority.

| Plan item | Status | Current boundary |
|---|---|---|
| 1. Eye / Brain / Executor definitions | **Definitions complete; implementation split** | The ownership boundary is documented and enforced for the Eye. The requested Brain and Executor implementations remain later work. |
| 2. Event-sourced hierarchical market state | **Complete within v1.1** | Immutable atomic facts reduce into dimensioned per-timeframe state; there is no combinatorial master enum. |
| 3. Semantic provenance | **Complete for emitted canonical atoms** | `event_time`, `known_at`, semantic identity, origin, separated provenance namespaces, closed ancestry, and deterministic order are enforced. Reserved non-emitted kinds are not misreported as events. |
| 4. Preregistered semantic system | **Complete as a v1.1 contract; semantic coverage partial** | The registry records definitions, hypotheses, falsification, OOS criteria, bindings, and emitted/reserved status. Swing hierarchy, IRL/ERL, and several reserved lifecycle kinds remain outside the executable surface. |
| 5. Eye organization | **Complete functionally** | Existing normalization and detectors were reused; event store, reducer, relation/session reducers, and snapshot publisher are connected without creating a duplicate Eye stack. |
| 6. Parent/child timeframe rules | **Complete** | Parent facts change only on parent events; child reversal evidence becomes a retracement/warning until parent invalidation. |
| 7. Independent relation object | **Complete** | `RelationState` is separate from both timeframe states and carries its own source cutoffs and known clock. |
| 8. Cross-timeframe Session object | **Complete** | `SessionState` is reduced from the M1 clock and is not embedded in a timeframe. |
| 9. Multiple competing Brain hypotheses | **Partial** | The legacy Brain retains multiple typed candidates, raw scores, and calibration machinery, but no ready calibration artifact or normalized Bayesian/log-odds competing path posteriors with a prior/evidence ledger. |
| 10. Signal Research / Execution Research separation | **Boundary complete; research partial** | Phase 5 constructs only Reader + Eye and structural outcomes. Execution Research has not started. |
| 11. Nested and non-nested comparisons | **Framework complete; evidence partial** | Both designs execute. The frozen nested chain has observations only through E2; E3–E6 have zero samples. |
| 12. Matched controls | **Partial** | Exact matched controls are persisted for 6,991 of 29,077 requested touches (24.04%). Pseudo-level and time-shifted controls remain unfinished. |
| 13. Structural outcomes before P&L | **Substantial but partial** | Target/invalidation, path, time, structural-event, FVG, MFE/MAE, and half-life outputs exist. Protected-swing survival and matched Origin-Zone first-retest work remain. P&L was deliberately excluded. |
| 14. OHLCV geometry / MBO mechanism split | **OHLCV complete for this diagnostic; MBO incomplete** | January research validates only price geometry. OFI/aggressor/replenishment/absorption/impact joins and E7 are not implemented as Phase 6 evidence. |
| 15. Arrow-by-arrow causal-chain tests | **Partial** | Atomic populations and source-linked C0–E2 are auditable; later arrows to Displacement/MSS/FVG/parent target are not validated by this run. |
| 16. Experiment preregistration | **Complete for the v2 diagnostic** | Definitions, split, outcome, controls, search space, success rules, census, and 34 exact code/data bindings were frozen before execution. |
| 17. New Brain organization | **Not complete** | No new normalized Hypothesis Manager, Bayesian Belief Updater, validated Signal Policy, or Trade Intent Builder is authoritative. |
| 18. Execution order FSM | **Not complete** | Existing pending/open/exit mechanics are retained, but the requested immutable order/partial-fill/cancel/replace/reconciliation FSM is absent. |
| 19. Recommended directory structure | **Adapted, not mechanically copied** | `semantics/`, `experiments/`, research code, event store, state code, and tests exist; reusable `smc_trader` modules were kept instead of creating parallel `eye/brain/execution` packages. |
| 20. Refactor sequence | **Phases 1–4 complete; Phase 5 diagnostic complete but research conclusions partial; Phases 6–9 incomplete** | Later MBO, Brain, Execution, Shadow Live, and final OOS work remains separately preregistered work. |

## Repository Hygiene Decision

The refactor does not maintain a parallel detector/state stack and no safely
removable core infrastructure was found. The schema-2 research template,
frozen manifests, historical pre-artifact manifests, historical audit reports,
and the formal v2 result plus its three hash-bound ledgers are retained because
they carry executable or audit identity. In particular, the large event-study
ledger is evidence bound by the formal result, not a cache.

The event-study ledger is 430,877,532 bytes, above the ordinary 100 MB object
limit used by common Git hosting. It should be published through Git LFS or an
immutable artifact store before repository publication. Its external copy and
SHA-256 must be verified before any local removal or path change.

The unused experiment-schema-1 template was removed after confirming that the
runner accepts only schema 2 and that no code, test, manifest, result, or
document references it. Generated Python/test/type-check caches and Finder
metadata are safe to regenerate and are removed during the final hygiene pass.
Unrelated pre-existing audit scripts, configurations, data, and outputs are not
part of this refactor and were intentionally left untouched.

## Phase 2: Core Atomic Semantics

The following production-emitted v1.1 atoms have executable detectors or
projectors, immutable event contracts, causal clocks, and replay coverage:

- Confirmed Swing and Structural Leg
- Candidate Liquidity Level, Touch, Penetration, Sweep, and Acceptance
- continuous Displacement observations
- Raw Boundary Break
- FVG creation and concrete fill/invalidation lifecycle transitions

Every canonical event uses an explicit `EventOrigin`, `event_time`, `known_at`,
`semantic_version`, and separated event/data/entity/context provenance
namespaces. Canonical source-event ancestry must be closed, same-version,
causally available, and ordered before the child. Critical semantic kinds also
enforce exact parent-kind contracts.

Sweep and Acceptance are mutually exclusive terminal outcomes for one frozen
crossing generation. The January run emitted 29,993 penetrations, 12,887
Sweeps, and 17,101 Acceptances without admitting a synthetic clock as a
semantic resolution bar.

## Phase 3: Derived Structure

The executable derived surface includes:

- Structure Direction facts
- Qualified BOS separated from Raw Boundary Break
- Protected Swing assignment with exact BOS/Leg/Swing ancestry
- MSS Core without hard-wiring Sweep, Displacement, or FVG into its definition
- Origin Zone creation and terminal lifecycle
- Active Dealing Range creation, activation, invalidation, and replacement
- continuous dealing-range location and deterministic Delivery Phase snapshot

Range invalidation variants are not pooled. The manifest freezes four distinct
variants and their required parent shapes. January observed 35
`forming_close_before_activation` and 6
`forming_maturity_deadline_elapsed` invalidations. It observed no active range
activation, so the active-acceptance path is covered by code and tests but not
by this month’s empirical sample.

## Phase 4: Hierarchical State and Relationships

`TimeframeState`, `RelationState`, and `SessionState` are separate immutable
objects. Parent facts are not copied into child state, and child structure does
not directly rewrite parent direction. A counter-directional child while the
parent protected swing remains intact is resolved as a parent retracement or
warning, not as a timeframe vote.

The snapshot publisher is explicitly authoritative through the atomic event
reducer and fails closed on missing roots, foreign timeframes, registry drift,
contract drift, or an unconsumed epoch boundary. Replay rebuilds timeframe,
relationship, and Session state from normalized and atomic events without
consulting rich frame projection.

Session remains a top-level cross-timeframe context reduced from the completed
M1 clock. It is not stored inside a timeframe state.

### Synthetic-clock repair

The structure/liquidity protocol specifies that a synthetic no-trade minute
advances market time but cannot change semantic state, age, or event clocks.
The original full v1 run found that Observer’s BAR index still included the
synthetic root when selecting the Swing left/pivot/right source window.

The repaired Observer retains two indexes:

- an inclusive normalized index for replay and Session clock continuity; and
- a real-completed-only index for all semantic BAR selection.

Swing windows, exact semantic BAR lookup, Swing/zone next-bar resolution, and
Group 4 terminal guards now use the real-only index. Reset clears both. The
formal v2 run crossed both January synthetic minutes, including the prior
failure at 2024-01-30 00:15 ET, without producing a semantic event on either
synthetic clock.

## Phase 5 Full Diagnostic

### Input and identity census

| Item | Full-run value |
|---|---:|
| emitted bars including warmup | 36,000 |
| diagnostic completed clocks | 30,479 |
| diagnostic real/ready rows | 30,477 |
| diagnostic synthetic clocks | 2 |
| warmup or diagnostic data-gap resets | 0 |
| contract changes | 0 |
| last diagnostic clock | 2024-01-31 23:59 ET |
| last processed clock | 2024-02-01 00:00 ET |
| selected atomic events | 157,802 |
| run-end audit-store events | 471,045 |

The 471,045 audit-event number is a run-end store count with fingerprint
`a4e412d60eb518bc93bcfdf1e39cbc15b6b5be25ab7fde131f78c3e5e8ceb4ec`.
The Phase 5 bundle does not contain a complete audit-event journal, so that
fingerprint is not independently reconstructible from the three research
ledgers alone. It also does not persist the complete ready-row feature ledger
used to form ATR, distance, relative-volume, and M1-direction matching strata;
reconstructing those inputs requires the frozen deterministic replay.

An independent streaming scan found 157,802 unique event IDs, all with origin
`semantic_atomic` and version `smc_semantics_v1.1`. It found 131,364
directional events; all had an outcome. Of those, 131,162 had the full 60-real-
bar path and 202 were retained under the frozen window-end censoring rule.
There were 130,179 resolved events, 1,095 ambiguous events, 90 unresolved
non-ambiguous events, no contract-censored events, and 65,366 successes.

### Research ledgers

| Ledger | Rows | SHA-256 |
|---|---:|---|
| `smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.event_study.jsonl` | 157,802 | `70edb9eb551214719c8b5fad6268da4ee210ab210622fb4fb55abaf9474b72c7` |
| `smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.control_pairs.jsonl` | 6,991 | `5dba1439e03ca747d939c53f664e083927a47d2b860a5d064dec55634c461e6a` |
| `smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.source_chains.jsonl` | 2,942 | `1e9b588f5df4acfbd5eb198e09ee899956c238dd2fdf58e568c59ba60cdd92d1` |

All three paths, row counts, and byte hashes match the result metadata. Pair,
treatment, and control identities are unique. The matched-control cohort is
6,991 of 29,077 requested touches, or 24.04%; this limited coverage must be
considered when interpreting the comparison.

### Descriptive results

The registered primary outcome is a one-ATR target before a one-ATR
invalidation over 60 later completed real M1 bars. These are descriptive
calibration-window estimates, not inferential or OOS evidence.

| Nested stage | Signals | Raw success rate | Increment vs prior Laplace rate |
|---|---:|---:|---:|
| C0 matched control | 6,991 | 49.46% | n/a |
| E1 matched level touch | 6,991 | 46.54% | -2.92 pp |
| E2 source-linked Sweep | 2,942 | 50.51% | +3.97 pp |
| E3 Sweep + Displacement | 0 | unavailable | unavailable |
| E4 + MSS | 0 | unavailable | unavailable |
| E5 + FVG | 0 | unavailable | unavailable |
| E6 + parent alignment | 0 | unavailable | unavailable |

The exact source-linked chain therefore stops at E2 in this dataset. It does
not validate the incremental value of Displacement, MSS, FVG, or parent
alignment in the nested design. Changing the chain after seeing these results
would violate preregistration; any broader linkage rule requires a new frozen
experiment.

The non-nested comparisons did run, but some cohorts are small. For example,
MSS with prior source-linked Displacement has only 20 events, below the frozen
minimum sample requirement of 30. FVG with prior Displacement was 52.16%
versus 52.19% without in this descriptive window. No multiple-testing
correction or causal claim is made.

## Verification

- The v2 manifest has 34 exact path/SHA bindings, all of which matched the
  then-frozen v1.1 checkout when this historical report was produced; later
  v1.2 work is expected to differ.
- The v1 manifest correctly rejects the repaired Observer as identity drift.
- Manifest SHA, semantic identity, dataset SHA, result identity, and all ledger
  hashes were independently recomputed.
- A real-data 3,150-bar replay spanning the second synthetic minute completed
  with 39,658 audit events and no causal error before the full run.
- The full repository test suite exited successfully across 1,709 collected
  test nodes with one existing skip and no failures.
- Focused synthetic Swing/crossing, provenance, reducer, and Phase 5 tests also
  passed.
- Python compilation and `git diff --check` passed.

The test run reports many existing pandas/NumPy generic-`Timedelta`
deprecation warnings. They are maintenance debt, not a logical failure in this
refactor.

## Explicitly Not Completed

The following items must not be inferred from this report:

- Swing micro/internal/structural/external rank and nesting are not yet
  executable; production values remain unresolved/zero.
- IRL/ERL membership is not an executable snapshot field.
- Eye DOL candidates have deterministic rank/strength, but path obstacles are
  empty and the Eye does not assign DOL probabilities.
- `FVG_TOUCHED`, `FVG_EXPIRED`, `ORIGIN_ZONE_TOUCHED`,
  `DEALING_RANGE_EXTENDED`, and `DELIVERY_PHASE_CHANGED` are reserved or
  compatibility concepts and are not production-emitted v1.1 atoms.
- Delivery Phase is snapshot-derived and does not yet use explicit range
  extension or volatility-compression inputs.
- FVG and Origin Zone production detection is currently M5-specific; Active
  Dealing Range production is H1-specific.
- Pseudo-level and time-shifted controls, Holm inference, protected-swing
  survival, and matched Origin-Zone first-retest research remain unfinished.
- Phase 6 MBO mechanism validation is not complete.
- The Brain has not been migrated to normalized Bayesian competing path
  hypotheses, and this run did not construct or evaluate the Brain.
- Signal Policy, Trade Intent, Execution Research, an explicit order state
  machine, Shadow Live, and final OOS testing are not complete.

## Acceptance Decision

Accept Phase 2–4 as the executable, event-sourced hierarchical Eye core and
accept the January 2024 Phase 5 run as a complete frozen diagnostic execution.
Do not use this result to admit a semantic, tune a threshold, fit a Brain
artifact, claim a trading edge, or authorize live orders. The next research
work must be separately preregistered and should first address the zero-sample
E3–E6 lineage, low matched-control coverage, and the reserved semantic fields
listed above.
