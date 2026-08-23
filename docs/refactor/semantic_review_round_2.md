# Evidence-Driven SMC Semantic Review — Round 2

Status date: 2026-08-23
Runtime semantic identity reviewed: `smc_semantics_v1.2`
Audience: technical reviewers of Trading Eye / Trading Brain semantics

## Technical summary

The second-round review does **not** validate all 17 SMC concepts as predictive
signals. It separates two questions that must remain independent:

- **Definition validity:** whether an event is unambiguous, causal,
  deterministic, replayable, versioned, and traceable to exact source facts.
- **Empirical validity:** whether that definition adds stable information over a
  matched control, corresponds to a distinct MBO state, survives regime/time
  splits, and ultimately passes unopened OOS evaluation.

The current system has a strong event-sourced base, but the target state is not
complete. This review identified and fixes only definition defects that need no
new market assumption: exact tick-grid normalization, immutable crossing
terminal generations, stronger canonical-event source contracts, and protected
external-regime transition precedence. It deliberately does **not** invent the
missing rules for geometric Swing nesting, Structural Leg path metrics,
internal-structure generations, Order Block decomposition, Structural Dealing
Range, persistent Delivery Phase, FVG expiry, or range extension.

The frozen v1.2 research artifacts remain historical evidence. Current working-
tree hardening does not rewrite their bytes or retroactively promote them. The
January Signal Research diagnostic remains sparse after E2; the June MBO study
supports only Acceptance continuation and Displacement impact for Phase 7
evidence admission. Neither result is predictive, OOS, causal, or trading
authority.

## The 17 concepts do not share one validity status

The table is the primary audit surface because the decision is an exact
concept-by-concept classification, not a trend. “Definition valid” never means
“empirically useful,” and “unsupported” never means that a descriptive primitive
must be deleted.

| # | Semantic | Definition status after this round | Empirical / predictive status | Required next gate |
|---:|---|---|---|---|
| 1 | Confirmed Swing | **Valid after tick/store hardening.** Strict pivot confirmation is causal and the full source-bar window is auditable. The current role hierarchy is a causal role-depth projection, not geometric nesting. | Descriptive coverage reviewed; incremental/predictive value unknown; OOS not tested. | If geometric hierarchy is still required, freeze duration, prominence, time/price containment, tie-break, and late-confirmation reparenting in a new version. |
| 2 | Structural Leg | **Partial.** Direction, amplitude, duration, retracement, and one efficiency measure exist. The requested `close_efficiency`, `extreme_path_efficiency`, `close_MAE`, `wick_MAE`, full path ancestry, and explicit `ATR_at_leg_start` reference do not. | Unknown. | Preregister formulas, units, ATR source event/value, and path-bar ancestry before changing the canonical event. |
| 3 | Candidate Liquidity | **Strong core; lifecycle partial.** Registered source taxonomy and immutable crossing identity are enforced. A fully specified same-level rearm rule is still absent. | Unknown. | Freeze rearm/generation-close rules; do not use a heuristic runtime flag. |
| 4 | Sweep | **Strong after generation uniqueness hardening.** `event_time` remains the crossing occurrence and `known_at` the terminal resolution clock; one generation has one terminal. | Phase 6 MBO: n=24, underpowered, Holm-adjusted p=1. Predictive value unknown. | Independent validation; no threshold relaxation or post-hoc extension. |
| 5 | Acceptance | **Strong.** Penetration, resolution, clocks, direction, candidate generation, and protected-swing invalidation context are exact-source bound. | Phase 6 MBO construct association supported: n=39, mean effect about 0.14522. Predictive value unknown; OOS not tested. | Fit/validate only under an independently frozen design. |
| 6 | Displacement ACTIVE v1.0 | **Strong/frozen.** Continuous episode state, strict-prior baseline, frozen `atr0`, lifecycle, and FVG separation remain intact; store provenance is hardened. | Phase 6 n=75 construct association supported. Score monotonicity is weak (rho about 0.024), so score is not a calibrated probability. Predictive value unknown; OOS not tested. | Preserve v1.0; preregister any new body/TR/volume-z metrics and their ATR/baseline rules in a new version. |
| 7 | Raw Break | **Valid after exact tick normalization.** Wick, close break, and Acceptance remain distinct. | Unknown. | No new detector; study only after upstream structure generations are frozen. |
| 8 | Structure Direction | **Partial, with reducer precedence hardened.** A live exact protected assignment cannot be overwritten by an opposite direction/Q-BOS without its registered Acceptance; MSS may change only internal direction. The detector-side generation lifecycle still needs a versioned redesign. | Unknown. | Freeze persistent external generation and transition feedback to the tracker; do not infer regime from the latest HH/HL snapshot. |
| 9 | Qualified BOS | **Strong after source cross-link hardening.** Qualification uses a prior persistent same-timeframe direction and exact Raw Break ancestry; it cannot simultaneously create the direction it consumes. | Unknown. | Revalidate after the persistent tracker generation is versioned. |
| 10 | Protected Swing | **Strong after provenance/lifecycle hardening.** Assignment binds Swing → Leg → BOS; invalidation binds the exact live assignment and opposite Acceptance; monotonicity remains. | Survival study not run. | Freeze assignment-time covariates, competing risks, horizon, pre-treatment matching, and repeated-assignment policy. |
| 11 | MSS Core | **Partial.** Sweep, Displacement, and FVG remain context, not definition. The current event does not yet prove an explicit `internal_structure_generation_id` or first terminal within that generation. | OHLCV comparison 48.08% with Sweep vs 48.28% without; Phase 6 MBO n=2. Underpowered/predictive unknown. | New semantic version with explicit generation start/terminal/transition identity; do not infer the missing remainder of the supplied brief. |
| 12 | FVG | **Geometry/lifecycle core strong after tick normalization; expiry deliberately undefined.** | Phase 6's historical first-concrete-lifecycle proxy (originally labelled `fvg_retest_response`) had n=31 and was unsupported; it was not a geometric first-retest estimand. This does not invalidate the descriptive primitive itself. | Run a newly preregistered true first-retest study; define `FVG_EXPIRED` only with an explicit clock, threshold, reset, and transition precedence. |
| 13 | Order Block / Origin Zone | **Partial and over-composite.** Current production output is a qualified origin-zone composite: active displacement + exact BOS/MSS relation + break membership + origin cluster. Terminal provenance is fail-closed. | Lifecycle counts only; matched causal first-retest study not run. | Version and publish base origin geometry separately from impulse/BOS qualification and retest lifecycle; freeze method before implementation. |
| 14 | Active Dealing Range | **Definition conflated.** The existing H1 detector implements a sparse two-sided **Mature Balance Range**, while event/state names and downstream consumers treat it as a generic Structural Dealing Range. | 448 forming candidates produced only 2 mature ranges in the bounded natural scan. This evidence belongs to Balance Range only. | New version must split Structural Dealing Range and Balance Range identities/events/consumers before IRL/ERL or Brain balance evidence can be promoted. |
| 15 | Premium / Discount / IRL / ERL | **Mathematically valid; production binding blocked.** Normalized location and boundary membership are deterministic, but their market meaning depends on the conflated range authority. | Unknown. | Rebind only after the range split; do not reinterpret historical Balance Range artifacts. |
| 16 | Delivery Phase | **Partial.** It is currently a deterministic snapshot projection recomputed from structure/leg/range, not a persistent independent generation with entered-at and transition source DAG. | Unknown. | Freeze phase states, transition events, source authority, and precedence after the range split. |
| 17 | DOL Candidate | **Candidate fact boundary strong; ranking integration partial.** Eye remains non-decisional and no-target mass is preserved. The compatibility ranking still consumes the legacy two-level rank rather than the event-sourced four-role projection. | Probability model not fitted, calibrated, validated, or admitted; shadow-only and action authority false. | Complete the `MarketSnapshot + events` sole-input migration, range/rank feature mapping, and independent calibration before promotion. |

## Scope, evidence, and claim classes

This review uses only already-authorized development evidence and current code:

- The supplied round-two brief, which is visibly truncated after the MSS
  evidence sentence. No absent sections or thresholds were reconstructed.
- The frozen Signal Research v1.2 protocol-v3 r2
  [manifest](../../experiments/manifests/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.yaml)
  (SHA-256 `830eed086ccc0cfd2d16245437583d80523f3dcbb2155f72aa0fe1ac8719052c`)
  and [result](../../experiments/results/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.json)
  (SHA-256 `378379df57b9efebc1ed134bfb9382aee20161088d7eefee997cf5048551ac64`).
  The completed 36,000-clock diagnostic contains 157,802 atomic events and E1–E6
  counts 1,124 / 317 / 17 / 1 / 1 / 0. Quiet/non-sweep controls match 372 and
  62; pseudo/forward-shift controls match zero. No Holm comparison rejects.
- The final Phase 6 two-week MBO [manifest](../../experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml)
  (SHA-256 `99901b1893cdea71615239fd9c536a318ec3c8088f0c491e48cb710b9d0eeafc`)
  and [result](../../experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.json)
  (SHA-256 `98f0f334cbae050a093bebca4cfbb85fcc877e96ba729fe04ea477e2761ddf99`).
  Its Phase 7 allowlist is exactly Acceptance continuation and Displacement
  impact; no Week 3 is authorized.
- The v1.2 [semantic registry](../../semantics/registry_v1_2.yaml),
  [parameters](../../semantics/parameters_v1_2.yaml), current producer,
  immutable event store, reducers, Brain adapter, and focused replay tests.
- The bounded 2023 [natural-authority evidence](../natural_authority_2023.md),
  including the sparse Mature Balance Range census.

The grain differs by evidence source. Signal Research measures canonical
treatment episodes and matched control pairs over completed M5 outcomes; Phase
6 measures MBO mechanism pairs over two registered development weeks; the 2023
scan measures outcome-blind detector coverage. None is a trade, P&L, fitted
posterior, or sealed OOS observation.

## Method: definitions were tested independently of outcomes

For each concept the review checked:

1. first causal availability (`event_time <= known_at`);
2. integer tick-grid authority for raw prices;
3. deterministic identity and version binding;
4. exact event/data/entity source namespaces and recursive source closure;
5. immutable lifecycle and generation terminality;
6. replay and checkpoint behavior;
7. parent/child and timeframe authority isolation;
8. whether a reported effect is descriptive, inferential, predictive, causal,
   or OOS; and
9. whether the existing evidence measures the concept actually named.

The implementation rule was deliberately narrow: strengthen an existing
authority when the correct behavior follows from the frozen contract; otherwise
record a versioned design gate. No second Eye, Brain, detector stack, research
framework, or execution path was introduced.

## Definition defects closed without new market assumptions

The current working tree closes four authority seams:

- raw OHLC prices are admitted only on the exact integer tick grid before
  detector state mutates; shared integer-ratio conversion replaces banker-
  rounded comparison and is independent of binary-float and Decimal-context
  precision;
- one Crossing Generation can terminate exactly once across single append,
  batch append, replay, and restore;
- the immutable store independently verifies source structure and identity for
  Swing, Leg, Candidate Liquidity, Displacement, Qualified BOS, Protected Swing,
  MSS, and protected invalidation rather than trusting only the producer; and
- the timeframe reducer preserves an intact protected external regime until its
  exact registered Acceptance. An opposed MSS may change internal direction,
  while a conflicting opposed Qualified BOS is rejected atomically.

The final independent integrated review reported P0 = 0 and P1 = 0. It ran
453 unique focused and engine/checkpoint tests, including explicit replays of
the earlier tick, lifecycle, source-contract, cross-timeframe custody, atomic
rollback, and restore failures. One subsequent targeted regression also proves
that the producer rejects a protected assignment whose `known_at` is later
than the crossing resolution, so even a manually corrupted private custody
index cannot backfill future context.

These are contract hardening changes. They do not alter frozen result files or
claim that current historical artifacts were contaminated; dedicated audits
found no dual terminal Crossing Generation in the r2 ledger.

The audit did find a narrower historical custody defect: 857 r2 Acceptances
carry protected-assignment context but map to only 807 assignment event IDs, so
50 assignments are referenced by a second later Crossing Generation. The frozen
artifact remains unchanged, and those extra context links are not reclassified
post hoc. Current producer/reducer hardening consumes one live assignment once
and prevents a stale context from terminalizing a later regime. Consequently,
the r2 bundle must not be reused as Protected-Swing survival evidence; that
study already requires a new frozen design and fresh version-bound output.

## The main remaining work is definition-gated, not coding backlog

The following work must not be implemented until its semantic contract is
frozen:

1. **Semantic vNext foundation:** geometric Swing containment; explicit
   Structural Leg path/ATR references; persistent external and internal
   structure generations; and complete path ancestry for multi-bar resolution.
2. **Range split:** preserve existing Mature Balance Range evidence under its
   real identity, then separately define Structural Dealing Range, IRL/ERL
   authority, and persistent Delivery Phase transitions.
3. **Zone decomposition:** separate base origin geometry from displacement,
   BOS/MSS qualification, and first-retest lifecycle; separately define FVG
   expiry if it is still desired.
4. **Independent empirical programs:** Protected-Swing survival, matched
   Origin-Zone first retest, a true FVG first-retest study, and any new path
   metrics.
5. **Brain promotion:** make `MarketSnapshot + events` the sole input, freeze
   scope-retirement semantics, fit and independently validate path/DOL/outcome
   artifacts, and keep default production at zero intents until all are admitted.
6. **Execution and operations:** finish method-price plus wait/cancel/stop/target
   provenance, run the formal paired execution study, and then run the real-time
   multi-day no-order pilot before OOF/OOS.

## Limitations and robustness boundary

- The supplied brief ends mid-MSS status. This report covers all 17 concepts
  named in the visible summary, but it does not claim compliance with unseen
  detailed instructions.
- The current role-depth hierarchy is causal and replay-stable, but it is not
  the continuous prominence/duration/nesting hierarchy described as a target.
- Existing range files and events are frozen historical contracts. Renaming
  their bytes in place would destroy reproducibility; the correct fix is a new
  semantic version and explicit migration.
- The current registered Group 3/FVG/Origin-Zone surface is deliberately NQ-
  specific and freezes a 0.25 tick. The shared normalizer and core structure/
  displacement utilities accept other exact grids, but this review does not
  claim full-stack multi-tick instrument support.
- The r2 Signal Research run is complete as a diagnostic, not accepted as a
  semantic or predictive validation. Sparse E3–E6 and empty control families
  cannot be repaired by post-hoc thresholds.
- Phase 6 is development association evidence. Only its exact allowlist can
  enter Phase 7 evidence admission, and those effects are not likelihoods.
- The repository working tree contains uncommitted/untracked implementation and
  large hash-bound evidence. A reproducible source snapshot plus Git LFS or an
  immutable artifact receipt is still required before ordinary publication.

## Recommended next decision

Do not start another broad refactor or open the sealed holdout. First review and
freeze one coherent semantic vNext proposal covering the persistent structure
generation and range split, with exact event schemas and migration rules. In
parallel, retain the current v1.2 artifacts as immutable development evidence
and complete only the already-defined engineering/empirical gates listed above.

Further questions that must be answered in preregistration—not in runtime code—
are: the precise Swing containment unit and reparent policy; Structural Leg ATR
reference; same-level liquidity rearm rule; structural-range anchors and
extension semantics; Delivery Phase transition precedence; and the estimands,
controls, horizons, and competing-risk rules for each remaining study.
