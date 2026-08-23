# Evidence-Driven SMC Semantic Review — Round 2

Status date: 2026-08-23
Runtime semantic identity reviewed: `smc_semantics_v1.2`
Additive foundation identity: `smc_semantic_foundation_v2.0`
Audience: technical reviewers of Trading Eye / Trading Brain semantics

## Technical summary

The second-round review does **not** validate all 17 SMC concepts as predictive
signals. It separates two questions that must remain independent:

- **Definition validity:** whether an event is unambiguous, causal,
  deterministic, replayable, versioned, and traceable to exact source facts.
- **Empirical validity:** whether that definition adds stable information over a
  matched control, corresponds to a distinct MBO state, survives regime/time
  splits, and ultimately passes unopened OOS evaluation.

Round 2 first fixed only definition defects that needed no new market
assumption: exact tick-grid normalization, immutable crossing terminals,
stronger source contracts, and protected-regime precedence. The later
[Canonical Semantic Foundation v2](canonical_semantic_foundation_v2.md) now
closes the requested geometry, generation, lifecycle, reinteraction, ancestry,
and factual-outcome definitions as an additive projection over those unchanged
v1.2 facts. It does not invent an FVG time TTL or range-extension rule.

The frozen v1.2 research artifacts remain historical evidence. Later versioned
source hardening does not rewrite their bytes or retroactively promote them. The
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
| 1 | Confirmed Swing | **Valid after tick/store hardening.** Strict pivot confirmation is causal and the full source-bar window is auditable. Foundation v2 adds a separate purely geometric containment tree; role depth remains independent. | Descriptive coverage reviewed; incremental/predictive value unknown; OOS not tested. | Test whether geometry adds information; do not infer semantic importance from depth. |
| 2 | Structural Leg | **Definition complete in foundation v2.** It freezes full native-bar ancestry, tick/ATR amplitude, two efficiencies, close/wick MAE, duration, and 14 strictly-prior BAR sources for `ATR_at_leg_start`. | Unknown. | Run a fresh version-bound construct study; historical v1.2 legs do not contain these features. |
| 3 | Candidate Liquidity | **Definition/lifecycle complete in foundation v2.** Source identity is retained across immutable interaction generations; Sweep can enter a registered departure-based rearm path while Acceptance/retirement remains terminal. | Unknown. | Validate source-family rearm and retirement effects; do not tune rules on outcomes. |
| 4 | Sweep | **Strong after generation uniqueness hardening.** `event_time` remains the crossing occurrence and `known_at` the terminal resolution clock; one generation has one terminal. | Phase 6 MBO: n=24, underpowered, Holm-adjusted p=1. Predictive value unknown. | Independent validation; no threshold relaxation or post-hoc extension. |
| 5 | Acceptance | **Strong.** Penetration, resolution, clocks, direction, candidate generation, and protected-swing invalidation context are exact-source bound. | Phase 6 MBO construct association supported: n=39, mean effect about 0.14522. Predictive value unknown; OOS not tested. | Fit/validate only under an independently frozen design. |
| 6 | Displacement ACTIVE v1.0 | **Strong/frozen.** Continuous episode state, strict-prior baseline, frozen `atr0`, lifecycle, and FVG separation remain intact; store provenance is hardened. | Phase 6 n=75 construct association supported. Score monotonicity is weak (rho about 0.024), so score is not a calibrated probability. Predictive value unknown; OOS not tested. | Preserve v1.0; preregister any new body/TR/volume-z metrics and their ATR/baseline rules in a new version. |
| 7 | Raw Break | **Valid after exact tick normalization.** Wick, close break, and Acceptance remain distinct; foundation v2 separately retains a wick-only Boundary Attack. | Unknown. | Study under the frozen generation identity; do not turn Boundary Attack into BOS/MSS. |
| 8 | Structure Direction | **Persistent lifecycle complete in foundation v2.** External generations persist to an exact termination cause; internal challengers remain forming until a separate confirmation fact. Snapshot direction remains a projection, not generation identity. | Unknown. | Study generation age and child evidence without treating snapshots as independent samples. |
| 9 | Qualified BOS | **Strong after source cross-link hardening.** Qualification uses a prior persistent same-timeframe direction and exact Raw Break ancestry; it cannot simultaneously create the direction it consumes. | Unknown. | Revalidate in a study explicitly bound to the frozen foundation-v2 Structure Generation. |
| 10 | Protected Swing | **Strong after provenance/lifecycle hardening.** Assignment binds Swing → Leg → BOS; invalidation binds the exact live assignment and opposite Acceptance; monotonicity remains. | Survival study not run. | Freeze assignment-time covariates, competing risks, horizon, pre-treatment matching, and repeated-assignment policy. |
| 11 | MSS Core | **Transition-evidence boundary complete in foundation v2.** MSS starts or updates one forming internal challenger and one external transition; it never confirms an opposite regime by itself. Confirmation/failure/censoring are immutable later facts. | OHLCV comparison 48.08% with Sweep vs 48.28% without; Phase 6 MBO n=2. Underpowered/predictive unknown. | Study `P(transition confirmation | MSS)` on fresh generation-bound episodes. |
| 12 | FVG | **Geometry, structural lifecycle, and true first-retest definition complete.** Age remains continuous; invalidation and structural/reset expiry are distinct; no arbitrary TTL exists. | Phase 6's historical first-concrete-lifecycle proxy (originally labelled `fvg_retest_response`) had n=31 and was unsupported; it was not the new geometric first-retest estimand. | Run a newly preregistered true first-retest study; only empirical evidence could justify a later TTL. |
| 13 | Order Block / Origin Zone | **Decomposed in foundation v2.** Base Origin Core freezes outcome-blind geometry; Qualified OB separately binds active Displacement and compatible BOS/MSS. First reinteraction waits for a strictly future departure and return. | Lifecycle counts only; matched causal first-retest study not run. | Run the fresh version-bound study; do not relabel historical composite counts. |
| 14 | Active Dealing Range | **Identities split in foundation v2.** The legacy detector remains historical Mature Balance Range; Structural Range is a separate structure-owned geometry and both may coexist. | 448 forming candidates produced only 2 mature Balance Ranges in the bounded natural scan. This evidence does not validate Structural Range. | Study each range independently and preserve the legacy evidence label. |
| 15 | Premium / Discount / IRL / ERL | **Dual location definition complete.** Foundation publishes independent `x_structural_range` and `x_balance_range`; premium/discount belongs primarily to Structural Range. | Unknown. | Validate location features without pooling the two range types. |
| 16 | Delivery Phase | **Persistent generation complete in foundation v2.** It reuses the deterministic classifier while retaining parent Structure Generation, entry/update/terminal clocks, age, extrema, and next phase. | Unknown. | Study phase age and transitions; no fitted inference exists. |
| 17 | DOL Candidate | **Candidate fact boundary strong; ranking integration partial.** Eye remains non-decisional and no-target mass is preserved. The public target map now honors foundation lifecycle eligibility, including Generation-2 rearm and Acceptance/retirement exclusion; the compatibility Brain still has legacy observation/rank inputs. | Probability model not fitted, calibrated, validated, or admitted; shadow-only and action authority false. | Complete the `MarketSnapshot + events` sole-input migration, freeze remaining range/rank features, and independently calibrate before promotion. |

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

The versioned v1.2 hardening closes four authority seams:

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

## The main remaining work is empirical, not more vocabulary

Foundation v2 closes the semantic-definition items previously listed here.
The following work remains deliberately outside this implementation:

1. **Independent empirical programs:** Protected-Swing survival, matched
   Origin-Zone first retest, a true FVG first-retest study, and any new path
   metrics.
2. **Brain promotion:** make `MarketSnapshot + events` the sole input, freeze
   scope-retirement semantics, fit and independently validate path/DOL/outcome
   artifacts, and keep default production at zero intents until all are admitted.
3. **Execution and operations:** finish method-price plus wait/cancel/stop/target
   provenance, run the formal paired execution study, and then run the real-time
   multi-day no-order pilot before OOF/OOS.
4. **Future definition changes:** any empirical FVG TTL, Structural Range
   extension, or new tutorial SMC object requires a new registry version. It
   cannot be silently patched into this foundation.

## Limitations and robustness boundary

- The historical round-two brief audited by this report ended mid-MSS status.
  The later, complete foundation-v2 request is governed by the separate
  foundation specification and current implementation status; this review does
  not retroactively invent content for the earlier brief.
- Role depth and geometric depth are now distinct replay-stable fields. No
  relationship between them has been empirically established.
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
- Large historical ledgers remain hash-bound evidence and are not copied into
  ordinary Git merely as cleanup. Their existing receipts remain the durable
  authority boundary.

## Recommended next decision

Do not start another broad refactor or open the sealed holdout. Freeze the
foundation-v2 source snapshot, retain v1.2 artifacts as immutable development
evidence, and move to the already-defined empirical gates without adding more
canonical SMC vocabulary.

Further questions that must be answered in preregistration—not runtime
heuristics—are the empirical estimands, controls, horizons, competing risks,
and any future expiry/extension threshold. The implemented containment, ATR,
rearm, range, and Delivery contracts are frozen in the v2 registry.
