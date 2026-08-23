# Phase 2-4 Canonical Semantic Event Audit

> **Historical / superseded audit snapshot (2026-08-20).** This is the
> pre-repair audit that drove the Phase 2–4 changes. Its defects, line references,
> bounded-replay counts, and failure verdict are retained as evidence of what
> was inspected, not as the current implementation status. Some findings have
> since been repaired and others remain explicitly reserved. Current acceptance
> must be read from the v1.2 runtime contract and the
> [Current SMC Refactor Implementation Status](current_implementation_status.md);
> the [Phase 2–5 completion report](phase_2_5_completion_report.md) is the
> historical v1.1 January-run record. Results and fingerprints from this audit
> are not portable to the v1.1 or v1.2 identities.

Audit date: 2026-08-20
Scope: read-only review of the current Phase 2/3 canonical semantic event surface and its Phase 4 reducer inputs. This document does not certify Phase 2, 3, or 4 as complete and does not treat the January 2024 data as out-of-sample evidence.

## Verdict

The repository has broad **type and emitter coverage**, but the canonical semantic contract is **not acceptance-ready**. Confirmed Swing, Structural Leg, Candidate Level, touch/penetration, Sweep/Acceptance, Raw Break, FVG transitions, Displacement, Qualified BOS, Protected Swing assignment, MSS Core, Dealing Range transitions, and Origin Zone transitions all have concrete implementations. Delivery Phase and Dealing Range `extended` are declared but are not emitted as immutable canonical transitions.

The principal blocker is provenance: `source_event_ids` is currently only an alias for the generic `source_ids` tuple and frequently contains candle or entity identities which do not resolve to stored events. Consequently, the event stream is hash-stable and replayable at the container level, but it is not yet a referentially closed, auditable semantic DAG. There are also competing canonical Sweep/Acceptance emitters and compatibility aliases marked canonical, so a single market occurrence can be counted more than once.

Acceptance status: **fail pending the P0 repairs below**.

## Method and Evidence Boundary

The audit used:

- static inspection of the registry, parameter freeze, event model, detector emitters, lifecycle trackers, reducer, and research lineage resolver;
- the current semantic/reducer test subset; and
- a bounded diagnostic replay of 600 completed one-minute rows beginning 2024-01-02, used only to expose contract defects. The replay is neither a Phase 5 study nor an OOS result.

The following test command passed 36 tests:

```text
.venv/bin/python -m pytest -q \
  tests/test_semantic_event_contract.py \
  tests/test_hierarchical_market_state.py \
  tests/test_phase1_semantic_identity_journal.py \
  tests/test_phase234_atomic_reducer.py
```

Those tests establish generic clock validation, immutability/fingerprinting, reducer ordering, and deterministic replay. They do **not** establish source-event referential closure, uniqueness of semantic occurrence, or per-kind evidence completeness. In fact, fixtures explicitly accept `pivot-candle`, `candle-2`, and `candle-3` inside `source_ids` (`tests/test_semantic_event_contract.py:21-41,92-101`; `tests/test_phase1_semantic_identity_journal.py:33-80`).

## Cross-Cutting Contract Audit

| Contract field | Current state | Evidence | Required correction |
|---|---|---|---|
| `semantic_type` | Present | Canonical kinds are declared in `smc_trader/model.py:180-214`. | Keep the enum, but bind every kind to a versioned schema. |
| `event_time` / `known_at` | Structurally present, semantically partial | `MarketEvent` defaults and validates clocks in `smc_trader/model.py:4483-4510`; identity includes both in `smc_trader/observation.py:2795-2809`. Different Sweep/Acceptance families assign different meanings to `event_time`. | Freeze clock meaning per kind. For a resolved crossing, use one crossing clock plus an explicit `resolved_at`, rather than changing the meaning by emitter family. |
| `direction` | Optional globally; valid for directional events but often null for neutral facts | `direction` is optional in `smc_trader/model.py:4480`; no kind-specific validator exists. Candidate/touch/penetration emitters generally record `side` and no `direction`. | Register nullability per kind. Neutral level facts should explicitly define direction as N/A; research code must not silently convert a touch into a trade direction. |
| `zone` | Optional globally; appropriate nulls are not distinguished from omissions | Zone inference/validation is generic at `smc_trader/model.py:4525-4540`. Point/path facts often have no zone, while some Sweep/Acceptance emitters omit the crossed region. | Define required/forbidden/nullable zone rules per kind. Preserve frozen crossed bounds for every level interaction. |
| `evidence` | Present but untyped | `evidence` falls back to `details` at `smc_trader/model.py:4516-4519`; `_append_semantic_atomic` accepts any mapping at `smc_trader/observation.py:2757-2794`. | Add per-kind required evidence fields, units, allowed enums, and semantic-version validation. |
| `source_event_ids` | **Fails semantic provenance** | It is only an alias for `source_ids` (`smc_trader/model.py:4575-4585`); validation checks only non-empty strings (`smc_trader/model.py:4520-4524`). Missing identities become opaque `source:<id>` tokens rather than errors in `smc_trader/signal_research.py:384-417`. | Split `source_event_ids`, `source_data_ids`, and `source_entity_ids`. Require every `source_event_id` to exist, precede or equal the child at `known_at`, use the same semantic version, and satisfy allowed parent-kind constraints. |
| atomic versus compound | Partial | MSS Core is not gated by Sweep, FVG, or Displacement, which is correct; however, duplicate resolution emitters and canonical compatibility aliases represent the same occurrence more than once. | Keep definitional ancestry separate from contextual evidence IDs, and designate exactly one canonical occurrence per semantic fact. |

## Capability Matrix

Status meanings: **covered** means there is an emitter with usable core fields; **partial** means the semantic exists but violates provenance, uniqueness, lifecycle, or preregistration requirements; **missing** means no authoritative canonical transition is emitted.

| Semantic | Status | Current implementation and field coverage | Material gap |
|---|---|---|---|
| Confirmed Swing | Partial | `SWING_CONFIRMED` is emitted with pivot `event_time`, later confirmation `known_at`, side, price, prominence, hierarchy and delay evidence (`smc_trader/observation.py:3130-3167`). | It links a compatibility Swing State event, not the completed left/pivot/right bar events. The first/equal swing may have null direction. More importantly, the executable threshold defaults to `minimum_prominence_atr=0.0` (`smc_trader/structure.py:58,131-132,1168-1175`) while the parameter registry says `prominence_ATR=null` and “continuous, not thresholded” (`semantics/parameters.yaml:22-25`). Freeze the actual v1 behavior explicitly and link raw bar evidence. |
| Structural Leg | Covered with one guard missing | It is emitted from opposite confirmed swings with direction, amplitude, duration, retracement and efficiency (`smc_trader/observation.py:3171-3218`). `event_time` is the ending pivot and `known_at` is the ending swing knowledge clock. | The emitter silently drops unavailable swing mappings. Require exactly two source events and validate that both are `SWING_CONFIRMED`; fail closed rather than emit an incomplete leg. |
| Candidate Liquidity Level | Partial | Zone/pool candidates preserve level identity, class, bounds and source descriptors (`smc_trader/observation.py:3455-3481,3605-3628`). | Inventory candidates may be created only when first crossed (`smc_trader/observation.py:5467-5493`), so the authoritative reducer can learn a candidate too late. Inventory `item.source_ids` are commonly entity/data IDs, not events. Candidate admission must occur when first knowable, with canonical source ancestry. |
| Level Touch | Partial | Touch events source the candidate and preserve frozen zone in the zone/inventory paths (`smc_trader/observation.py:3485-3510,5494-5515`). | They do not source the completed bar that caused the touch. Direction is correctly neutral but is not registered as N/A. Add the bar event and occurrence-specific extreme/distance evidence. |
| Level Penetration | Partial | Inventory penetration stores extreme, penetration points and frozen bounds (`smc_trader/observation.py:5516-5535`). | Other paths are less complete, and penetration generally does not source the corresponding Touch or completed bar. Enforce Candidate + Touch + BAR ancestry and one crossing-generation ID. |
| Liquidity Sweep / Acceptance | **Fail** | Canonical terminal events are emitted by BOS post-break resolution (`smc_trader/observation.py:3956-3995`), Group 4 manipulation (`smc_trader/observation.py:4804-4838`), same-bar inventory resolution (`smc_trader/observation.py:5558-5576`), and projected-pool resolution (`smc_trader/observation.py:5627-5669`). | These are competing definitions under the same kind/version. The registry requires mutually exclusive terminal resolutions for one crossing generation (`semantics/registry.yaml:64-80`), but the current paths can produce duplicate Acceptance facts. Clock meaning also varies: crossing, formation, or resolution time. Consolidate on a canonical crossing object and exactly one terminal resolution; otherwise version/distinguish the families. |
| Displacement | Partial | `DISPLACEMENT_OBSERVED` preserves direction, lifecycle and continuous transition metrics (`smc_trader/observation.py:4397-4455`). | Source IDs are admitted candle IDs, not `BAR_COMPLETED` event IDs. The preregistered feature wording includes explicit body/ATR, true-range/ATR, volume anomaly and FVG creation (`semantics/registry.yaml:82-89`), while the emitted metric set is not schema-enforced to contain those exact fields. Link bars and either emit the exact frozen vector or revise the definition in a new semantic version. |
| Raw Boundary Break | Partial | It preserves direction, target boundary, break bar identity, break distance and close-beyond standard (`smc_trader/observation.py:3745-3784`). | It does not source the `BAR_COMPLETED` event for `break_bar_id`; `price` is the boundary and evidence does not freeze the actual break close/buffer value. Add completed-bar ancestry and exact comparison inputs. |
| Qualified BOS | Partial | It is cleanly derived only for continuation-scope raw breaks, and Displacement remains context rather than a requirement (`smc_trader/observation.py:3788-3813`). | Its only canonical source is Raw Break. It does not source the `STRUCTURE_DIRECTION_CONFIRMED` event/generation that proves alignment. Add the exact prior structure fact to definitional ancestry. |
| Protected Swing | Partial | Assignment links Qualified BOS and the selected confirmed swing and records the origin leg identity (`smc_trader/observation.py:3814-3874`). | The canonical Structural Leg event is not a source. No explicit protected-swing invalidation event exists. Reducer validation matches Acceptance by price and `bos_id`, not by exact protected swing/generation (`smc_trader/market_state.py:1769-1787`), allowing same-price ambiguity. Link BOS + leg + swing, and invalidate by exact identity through a source-linked Acceptance event. |
| MSS Core | Partial, atomic boundary is correct | It is emitted for an opposed break and does not require Sweep, Displacement, or FVG (`smc_trader/observation.py:3875-3898`). Those remain context flags, so the core/composite boundary is conceptually correct. | The only canonical source is Raw Break. The event does not source the prior internal-direction/structure-generation fact that proves “first opposed break”; context is represented by booleans rather than exact event IDs. Add the prior structure event as definitional ancestry and keep optional context IDs in a separate field. |
| FVG lifecycle | **Fail** | Creation and partial/midpoint/full/invalidated/expired kinds are mapped at `smc_trader/observation.py:4094-4188`; direction, bounds, lifecycle and fill evidence are retained. | Creation places three candle IDs directly in `source_event_ids` (`smc_trader/observation.py:4106-4119`). Partial or full-fill also emits `FVG_TOUCHED` as “compatibility transport,” but `_append_semantic_atomic` marks it canonical (`smc_trader/observation.py:4190-4210`), double-representing one transition. `FVG_EXPIRED` is recognized as terminal (`smc_trader/group3.py:447-453`) but no tracker transition assigns it; boundary expiry-like behavior becomes Invalidated (`smc_trader/group3.py:521-545`). The registry simultaneously requests expiry observations and states the reducer has no separate expiry semantic (`semantics/registry.yaml:127-135`). Link completed bars, make the alias noncanonical, and either implement/freeze expiry or remove/deprecate it consistently. |
| Active Dealing Range lifecycle | **Fail** | Created, Activated, Invalidated and Replaced are emitted with frozen bounds and source descriptors (`smc_trader/observation.py:4617-4705`). | `DEALING_RANGE_EXTENDED` is declared but has no emitter. The underlying range terminates on the first close beyond bounds (`smc_trader/group4.py:740-768,806-818`), while the preregistration requires registered external Acceptance before replacement (`semantics/registry.yaml:145-152`). Source anchor swing/zone IDs remain evidence/entity IDs, not a canonical anchor graph. Align the executable rule with the frozen definition or issue a semantic-version change. |
| Delivery Phase | **Missing as an event** | Phase is derived deterministically in reducer/snapshot projection (`smc_trader/market_state.py:617-633,1936-1961`). | `DELIVERY_PHASE_CHANGED` is declared (`smc_trader/model.py:210`) but never emitted. There are two phase calculators, and the current rules do not include explicit range extension or volatility compression required by the registry (`semantics/registry.yaml:163-170`). Establish one authoritative function and emit immutable old/new phase changes with exact source IDs, or explicitly remove the event kind and preregister phase as snapshot-only. |
| Order Block / Origin Zone | **Fail** | Created, Touched, Mitigated and Invalidated events preserve direction and zone geometry (`smc_trader/observation.py:4304-4395`). | Creation sources only the compatibility Order Block State event; Displacement, BOS, and anchor candles are evidence identities, not canonical sources. A `MITIGATED` transition emits both `ORIGIN_ZONE_TOUCHED` and `ORIGIN_ZONE_MITIGATED` with the same clock/source/evidence, while the tracker defines mitigation as simple zone intersection (`smc_trader/group3.py:923-981`). Choose one authoritative atomic fact or separate Touch from a later, strictly defined mitigation outcome; link exact Displacement, BOS, and completed-bar events. |

## Atomic and Composite Boundary Findings

The intended separation is partially successful:

- Raw Break is emitted before Qualified BOS.
- Qualified BOS is based on continuation scope; Displacement is only a context indicator.
- MSS Core is based on opposed scope; Sweep, Displacement, and FVG are not mandatory gates.
- Candidate, Touch, and Penetration exist separately.

The following still mix or duplicate facts:

1. Sweep and Acceptance share one semantic kind across four independently resolving families without a common crossing-generation identity.
2. FVG partial/full transitions and `FVG_TOUCHED` are both marked canonical for the same occurrence.
3. Origin Zone “touched” and “mitigated” are simultaneous synonyms under the current tracker.
4. Context is often reduced to a boolean or opaque entity ID, preventing exact ablation by immutable event identity.
5. Compatibility lifecycle state events are used as definitional canonical parents. This preserves some audit information, but it does not make the atomic event graph authoritative because those state transports themselves can source non-event identities.

## Bounded Replay Contract Evidence

The 600-row January 2024 diagnostic produced 13,483 audit events, of which 2,284 were marked canonical and non-projection. The following counts are contract diagnostics only:

| Kind | Count | Null direction | Null zone | Dangling source references / total source references |
|---|---:|---:|---:|---:|
| Swing Confirmed | 254 | 21 | 254 | 0 / 254 |
| Structural Leg Created | 189 | 0 | 189 | 0 / 378 |
| Liquidity Level Created | 469 | 469 | 0 | 224 / 469 |
| Level Touched | 326 | 326 | 0 | 0 / 326 |
| Level Penetrated | 433 | 433 | 0 | 0 / 433 |
| Sweep Confirmed | 132 | 0 | 24 | 0 / 181 |
| Acceptance Confirmed | 112 | 0 | 62 | 0 / 224 |
| Displacement Observed | 58 | 0 | 58 | 126 / 126 |
| FVG Created | 30 | 0 | 0 | 90 / 120 |
| Raw Boundary Break | 86 | 0 | 86 | 0 in its canonical parents, but break-bar identity is not linked |

The same diagnostic found 18 duplicate `(kind, level_id, known_at, direction)` groups, covering 36 Acceptance events. Each pair was emitted once by Group 4 registered outside acceptance and once by projected-pool later-close acceptance. This is direct evidence that current canonical uniqueness is not guaranteed.

Null values alone are not automatically defects: a neutral level does not need trade direction, and a point/path semantic may not need a zone. The defect is that this nullability is neither kind-specific nor preregistered, so omission cannot be distinguished from intentional N/A.

## Must-Fix Items Before Phase 2/3 Acceptance

### P0 — semantic correctness

1. **Close the source graph.** Introduce separate `source_event_ids`, `source_data_ids`, and `source_entity_ids`; map all consumed candles to their `BAR_COMPLETED` events; validate source existence, clock ordering, semantic version, and allowed parent kinds at append/replay time.
2. **Guarantee one occurrence, one canonical terminal resolution.** Create a stable crossing-generation identity and route each penetration to exactly one Sweep or Acceptance. Suppress/version the competing Group 4, pool, and BOS resolution semantics rather than publishing all under one type.
3. **Add kind-specific schemas.** Require clocks, direction/nullability, zone/nullability, source kinds, evidence fields, units, and lifecycle values per semantic type. Generic `MarketEvent` validation is insufficient.
4. **Remove canonical aliases.** Make FVG compatibility Touch noncanonical or remove it; separate Origin Zone Touch from a genuinely later mitigation rule, or retain only one canonical transition.
5. **Make the preregistration executable.** Resolve the Swing prominence mismatch, FVG expiry contradiction, Dealing Range single-close versus Acceptance rule, and Delivery Phase rule/transition gap. Any definition change after freeze requires a semantic-version increment.

### P1 — derived-structure provenance

6. Qualified BOS must source Raw Break plus the exact same-timeframe Structure Direction fact used to qualify it.
7. Protected Swing assignment must source Qualified BOS, the canonical origin Structural Leg, and the protected Confirmed Swing. Its invalidation must reference the exact protected swing/generation, not only price.
8. MSS Core must source the prior internal Structure Direction/generation that makes the break opposed and first; optional Sweep/Displacement/FVG context must remain non-definitional and refer to exact event IDs.
9. Dealing Range creation/replacement must source canonical anchor facts and an exact qualifying Acceptance when the frozen definition requires it. Implement or remove the unused Extended transition.
10. Delivery Phase must have one authoritative deterministic derivation. If it remains a canonical event, emit only on phase changes with old/new values and complete sources.

### P2 — evidence parity

11. Freeze the exact Displacement vector and require its fields; do not let registry wording and executable metrics drift.
12. Preserve actual break close, buffer, penetration extreme, maximum penetration, recovery duration, and frozen bounds where applicable.
13. Register neutral-event direction and point/path-event zone nullability explicitly; remove research-time implicit direction assumptions from Eye facts.

## Required Tests

Add acceptance tests that fail under the current implementation:

1. Every canonical `source_event_id` resolves to an immutable stored event; raw/entity IDs are rejected from that namespace.
2. Source graphs are acyclic, same-version, causal by `known_at`, and valid by parent-kind schema.
3. Every canonical kind passes its required/forbidden direction, zone, evidence, and source-cardinality contract.
4. One penetration generation reaches exactly one terminal Sweep or Acceptance, including overlapping Group 4 and projected-pool paths.
5. Per-kind clock tests freeze the meanings of occurrence, crossing, confirmation, and resolution time.
6. FVG creation sources exactly three completed bars; each lifecycle occurrence emits once; expiry behavior is either demonstrated or rejected as unsupported.
7. Qualified BOS, Protected Swing, and MSS Core ancestry reconstructs the precise structure generation used by the decision.
8. Protected Swing remains intact on wick/ordinary close and invalidates only on its registered, identity-linked Acceptance criterion.
9. Active Dealing Range cannot invalidate/replace before the registered external Acceptance; Extended behavior is covered or prohibited.
10. Delivery Phase transition replay produces identical phase changes and includes every registered input.
11. Origin Zone touch and mitigation are demonstrably distinct, or only one canonical transition is present.
12. Full replay and incremental replay produce the same canonical IDs, source graph, terminal-resolution cardinality, and reduced state.

## Minimal Repair Order

1. Add source namespaces, BAR-to-event mappings, and a per-kind validator without rewriting the existing detectors.
2. Consolidate duplicate canonical aliases and the four crossing-resolution emitters behind one occurrence identity.
3. Repair derived ancestry for BOS, Protected Swing, MSS Core, Dealing Range, FVG, Displacement, and Origin Zone.
4. Reconcile the four frozen-definition conflicts: Swing prominence, FVG expiry, range Acceptance, and Delivery Phase. Increment `semantic_version` wherever behavior changes.
5. Run unit, lineage, leakage, deterministic replay, and duplicate-occurrence tests.
6. Only after semantic identities are frozen should the January 2024 Phase 5 diagnostic be rerun. Results produced under the current event identities are not portable to a corrected semantic version.

## Completion Boundary

This audit confirms useful reusable implementation, not completion. Phase 2/3 cannot be declared complete until the P0 contract defects are repaired and tested. Phase 4 reducer replay passing does not waive defects in the semantic facts supplied to the reducer.
