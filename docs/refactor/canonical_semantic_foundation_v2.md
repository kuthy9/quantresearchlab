# Canonical Semantic Foundation v2

Status date: 2026-08-23
Foundation identity: `smc_semantic_foundation_v2.0`
Parent atomic identity: `smc_semantics_v1.2`
Canonical-JSON registry identity SHA-256: `ac04636919931d774309a0c306764fdf8eb53aee41df0f31d4d94e5b9125732b`

This document is the implementation authority for the canonical semantic
foundation layered over the immutable v1.2 atomic event stream. It completes
object identity, geometry, generation, lifecycle, relation, transition,
reinteraction, ancestry, and factual outcome contracts. It does not add a
trading signal, fit a probability model, or grant action authority.

The machine-readable authority is
[`semantics/foundation_v2_0.yaml`](../../semantics/foundation_v2_0.yaml).
Changing any registered definition requires a new semantic version and a new
registry identity. Historical v1.2 events and frozen research artifacts are
not rewritten.

## Phase A audit result

The repository was inspected before implementation. Existing correct
detectors and reducers were retained.

| Object or contract | Before v2 | Resolution |
|---|---|---|
| Structural Leg path | Partial | Existing leg identity retained; full native-bar path, two efficiencies, close/wick MAE, tick amplitude, duration, and strictly-prior `ATR_at_leg_start` were added. |
| Swing hierarchy | Conflicting | Existing `micro/internal/structural/external` depth was role depth. A separate geometric containment tree now carries geometric depth and parent/child identity. |
| Same-level rearm | Partial | Equal-pool private generations existed, but general levels were terminal once. A registered level/interaction lifecycle now owns rearm. |
| Structure generation | Partial/conflicting | Tracker records and snapshot directions existed, but no canonical internal/external generation lifecycle existed. It is now an event-sourced projection over exact v1.2 facts. |
| Origin / OB | Partial/conflicting | A private frozen origin was folded into the qualified OB result. Base Origin Core and Qualified OB are now separate immutable objects. |
| Structural / Balance range | Conflicting | The legacy dealing-range slot represented Mature Balance Range while also feeding location labels. Structural Range and Balance Range are now independent records and locations. |
| Delivery generation | Missing | The deterministic phase classifier is reused and wrapped in a persistent generation lifecycle. |
| FVG expiry | Partial | The enum existed but no producer used it. v2 permits only registered structural/reset causes and keeps invalidation separate; no time TTL was introduced. |
| True first retest | Missing | The historical Phase 6 first-concrete-lifecycle proxy remains historical. A strictly future geometric First Retest/Reinteraction object now freezes the information set at first contact. |
| Multi-bar terminal ancestry | Partial | Crossing generation and competing terminal uniqueness already existed. v2 adds the complete real-bar formation ledger and terminal-specific roles. |
| Relation generation | Missing | `RelationState` was recomputed as a snapshot. A persistent parent/child generation now distinguishes start, update, reclassification, and termination. |
| Structural outcome | Fragmented | The signal runner had a private calculator. A shared factual engine now owns horizon, target/invalidation, same-bar ambiguity, censoring, MFE, and MAE. |

## Architecture and authority

```text
normalized BAR + canonical v1.2 atomic facts
                      |
                      v
       CanonicalFoundationAdapter
                      |
                      v
 immutable FoundationRecord history
                      |
                      v
 FoundationProjection + technical FOUNDATION_STATE_CHANGED transport
                      |
                      v
 MarketSnapshot.foundation / deterministic replay
```

The adapter is not a detector. It can consume only exact normalized-data or
semantic-atomic facts already admitted by the immutable event store. Technical
state projections, legacy transport, entity IDs, and candle IDs cannot be
silently promoted into definitional ancestry. Foundation transport carries
`canonical_semantic=false` and has no timeframe-reducer or trading authority.

Reset never clears history out of band. Active generations receive explicit
terminal or archived revisions before the next epoch. Checkpoint restore and
atomic replay reconstruct the same record identities and current views.

The production model config must explicitly set
`observer.canonical_foundation_enabled=true` and bind both
`observer.canonical_foundation_registry=semantics/foundation_v2_0.yaml` and the
canonical registry identity printed above. `ContinuousSMCEngine` rejects a
missing, false, or non-boolean gate, a missing binding, or an identity mismatch;
it strict-loads the registry before mapping the admitted setting to the
Observer's additive path. Foundation publication is therefore not a
test-helper-only behavior. This flag remains separate from the scan-only
`eye_authority_mode`, so normal execution-reality inputs retain their
established contract. The Engine freezes the admitted foundation version and
registry identity into its runtime/checkpoint state, and Shadow Live compares
both bindings on construction, restore, and every clock. The model-config byte
hash remains part of the runtime/shadow receipt identity. Engine checkpoint
schema v3 is the first schema that guarantees this production foundation state;
earlier checkpoint schemas fail closed instead of resuming without the
projection.

## Frozen vocabulary boundary

The registry contains 24 canonical primitives/state objects. Some are atomic
facts, some are immutable geometric objects, and some are persistent
generations. They are not forced into one artificial class.

HH/HL/LH/LL, Premium/Discount, IRL/ERL, and DOL Candidate remain attributes or
derived views. Breaker Block, Mitigation Block, OTE, Unicorn, Liquidity Void,
BPR, and other tutorial vocabulary remain experimental by default. Promotion
requires a written definition, leakage/replay validation, construct study,
matched controls, incremental-information evidence, validation, and a
versioned promotion.

## Geometry

### Swing Geometry / Nesting

A Swing's geometric envelope is its real completed definitional pivot-through-
confirmation bar window. A parent must have a strictly larger time window and
contain the child in both time and price. If several parents qualify, the
deterministic order is minimum enclosing duration, minimum price span, then
parent Swing ID. A late enclosing parent appends a superseding assignment; it
never rewrites an earlier snapshot.

`geometric_depth` is independent from append-only `role_depth`. BOS,
protection, later usefulness, and future outcomes cannot change geometric
parentage.

### Structural Leg

The existing leg ID remains stable. A foundation-v2 leg additionally freezes:

- `amplitude_ticks`, `amplitude_points`, and `amplitude_atr`;
- `close_efficiency` and `extreme_path_efficiency`;
- close MAE and wick MAE;
- inclusive native real-bar duration and wall-clock seconds;
- the complete ordered native path BAR ancestry;
- `ATR_at_leg_start`, calculated from exactly 14 strictly prior real completed
  native bars, plus those 14 source IDs.

Cold-prefix compatibility objects without both a frozen ATR value and its full
14-bar ancestry remain v1.2 records and cannot masquerade as v2.

### Liquidity Cluster

Source level identities are preserved. Same-side point levels form a spatial
cluster only when complete-link distance is at most one tick. Membership
change terminals the old cluster and starts a new one with explicit
supersession. Clustering is descriptive geometry; it never asserts that the
cluster is stronger or more likely to be a target.

### Structural Range and Balance Range

Structural Range binds two opposite structural Swings to one confirmed
external Structure Generation. It terminals when its owner terminates or is
replaced. Balance Range reuses the existing Mature Balance Range detector and
keeps its auction/compression lifecycle. Both may be active simultaneously.

Published location contains separate `x_structural_range` and
`x_balance_range`. Canonical Premium/Discount context belongs to Structural
Range; Balance location remains a separate local auction coordinate. The old
v1.2 dealing-range field remains compatibility state, not structural authority.

## Liquidity lifecycle and interaction generations

A Candidate Liquidity Level preserves source identity independently from a
spatial cluster. Its states are `ACTIVE`, `DISARMED`, `REARMABLE`, `REARMED`,
`RETIRED`, and `ARCHIVED`.

The registered v2 rearm rule is deliberately narrow:

- Sweep disarms an eligible point level; Acceptance permanently retires it.
- A strictly later real interaction-timeframe bar must close at least one tick
  beyond the frozen tradable zone toward the non-target side.
- That observed departure first creates an immutable `REARMABLE` fact and then
  a new `REARMED` interaction generation referencing the prior generation.
- Same-clock rearm is forbidden. The old terminal generation is immutable.
- Formed/equal-liquidity pools do not rearm the same source identity; new
  confirmed members must create a new pool source generation.

Continuous legacy zone bounds are projected inward to tradable tick bounds
using exact rational ceil/floor. An exact formed-pool arithmetic midpoint may
be nontradable; the lifecycle anchor is then the near-side tradable boundary,
with that rule recorded explicitly. The source event retains the original
midpoint.

Retirement requires an exact registered cause: reference-period rollover,
source supersession, owning structure/range termination, Acceptance, contract
reset, or semantic reset. Cache eviction or disappearance from a bounded
runtime view is never a semantic retirement fact. Retired/archived levels are
excluded from foundation DOL candidates, liquidity density, and target maps.

`MarketSnapshot` is the public DOL visibility gate. An eligible Sweep departure
creates Generation 2 and returns the rearmed level to the target map;
Acceptance, retired levels, and formed pools without their exact live source do
not flow back in. Candidate `age_bars` is the current real-bar ordinal minus the
active interaction's armed ordinal, so touch/penetration does not reset age and
streaming/replay derive the same value.

Protected rank is equally source-bound. A legacy tracker may label a Swing
protected in an initial structure snapshot before a canonical protected-Swing
assignment exists; that tracker-only role remains in the legacy inventory and
cannot promote the foundation candidate. Only an active Structure Generation
with the exact `protected_swing_id` and
`protected_swing_assignment_event_id` may publish the canonical protected role.
When that exact assignment arrives, stream and replay promote the already
frozen candidate rank one way to `external`; strength, geometry, source
identity, and creation metadata remain unchanged. The model invariant
protected-implies-external is not weakened. A formed-pool arithmetic midpoint
also never bypasses tick admission: the public DOL price is the lifecycle's
frozen near-side tradable anchor while the source fact retains its midpoint.

`LiquidityInteractionGeneration` owns `ARMED -> TOUCHED -> PENETRATED` and one
terminal: Sweep, Acceptance, Unresolved, Censored, or Expired. Sweep and
Acceptance are competing terminal outcomes of the same generation. The event
store and lifecycle reducer reject a second terminal or mutation of a terminal
generation.

One retained v1.2 source boundary remains explicit: legacy Swing/pool Touch
facts carry the source's confirmation BAR rather than a separate pivot BAR, so
technical transport cannot independently recompute their contact geometry from
the Touch fact alone. It validates the available level-specific facts, clocks,
source lineage, and terminal ownership and never invents the missing pivot
binding. Direct and range-boundary Touch paths are checked against their exact
available authoritative geometry. This is a provenance limitation, not license
to backfill a future or heuristic contact.

## Structure and transitions

`StructureGeneration` is separate from detector state and carries timeframe,
scope (`internal` or `external`), direction, origin, confirmation, protected
assignment, BOS/MSS evidence, and explicit termination.

An external generation persists until exact protected-break Acceptance,
rollover, reset, or supersession. A detector close break alone is not external
terminal authority. A generation cannot disappear because a snapshot
classifier fluctuated.

MSS is transition evidence, not a confirmed opposite regime. Against an
incumbent generation it starts one challenger transition and one FORMING
internal generation. Repeated same-owner/same-direction MSS facts append
evidence to that generation rather than creating churn. Confirmation requires
a strictly later independent registered structure confirmation bound to the
same challenger. Original-regime resumption produces `transition_failed`;
exact protected Acceptance followed by the opposite confirmation produces
`transition_confirmed`; reset produces a censored/terminated history.

`BOUNDARY_ATTACK` records a wick strictly beyond a structural boundary when
the completed close does not strictly confirm Raw Break. It binds the target
Swing and exact real BAR, remains distinct from BOS/MSS, and is deduplicated per
break generation and BAR.

## Cross-timeframe Relation and Delivery generations

A Relation Generation is identified by the confirmed parent/child structure
owners plus relation role. Equal signatures update the same generation;
reclassification or owner change terminals it and starts another. Parent
invalidation, child realignment, rollover, and reset are explicit terminal
reasons. Child internal MSS/retracement is evidence about the relationship; it
does not pretend that the confirmed child external owner changed direction.

A Delivery Phase Generation is identified by parent Structure Generation and
phase. The existing deterministic phase classifier is reused. Age, duration,
maximum favorable extension, and maximum adverse retracement update only on a
new real completed native-timeframe observation. Phase/owner change terminals
the old generation and records `next_phase`; reset and parent termination are
explicit. Multiple M1 clocks do not become independent HTF delivery samples.

## Zones and first reinteraction

Base Origin Core freezes the preregistered opposing candle/cluster full-range
geometry at the exact Displacement start knowledge clock. It does not require
future BOS or success. Qualified OB is a separate derived object that cites an
exact Base Origin Core, Active Displacement, and compatible Qualified BOS or
MSS according to the registered rule. The core remains immutable history when
qualification never occurs.

FVG keeps `age_bars` and `age_seconds` as continuous native-timeframe state.
Waiting bars are temporal observations, not definitional ancestry and do not
create append-only heartbeat records. v2 has no bar TTL. `INVALIDATED` means a
registered price close-through; `EXPIRED` is restricted to parent Structure
termination, relevant Structural Range replacement, contract rollover, or
semantic reset; data gaps are censored.

First Retest/Reinteraction is a strictly later native completed-bar geometric
entry after creation/departure. Boundary equality and a gap open inside count.
The immutable event freezes entry side, fill fraction, age, approach facts,
context, exact BAR, and source object. Eventual midpoint/full fill,
invalidation, continuation, or any other future outcome is forbidden from its
payload. The same primitive supports FVG, Base Origin/Qualified OB, range
boundary, and liquidity-zone studies without outcome-defined grouping.

## Multi-bar ancestry

Every Sweep/Acceptance formation ledger includes all real interaction-
timeframe bars from penetration through confirmation, their clocks, and their
actual roles. A Sweep records reentry; a direct Acceptance records outside
holds and confirmation without inventing reentry. The registered legacy pool
branch that reenters before later Acceptance records that distinct sequence.
The terminal record also cites level creation, penetration, and exact semantic
formation facts.

Formation ends at terminal `known_at`. A response window must start strictly
after it. A later Sweep-to-Displacement link is a temporal relationship, never
Sweep ancestry.

## Structural Outcome Engine

`StructuralOutcomeEngine` is the common factual research engine for Sweep,
MSS, FVG, Origin/OB, and other semantic sources. An immutable specification
freezes source, observation start, native timeframe, bar/second horizon,
target, invalidation, ATR reference, contract, and data window. The scan starts
strictly after the source is known and requires the contiguous native
completed-bar census; the first gap censors before any later bar can enter.

Terminals are `target_first`, `invalidation_first`,
`ambiguous_same_bar`, and `censored`. When one OHLCV bar touches both target
and invalidation without higher-frequency ordering evidence, the factual
result is `ambiguous_same_bar`. A separate execution projection may apply a
declared conservative rule; it cannot rewrite the factual result.

The existing Signal Research runner now delegates its common terminal,
MFE/MAE, horizon, contract/window, and same-bar behavior to this engine.
Historical compatibility metrics retain their old names but do not redefine
the new First Retest object.

## Explicit replay-seam limits

The self-review retains the following schema limits rather than converting
missing fields into inferred authority:

- Legacy Swing/pool Touch transport has the confirmation BAR limitation noted
  above.
- The first Boundary Attack may carry a `bos_generation_id` without an atomic
  pending-generation object to cite; subsequent ID/ordinal sequencing is closed
  and replay-stable. The unbacked first ID is not treated as proof of BOS, MSS,
  or a confirmed generation.
- An administrative Delivery terminal has no new terminal-price observation;
  it freezes the last causally bound price rather than inventing a reset price.
- Qualified OB has no legacy `origin_zone_id` field. It is instead checked by
  exact Base Origin Core, Displacement, direction, scope, and frozen zone
  geometry.
- The FVG lifecycle DTO does not duplicate direction/zone fields. Replay binds
  those facts through the exact creation event and its three BARs, and checks
  First Retest geometry against that creation.
- Balance Range replay validates lifecycle provenance and frozen geometry but
  does not independently recompute every Group-4 diagnostic metric.
- A generic Swing Geometry node does not duplicate Swing side/pivot fields.
  Structural Range therefore validates its low/high sides and pivot prices
  directly against the authoritative Swing events.

These boundaries fail closed and do not authorize heuristic backfill. Adding a
new duplicated field or stronger independent recomputation contract requires a
new registered foundation version; it cannot silently change v2.0.

## Replay, test, and empirical boundary

The bounded real-data verification uses the local causal OHLCV source
`data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`
(56,697,133 bytes; SHA-256
`84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b`).
Rows are source-ordered after the exact America/New_York filter
`2024-06-01 <= ts < 2024-07-01`; no time resampling or outcome selection is
performed. The currently checked production model config SHA-256 is
`dbb6662ee748531feeb556ee891745445e4df4f919e402fd1126d939333d84e5`.
The direct Observer construction/replay checks below do not load that production
file; the current Engine file-parity row does and binds this exact hash plus the
foundation version and registry identity.

| Verification | Frozen scope | Result |
|---|---|---|
| Registry/config admission | Canonical-JSON registry plus checked-in production model | Registry identity matches this document; exactly 24 objects; all four empirical/Brain/action authority flags are false. The production gate is exact boolean `true`; registry path and identity are mandatory and strict-loaded; Engine/Shadow/checkpoint state freezes the admitted foundation version and identity; Engine checkpoint schema is v3. |
| Focused contract tests | Lifecycle/geometry/zones/outcomes/projection/adapter plus source-provenance and Observer integration | 204 foundation-focused tests passed; Observer integration passed 17/17, including a real 2024-06 495-clock stream/replay case; 47 related event-provenance tests and a separate real 60-clock parity check passed. These are named runs, not an overlap-free sum. |
| Production Engine/Shadow integration | Foundation DOL rearm/Observer/Engine/atomic reducer/typed Brain/Shadow parity/file-pilot suites | 164/164 passed in 113.45 s after the final registry-freeze and DOL fixes; configuration JSON, binding order, Python compilation, and diff checks passed. This named run overlaps other focused suites and is not added to them as a unique-test census. |
| Independent adversarial/self-review | Represented high-priority source-kind, cross-object, lifecycle, and replay fields | P0 = 0, P1 = 0; nine targeted adversarial tests passed with warnings treated as errors. P2 consists only of the seven explicit replay-seam limits above; none grants future-data access or permits production semantic forgery. All 31 changed Python files parsed as valid ASTs, `git diff --check` passed, and tracked cache artifacts were zero. |
| Final post-freeze repository regression | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider` | Exit 0 in 270.01 s: 2,376 passed, 1 skipped, 6 deselected, 0 failures. Pytest reported 284,906 warnings, overwhelmingly pre-existing generic NumPy/pandas `Timedelta` deprecations. This run includes the final Engine registry binding, protected-rank, formed-pool anchor, and Shadow parity changes. |
| First-400 construction A/B | Same contiguous M1 bar starts, 2024-06-02 18:00 through 2024-06-03 00:39 ET; 4,676 Foundation records | Pre-optimization 138.160 s; derived-cache build 38.203 s, about 3.62x faster. |
| First-400 atomic replay A/B | Exact same 4,676-record payload | Pre-optimization 88.100 s; derived-cache build 3.745 s, about 23.52x faster; payload exact. |
| Final first-1,000 construction | Contiguous M1 bar starts, 2024-06-02 18:00 through 2024-06-03 10:39 ET; final `asof` 10:40 ET | Green in 320.970 s; 37,429 unique audit events; 13,664 unique Foundation records; 5,180 latest records; event and record IDs 100% unique; exceptions 0 and warning output 0. |
| First-1,000 Foundation record census | Same final stream | liquidity level 4,218; interaction generation 3,819; delivery generation 1,454; relation generation 951; Swing assignment 1,186; Swing node 433; Structural Leg 306; cluster 549; cluster supersession 259; Structure Generation 168; Structure Transition 58; Boundary Attack 150; FVG lifecycle 54; First Retest 33; Base Origin Core 17; Structural Range 9; Balance Range 0; Qualified OB 0. The two zero cells mean no qualifying object occurred in this bounded prefix, not missing wiring. |
| Final first-1,000 atomic replay | Same 37,429-event immutable stream | Green in 24.405 s; exact parity for `foundation`, `timeframe_states`, `relations`, `session`, and `foundation_range_locations`; parity mismatches 0. |
| Current production-Engine file parity | Exact 200-row `phase9_shadow_file_input_v2` prefix, 2024-06-02 18:00 through 21:19 ET; input SHA-256 `731d8069982a7bf35e6b1aabe80800e068c4e39c112a18846f5570fded7f2796`; current model/foundation bindings; `NullExecutionGateway` | `attempted = accepted = parity = 200`; gate, coverage, and live/cold parity all true; live/cold record fingerprint `7ae14df74e591ae32b162e7d741a9481fe8bd2fc6dd9670f1c671ab71d5c945c`; live/cold journal fingerprint `21a6b162b0a62268ff25eb6aaca02001eaf2a9cde884a912f4e64e1a35678e25`; zero approved intents, execution events, or external submissions; no warning or exception output. The portable [machine receipt](../evidence/phase9_foundation_v2_prefix_200_receipt.json) contains the exact bindings without claiming the temporary result bundle is published. |

The A/B figures are one local engineering measurement captured when the
non-authoritative derived indexes/digest reuse were introduced. The caches are
excluded from canonical checkpoint state and do not change record identity or
payload. The final first-1,000 run includes the fail-closed Foundation-record
hardening; the later production DOL/freeze bindings are covered separately by
the current Engine row. Neither measurement may be extrapolated into a
6,900-clock capacity claim.

The first-400/1,000 Observer checks read OHLCV only. They do not open new MBO
outcomes or rerun Phase 6. The pre-existing 6,900-clock Phase-9 JSONL (SHA-256
`fd9e48850d1657cf369e3e617e3e8b464790e9823f48c79b01f65bfc111a46e4`)
is a different historical artifact and was not the input to those checks. The
current Engine row uses its exact retained 200-row prefix with causal
bar/execution/account fields, but does not complete or rematerialize the
6,900-clock artifact. The Observer checks are construct/replay evidence and the
Engine check is bounded file-parity engineering evidence. Neither is an MBO
mechanism study, a real-time or multi-day Phase-9 pilot, empirical validation,
calibration, rolling OOF, or sealed OOS.

The final post-freeze repository-wide regression passed. The clean Git save
and its commit ID are reported by the handoff rather than embedding a
self-referential commit hash in this frozen definition document.

Foundation v2 is definition/replay infrastructure only. Its authority flags
are all false: no empirical validation, Brain probability, Trade Intent, or
execution authority follows from implementation. Phase 5/6 frozen results are
not reinterpreted. New construct, MBO, calibration, rolling OOF, and sealed OOS
research must bind this exact identity or a later explicitly versioned one.
