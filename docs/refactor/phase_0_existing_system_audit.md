# Phase 0 Existing System Audit

> **Historical / superseded audit snapshot (2026-08-20).** This document
> records the repository before the Phase 2–4 refactor and is intentionally
> preserved as migration evidence. Some findings and status labels below have
> since been superseded; they must not be used as the current acceptance
> verdict. Current authority comes from the v1.2 runtime contract and the
> [Current SMC Refactor Implementation Status](current_implementation_status.md);
> the [Phase 2–5 completion report](phase_2_5_completion_report.md) is the
> historical v1.1 January-run record. Any run counts or fingerprints below
> apply only to the older development identity under which they were produced.

This document records the repository state observed before further architecture
migration. It is an implementation audit, not a declaration that later phases
are complete or that any model is authorized for trading.

Status terms used below:

- **Present**: an implementation and a usable contract exist in the repository.
- **Partial**: useful implementation exists, but it does not yet satisfy the
  target boundary or authority model end to end.
- **Missing**: no explicit implementation of the requested contract was found.
- **Diagnostic only**: research code or results exist but cannot tune, admit, or
  authorize a production model.

Four conclusions must remain explicit throughout the refactor:

1. The current Brain is **not Bayesian**. It uses typed scores and fitted
   reliability/calibration maps, not priors, likelihoods, posterior odds, or a
   normalized posterior over mutually exclusive path hypotheses.
2. The current runtime is **not an atomic-event-authoritative reducer end to
   end**. It publishes immutable atomic/audit events and can replay public
   projection events, but the existing observer reducers and rich observation
   objects still create the authoritative runtime facts consumed by the Scene
   Graph and Brain.
3. The January 2024 signal study is **not out of sample**. Its registered split
   role is `brain_calibration_trial`, its manifest says `out_of_sample_period:
   not_opened`, and its output is diagnostic only.
4. Execution has **no explicit order finite-state machine (FSM)**. The replay
   portfolio has implicit pending/open/exit fields, but not named, audited order
   states and transitions.

## Current Architecture

### Runtime data and Eye path

The current causal runtime is broadly:

```text
completed 1m data
  -> CausalMarketReader and timeframe aggregators
  -> CausalObserver and existing semantic reducers
  -> MarketObservation + bounded EventMemory
  -> immutable audit events / ImmutableEventStore
  -> MarketSnapshotPublisher
       - independent TimeframeState objects
       - independent RelationState objects
       - one cross-timeframe SessionState
       - events_this_update and composite labels
  -> TemporalMarketSceneGraph / focused observation
```

The existing reducers already recognize and maintain substantial structure,
liquidity, displacement, imbalance, range, zone, and entry-path state. The
semantic v1 work reuses those reducers and projects canonical immutable events
with `event_time`, `known_at`, `semantic_version`, source IDs, evidence, and
deterministic identity. `ImmutableEventStore` is append-only, rejects mixed
semantic versions, orders by the causal availability clock, fingerprints the
stream, and supports as-of replay with a supplied pure reducer.

`market_state.py` contains immutable `TimeframeState`, `RelationState`,
`SessionState`, and `MarketSnapshot` contracts. It also contains a pure
`reduce_hierarchical_state` projection reducer. This is valuable audit and
reconstruction machinery, but it does not yet make atomic semantic events the
sole source from which every live Eye fact is reduced. Today, canonical events
are predominantly projections of existing reducer state and lifecycle output.

### Brain and decision path

The current decision path is broadly:

```text
MarketObservation / Scene Graph context
  -> PlaybookBrain
       - root- and playbook-specific HypothesisBelief candidates
       - lifecycle, evidence, contradiction, invalidation, target and deadline
       - typed quality dimensions and calibration maps
  -> UtilityDecisionLayer
       - ENTER / WAIT / ABSTAIN
       - HOLD / PROTECT / EXIT for an open position
  -> StructuralRiskEngine
       - causal/risk vetoes
       - frozen thesis and protected stop
  -> SequentialPortfolio
       - next-bar simulated entry and position management
```

`PlaybookBrain` retains multiple candidates keyed by concrete mechanism/root
identity instead of collapsing the market to one bullish/bearish scalar. Each
candidate can carry supporting and contradicting evidence, a structural
invalidation, targets, deadlines, terminal reasons, source identities, and
episode/context ownership. This is a strong reusable basis for falsifiable
hypotheses.

It is not yet the requested `HypothesisManager` over explicit competing market
paths such as continuation, deeper retracement, reversal, balance, and failed
breakout. Candidate values are also not normalized relative probabilities.
`HypothesisBelief.probability` currently mirrors calibrated thesis strength,
while entry utility uses calibrated `delivery_quality`; neither is a Bayesian
posterior. `_typed_thesis_strength` retains an unchanged evidence revision and
replaces the descriptive raw score on a new revision. There is no immutable
old-value/evidence-delta/new-value belief-update ledger.

The Brain contains deterministic target selection (`_draw_rank` and
`_select_target`) and freezes the selected draw and liquidity route. It does
not publish a ranked set of DOL hypotheses with calibrated probabilities,
expiry, and path obstacles as a first-class contract.

The utility layer can compare trade, wait, abstain, protect, hold, and exit
utilities. However, there is no explicit `SignalPolicy` output or `TradeIntent`
contract. `TradePlan`, `Decision`, `RiskAssessment`, and `FrozenThesis` jointly
approximate parts of a trade intent. The active model configuration has no
calibration artifact and has all release-readiness permissions, including live
execution, set to false.

### Execution and position path

`TopOfBookExecutionProvider` converts a causal BBO snapshot into descriptive
spread, displayed depth, cost, and fillability inputs. `MBOOrderBook` and
`MinuteExecutionRealityStore` provide reusable market-by-order reconstruction
and minute execution-reality infrastructure.

`SequentialPortfolio` owns replay exposure through `_pending_entry`, `_open`,
and `_pending_exit_reason`. It freezes stop and target geometry, handles
contract changes and deadlines, applies tighten-only protection, tracks MFE and
MAE, resolves same-bar stop/target ambiguity conservatively, and records closed
trades.

This is an implicit replay lifecycle, not an explicit execution FSM. An
approved entry becomes one next-bar, full-fill limit-style attempt at
`planned_entry`; the pending attempt is cleared after that bar whether or not
it fills. There are no order IDs, acknowledgements, working-order persistence,
cancel/replace, rejected orders, partial fills, queue position, broker
reconciliation, or an explicit market-versus-limit policy. The executor also
receives the full `EngineSnapshot`, rather than a narrow approved trade intent.

### Research path

The January 2024 manifest preregisters semantic version, dataset identity,
contract handling, session definition, diagnostic window, event and control
definitions, outcomes, minimum sample size, and interpretation rules. Its
runner produces atomic event studies, a nested chain, non-nested comparisons,
matched controls, and structural path outcomes.

That run replayed 30,477 real 1m rows, retained 42,150 canonical research events
and 671,361 audit events, and produced a deterministic fingerprint. E4 through
E6 are below the registered minimum sample size. The study excludes execution
costs, fills, stops, and P&L and cannot fit a Brain artifact, change semantic
thresholds, claim OOS performance, or authorize trading.

## Reusable Components

The migration should extend these components rather than build parallel
detectors or a second trading stack:

- `CausalMarketReader`, completed-bar aggregation, contract-boundary handling,
  and causal clock checks.
- Existing structure, liquidity, displacement, imbalance, range, zone, and
  entry-path reducers in the Eye.
- The semantic registry and frozen parameter files as the machine-readable
  definition/version boundary.
- The immutable semantic event envelope, source provenance, event identity,
  append-only store, fingerprints, and as-of replay.
- Canonical atomic projections for Swing, Structural Leg, candidate liquidity
  level interactions, Sweep, Acceptance, Displacement features, raw break, and
  FVG lifecycle observations.
- Derived projections for qualified BOS, protected swing, MSS core, active
  dealing range, continuous range location, and delivery phase.
- Independent `TimeframeState`, `RelationState`, `SessionState`, and
  `MarketSnapshot` contracts and the parent-confirmation relation rules.
- `TemporalMarketSceneGraph` identity and provenance mechanisms.
- Root-specific playbook candidate lifecycle, evidence, invalidation, deadline,
  terminal-state, and rearm logic.
- Deterministic draw selection and frozen liquidity-route provenance.
- Typed Brain calibration recorder/fitter and the utility decision/risk veto
  boundaries, while preserving their current non-live status.
- BBO and MBO reconstruction inputs, conservative bar-resolution helpers,
  sequential replay checkpoints, and trade outcome records.
- Frozen experiment manifests, structural-outcome runner, matched controls, and
  diagnostic report generation.

## Conflicting Components

These are migration conflicts, not reasons to discard the reusable code:

1. **Two authority shapes coexist.** Immutable canonical events and replayable
   projection events exist, while `MarketObservation` and existing mutable
   reducer state remain the operational authority. A future event-authoritative
   path must converge these without adding duplicate detectors.
2. **Brain candidates are mechanism-centric, not path-centric.** Multiple
   candidates exist, but they do not explicitly form the requested exhaustive
   or mutually exclusive market-path competition sets.
3. **Probability semantics are overloaded.** Thesis strength, delivery quality,
   raw scores, and calibrated scores are separate dimensions, but the field
   named `probability` is thesis strength and is not normalized or Bayesian.
4. **Belief updates are state replacement, not an auditable evidence ledger.**
   Evidence is explainable, but exact old score, evidence contribution, decay,
   new score, and normalization effects are not persisted as immutable updates.
5. **Eye contains execution-adjacent facts.** `MarketObservation` carries
   spread, slippage, fillability, cost, and deadline inputs. These may remain
   descriptive facts, but must not become Eye market-semantic authority.
6. **Brain contains execution-method choices.** Planned entry selection and
   fillability evidence partially decide how to enter, although market/limit,
   passive/aggressive, waiting, cancellation, and replacement should be owned
   by execution research and the executor.
7. **Execution consumes too broad a contract.** `SequentialPortfolio` receives
   an `EngineSnapshot` instead of a narrow, risk-approved `TradeIntent` or
   execution command.
8. **Decision and execution lifecycle ownership overlap.** The decision layer
   selects protect/hold/exit actions, while the portfolio owns the mechanical
   position transitions. The target design needs an explicit boundary between
   policy intent and order/position mechanics.
9. **One-bar fill behavior hides order outcomes.** Unfilled, expired,
   cancelled, rejected, and partially filled states are not separately
   represented or audited.

## Missing Components

The following target contracts were not found as explicit, production-ready
components:

- A path-level `HypothesisManager` for continuation, deeper retracement,
  reversal, balance, and failed breakout hypotheses.
- A replayable `BeliefUpdater` that records each evidence contribution,
  contradiction, invalidation, expiry, decay, prior value, new value, and
  competition-set normalization.
- Bayesian or log-odds posterior updating. This should not be claimed merely
  because calibration maps return numbers in `[0, 1]`.
- A probabilistic DOL hypothesis/ranking contract with calibration and explicit
  OOS validation against baseline ranking rules.
- A first-class `Signal`, `SignalPolicy`, and signal-expiry/cancellation
  contract.
- A narrow, risk-approved `TradeIntent` containing side, signal identity,
  permitted entry methods, invalidation, structural targets, risk budget,
  maximum wait, and cancellation conditions.
- An explicit immutable order-event stream and order FSM covering `IDLE`,
  `ARMED`, `ORDER_WORKING`, `CANCELLED`, `EXPIRED`, `FILLED`,
  `POSITION_OPEN`, `POSITION_MANAGED`, and `EXITED`.
- Partial-fill aggregation, cancel/replace, rejection, market/limit selection,
  passive/aggressive limit behavior, queue/fill modeling, and broker-state
  reconciliation.
- Execution research comparing FVG50, OB50, reclaim, breakout, market, waiting
  times, stops, targets, cancellation, adverse selection, and missed trades.
- MBO mechanism validation for Sweep, Acceptance, Displacement, MSS, and FVG
  retests. Reconstruction infrastructure exists; the registered validation is
  still a later phase.
- A frozen train/validation/OOS promotion sequence for Brain/DOL/Signal models.
- Shadow-live parity and stability evidence. No live-trading authorization is
  present.

## Capability Matrix

| Target capability | Current evidence | Status | Remaining gap |
|---|---|---|---|
| Causal multi-timeframe normalization | Completed-bar reader and aggregators with contract and clock checks | Present | Continue data QA; do not duplicate ingestion |
| Semantic provenance | `event_time`, `known_at`, semantic version, source IDs, evidence, deterministic identity | Present | Enforce the same envelope on every later Brain and execution event |
| Immutable audit log and deterministic replay | Append-only store, causal ordering, fingerprint and pure-reducer API | Present | Persist/operate it as an authority boundary where required |
| Core atomic semantic vocabulary | Canonical projections reuse existing Swing, leg, liquidity interaction, displacement, raw-break and FVG reducers | Present, subject to integration acceptance | Complete parity/leakage tests; avoid a second detector |
| Derived structure | Qualified BOS, protected swing, MSS core, range/location and delivery-phase projections exist | Present, subject to integration acceptance | Verify same-timeframe confirmation and context-only gates end to end |
| Hierarchical timeframe state | Independent immutable `TimeframeState` objects | Present | Converge runtime authority onto the event/reducer contract |
| Parent/child relation object | Independent `RelationState` map with deterministic roles/warnings | Present | Add broader parity and transition-stability coverage |
| Session independent of timeframe | One independent `SessionState` in `MarketSnapshot` | Present | Preserve as context/research feature, not a hard gate |
| Atomic-event-authoritative Eye reducer | Projection reducer exists, but runtime facts originate in existing observer reducers/rich observations | **Partial; not authoritative end to end** | Make ordered events the accepted state-change input without rebuilding detectors |
| Multiple falsifiable Brain candidates | Root/playbook candidates have evidence, contradiction, invalidation, expiry and terminal state | Partial | Add explicit competing market-path hypotheses and competition sets |
| Bayesian belief update | Reliability maps and score replacement only | **Missing; current Brain is not Bayesian** | First add an auditable calibrated updater; add Bayesian/log-odds only after preregistration and validation |
| Relative hypothesis credibility | Independent scores exist | Missing | Define exclusivity, residual/other mass, normalization and update ledger |
| DOL ranking | Deterministic structural tier/strength/distance selection and frozen draw | Partial | Publish candidate probabilities, obstacles, expiry, calibration and OOS comparison |
| Signal probability and policy | Calibrated delivery quality and utility actions exist | Partial / disabled for release | Define SignalPolicy contract and fit only on permitted train/validation data |
| Trade intent | TradePlan, Decision, risk assessment and frozen thesis carry parts | Missing as an explicit boundary | Add one narrow risk-approved intent consumed by execution |
| Nested and non-nested event studies | January diagnostic reports both | Diagnostic only | Resolve sparse cells, controls and later registered windows; no promotion claim |
| Experiment preregistration | January manifest freezes definitions, controls, outcome and minimum sample | Present for the diagnostic | Add frozen train/validation/OOS manifests before model promotion |
| January 2024 OOS evidence | Split is `brain_calibration_trial`; OOS is unopened | **Missing; 2024-01 is not OOS** | Reserve and run a later untouched OOS period after freezing |
| MBO reconstruction | Order-book reconstruction and minute execution-reality store exist | Partial | Run registered mechanism and stability validation |
| Explicit execution FSM | Only implicit pending/open/exit fields | **Missing; Execution has no explicit FSM** | Add named states, commands, events, guards and replay tests |
| Entry/fill/cancel lifecycle | One next-bar full-fill limit-style attempt | Partial | Distinguish working, filled, partial, cancelled, expired and rejected outcomes |
| Stop/target/position mechanics | Frozen stop/target, conservative same-bar handling, protection, deadlines, MFE/MAE | Partial | Put mechanics behind FSM and add multi-target/partial-position policy only if research requires it |
| Shadow-live parity | No authorized shadow-live evidence | Missing | Compare replay and real-time fingerprints/state transitions before any live action |

## Minimal Migration

The smallest safe migration keeps existing detectors and introduces narrow
contracts around them in this order:

1. **Freeze and test the current baseline.** Preserve semantic registry
   identity, event fingerprints, existing reducer outputs, and the January
   diagnostic as non-authoritative evidence. Add no threshold changes from that
   diagnostic.
2. **Converge Eye authority incrementally.** Route existing reducer lifecycle
   output through canonical atomic/state-change events, then reduce those
   ordered events into `TimeframeState`, `RelationState`, `SessionState`, and
   `MarketSnapshot`. Use parity tests against current output before removing any
   compatibility path. Do not implement parallel Swing, BOS, FVG, or range
   detectors.
3. **Wrap existing Brain candidates with path contracts.** Add stable IDs and
   path type, target, support, contradiction, invalidation, expiry, and terminal
   reason around the current root-specific candidates. Define which hypotheses
   are mutually exclusive before applying any normalization.
4. **Add an immutable belief-update ledger.** Record the prior score,
   source-event IDs, signed evidence contribution, decay, invalidation/expiry,
   unnormalized new score, and normalized result. Start with a preregistered,
   calibratable scoring/log-odds rule; do not label it Bayesian unless priors,
   likelihood/update assumptions, and posterior validation are actually
   implemented.
5. **Extract DOL, SignalPolicy, and TradeIntent.** Reuse deterministic draw
   ranking as a baseline. Publish all eligible DOL candidates first, then let
   the Brain rank them. Convert an approved signal into one narrow trade intent
   with risk and cancellation constraints; keep entry tactics out of the Eye.
6. **Wrap current replay execution in an explicit FSM.** Initially preserve the
   existing one-next-bar behavior, but emit distinct command and transition
   events for armed, working, filled, expired, cancelled, position-open,
   managed, and exited states. Only then add partial fills, cancel/replace,
   market entry, or queue models when execution research requires them.
7. **Narrow subsystem dependencies.** Execution should consume an approved
   intent plus execution-market data and account state, not the entire engine
   snapshot. Eye execution-reality fields remain descriptive inputs; Brain
   policy decides whether they matter, and the executor owns order mechanics.
8. **Promote through registered evidence, not the January diagnostic.** Run
   signal research on permitted training and validation periods, freeze model
   artifacts and thresholds, then open a previously untouched OOS period.
   MBO mechanism validation, execution research, and shadow-live replay parity
   remain separate later gates.

This sequence advances the requested architecture while minimizing new
infrastructure and preserving the repository's mature semantic reducers,
identity model, replay machinery, and risk controls.
