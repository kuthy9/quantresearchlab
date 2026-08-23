# SMC Continuous Trader

Version: **1.0.0**

This repository implements a causal, continuously updated multi-timeframe SMC
research model. It does not predict a complete future path and then choose a
strategy. Each completed M1 clock first advances the authoritative Eye; the
existing development trader then continues after that publication:

```text
normalized completed M1 clock (real or clock-only synthetic)
    ↓
immutable BAR_COMPLETED root + preregistered SEMANTIC_ATOMIC events
    ↓
definition-bound ImmutableEventStore
    ↓
per-timeframe reducers + independent RelationState / SessionState
    ↓
smc_semantic_foundation_v2.0 generation/lifecycle/geometry projection
    ↓
MarketSnapshot (atomic authority + replayable foundation records)
    ───── optional development-trader downstream ─────
Temporal Market Scene Graph + GlobalMarketContext + FocusState
    ↓
playbook-neutral open theses + root-specific DFP / LSR candidates
    ├── shadow Hypothesis Manager + Bayesian-shaped belief updates
    ├── obstacle-aware DOL ranking + fitted-artifact-only DOL probability
    ├── Signal Policy / Trade Intent
    └── retained typed action-candidate interface
    ↓
Decision / Risk development gates
    ↓
optional conservative next-bar simulation and position feedback
```

The Eye describes what has happened. The existing `PlaybookBrain` retains its
typed causal candidates and now also hosts the Phase 7 Hypothesis Manager,
Bayesian-shaped belief updater, obstacle-aware DOL ranking, fitted-artifact-only
DOL probability, Signal
Policy, and Trade Intent projection. These outputs are explicitly
`development_unvalidated` and `shadow_only`: no fitted/admitted path-likelihood,
DOL-probability, or outcome-calibration artifact exists, so production emits
zero Trade Intents and Decision/Risk receive no new action authority. The
legacy typed calibration artifact is also absent (`calibration_artifact=null`),
its fallback identity is `identity-unvalidated`, and
`live_execution_allowed=false`. DFP and LSR remain setup/entry-episode
generators rather than market-path probability owners; FAVR remains parked.
This is an integrated fail-closed runtime, not a fitted competition model: the
published path configuration uses equal priors, zero evidence increments, and
zero decay. The Brain adapter now maps exact production-shape market facts to
registered path invalidation and realized-winner events inside the current
authority scope, with same-clock terminal precedence and no authority from
local EntryEpisode failure. Authority/scope rollover retirement is not yet
preregistered, so those lifecycle facts remain shadow-only alongside the
missing probability artifacts.

Path protocol v1.2 treats `correlation_key` as a global dependency-cluster
identity rather than a family-local token. An admitted model cannot multiply
two evidence families from one declared cluster unless one separately frozen
joint or history-conditioned contribution represents that cluster. The current
source-only ledger remains neutral, so Sweep, Displacement, MSS, or FVG facts
are never silently treated as independent likelihood multipliers. If an inline
likelihood is opened without a registered dependency resolver, the Brain uses
one conservative unresolved cluster for the whole competition set and fails
closed instead of accumulating evidence.

Without an exact fitted/admitted DOL model artifact, the Brain publishes only
candidate rankings; its runtime DOL-probability map is empty. The separately
versioned probability layer retains the path marginalization formula and
explicit `no_target_before_common_horizon` outcome for future admitted models.

The downstream Scene Graph and Brain are sequenced after Eye publication but
still use the retained `MarketObservation`/Scene Graph contract rather than
`MarketSnapshot` as their sole evidence source. The Phase 7 path/DOL/Signal/
Intent projection is integrated into that existing Brain; it does not create a
parallel trader.

Scene-Graph `PRECEDES` edges remain diagnostic temporal associations. Explicit
causal/open-thesis and action-connectivity allowlists exclude them, so temporal
proximity alone cannot satisfy a playbook or FAVR action gate.

This is research software. A connected software path is not evidence of market
edge. Phase 6 produced development-association evidence only; Brain
calibration, Phase 8 empirical execution research, the multi-day Phase 9
operational pilot, rolling OOF, stability, and the sealed holdout remain
separate gates defined in [`configs/data_splits.json`](configs/data_splits.json).
The [second-round semantic review](docs/refactor/semantic_review_round_2.md)
separately grades definition validity and empirical validity for all 17 named
SMC concepts. An executable/replayable primitive is not thereby predictive;
an unsupported study does not by itself erase a valid descriptive primitive.
The [DOL/Belief/Temporal supplement](docs/refactor/dol_belief_temporal_supplement.md)
records the correlated-evidence contract, temporal-versus-ancestry boundary,
the historical FVG first-concrete-lifecycle freeze, and the exact 2024-06
data-role audit. Foundation v2 now defines and replays a true geometric first-
retest event, but its empirical estimand/study remains separately gated. June
Week 4 passed outcome-blind input preflight, and its
[temporal/branching design](experiments/manifests/smc_semantics_v1_2_2024_06_week4_temporal_branching_construct_v1.yaml)
is now frozen. The strict identity validator passes without opening market
data, but execution bindings, mechanism materialization, semantic replay, and
research results remain absent and the manifest does not authorize a run.
No probability study has been run, and sealed OOS remains closed.

The canonical market-language foundation requested by the target-state prompt
is now implemented as the separately versioned
[`smc_semantic_foundation_v2.0`](docs/refactor/canonical_semantic_foundation_v2.md)
projection. It completes registered object generations, lifecycles, geometric
Swing nesting, same-level rearm, structure/range/zone separation, first
reinteraction, multi-bar ancestry, and the shared factual outcome layer
without rewriting v1.2 facts. Overall research-program conformance remains
**partial**: empirical fits, complete Execution Research, operational Shadow
Live, rolling OOF, and sealed OOS gates remain incomplete.

The runtime event envelope is governed by **SMC Semantic Specification v1.2**.
Every runtime-emitted `MarketEvent` exposes separate `event_time` and
`known_at` clocks, carries `semantic_version=smc_semantics_v1.2`, and has an
immutable payload. See
[`docs/smc_semantic_specification_v1.md`](docs/smc_semantic_specification_v1.md)
and the machine-readable
[`semantics/registry_v1_2.yaml`](semantics/registry_v1_2.yaml). The frozen v1.1
registry and parameters remain unchanged for historical artifact verification.

The additive canonical object/state contract is governed by
[`semantics/foundation_v2_0.yaml`](semantics/foundation_v2_0.yaml), identity
`smc_semantic_foundation_v2.0`. It consumes only exact normalized or v1.2
semantic-atomic parents and publishes technical replay records with no action
authority. Its definitions and before/after audit are documented in
[`Canonical Semantic Foundation v2`](docs/refactor/canonical_semantic_foundation_v2.md).
The checked-in production model must explicitly set
`observer.canonical_foundation_enabled=true`; missing, false, or non-boolean
values fail closed during `ContinuousSMCEngine` construction. The same model
must bind `observer.canonical_foundation_registry` and the exact canonical
registry identity; the Engine strict-loads both, freezes the admitted version/
identity into checkpoint state, and Shadow Live compares them. Engine
checkpoint schema v3 is the first schema that includes this projection, so
older schemas cannot resume into the current runtime.
The exact June-2024 bounded construction/replay census, performance A/B, and
current Foundation-enabled 200-clock Engine file parity are kept in the
foundation specification's
[release-verification table](docs/refactor/canonical_semantic_foundation_v2.md#replay-test-and-empirical-boundary);
they are engineering evidence, not a 6,900-clock or real-time multi-day Phase-9
pilot or empirical validation.

The Eye now also publishes an event-sourced hierarchical market contract:
independent `TimeframeState` objects, cross-timeframe `RelationState` objects,
a separate `SessionState`, and one `MarketSnapshot` containing the current
facts plus the event delta. Parent structure is never changed by child-frame
votes; a child reversal sequence remains a warning/hypothesis input until the
parent's own registered confirmation completes. Normal runtime snapshots have
`authority=atomic_event_reducer`; `STATE_PROJECTION` is compatibility transport,
not a second source of market facts. v1.2 adds append-only
micro/internal/structural/external Swing roles and executable same-timeframe
IRL/ERL candidate membership without rewriting earlier events.
That hierarchy remains a causal role-depth projection. Foundation v2 adds an
independent pure geometric containment tree; geometric depth never implies
structural role. The existing Group 4 range is
the sparse Mature Balance Range evaluated by the natural scan; its legacy
`DEALING_RANGE_*` event name remains compatibility state. Foundation v2
publishes separate Structural Range and Balance Range identities and separate
normalized locations; this definition split is not predictive validation.

The immutable historical January-2024 v1.1 Phase-5 structural diagnostic is
bound by the frozen
[`v1.1 manifest`](experiments/manifests/smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.yaml),
with machine-readable
[`results`](experiments/results/smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.json),
a derived
[`report`](experiments/results/smc_semantics_v1_1_2024_01_phase5_diagnostic_v2.md),
and an explicit
[`completion and limitation report`](docs/refactor/phase_2_5_completion_report.md).
It is an Eye-only structural Signal Research diagnostic: it did not construct
or evaluate the Brain, MBO mechanism, or execution, and the source-linked
nested chain has zero samples at E3–E6. It is development/calibration evidence
only—not OOS evidence or trading authority. Its 24.04% matched-control coverage
and zero E3–E6 samples are historical v1.1 findings, not v1.2 implementation
limits. Research protocol v3 implements a corrected M5 E1–E6 chain,
episode de-duplication, four separate control families, pseudo-levels, forward
time shifts, adjacent-stage deltas, exact McNemar, and fixed-family Holm.
Strict ancestry follows only `source_event_ids`; cross-event composition must
share an exact normalized M5 source bar and is reported separately from
ancestry. Forward shifts must remain in the exact registered stratum. Because
different matched pairs may still have overlapping outcome windows, all v3
inference remains descriptive and unvalidated. A separately frozen
[`v1.2 r2 manifest`](experiments/manifests/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.yaml)
and [complete result](experiments/results/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.json)
now record the full January development diagnostic: E1–E6 episode counts are
1,124 / 317 / 17 / 1 / 1 / 0; quiet and non-sweep controls match 372 and 62,
while pseudo and forward-shift match zero. No Holm comparison rejects, and the
result grants no inference, fit, semantic-acceptance, OOS, or trading
authority. The current plan matrix is in the
[`implementation status`](docs/refactor/current_implementation_status.md);
the older completion report remains the historical v1.1 run record.

Phase 6 then completed its preregistered June 2024 primary week and registered
second-week MBO extension. The final registered
[`final manifest`](experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml)
and [`result`](experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.json)
admit only `acceptance_continuation` and `displacement_impact` as Phase 7
evidence. Sweep and MSS remain underpowered, the historical FVG
first-concrete-lifecycle proxy (originally labelled `fvg_retest_response`) is
unsupported, and
the pseudo-zone comparison is descriptive only. The registered extension is
consumed and `further_extension_authorized=false`; no Week 3 is open. This is
mechanism association, not a causal, OOS, model-fit, or trading-authority claim.
The exact Phase 6–9 engineering/validation boundary is summarized in the
[`completion report`](docs/refactor/phase_6_9_completion_report.md).

## Current configuration

Only runtime model-configuration schema version 1 is active (this is separate
from the `smc_semantics_v1.2` semantic identity):

- [`configs/model.json`](configs/model.json): runtime wiring and risk/decision settings;
- [`configs/path_hypotheses.json`](configs/path_hypotheses.json): exact-fingerprint-bound shadow path and DOL-ranking protocols;
- [`configs/dol_probability.json`](configs/dol_probability.json): shadow
  DOL probability/no-target protocol; neither the Brain nor the standalone
  marginalizer publishes a probability result without a fitted/admitted model
  artifact bound to the exact path protocol/model;
- [`docs/refactor/dol_belief_temporal_supplement.md`](docs/refactor/dol_belief_temporal_supplement.md):
  correlated-evidence, temporal/branching, historical FVG lifecycle, and
  2024-06 joint-data boundary;
- [`configs/phase7_probability_fit_admission.json`](configs/phase7_probability_fit_admission.json):
  read-only Phase 7 fit-readiness contract; the checker inspects the 7,381-row
  Phase 6 compact evidence bundle, reports 13 blockers, writes no artifact, and
  does not fit a model;
- [`configs/signal_policy.json`](configs/signal_policy.json): fail-closed Signal Policy and Trade Intent admission rules;
- [`configs/execution_research_v1.json`](configs/execution_research_v1.json): Phase 8 execution-research protocol configuration;
- [`configs/shadow_live_v1.json`](configs/shadow_live_v1.json): Phase 9 no-submission parity protocol;
- [`configs/playbooks.json`](configs/playbooks.json): typed DFP, LSR and FAVR definitions;
- [`configs/primitives_structure_liquidity.json`](configs/primitives_structure_liquidity.json): candle, swing, BOS and liquidity inventory;
- [`configs/primitives_displacement.json`](configs/primitives_displacement.json): incremental displacement episodes;
- [`configs/primitives_zones.json`](configs/primitives_zones.json): FVG and order-block zones;
- [`configs/primitives_range.json`](configs/primitives_range.json): accumulation, dealing range and manipulation;
- [`configs/primitives_entry.json`](configs/primitives_entry.json): entry location, first pullback, reacceptance, micro BOS and path sequence.

Current primitive protocol status:

| Family | Protocol | Status |
|---|---|---|
| Structure/liquidity | `3.2.0-group12.7` | finite real replay and stratified review passed |
| Displacement | `3.2.0-displacement-episode.3` | finite real replay and stratified review passed |
| FVG/order block | `3.2.0-group3.4` | FVG finite replay passed; OB failure coverage sparse |
| Range/manipulation | `3.2.0-group4.1` | The legacy v1.2 atomic protocol detects Mature Balance Range only; additive foundation v2 publishes Structural Range separately |
| Entry/path | `3.2.0-group5.4` | LSR reversal context freezes root/displacement independently of each FVG/OB entry zone; DFP/LSR input authority enabled; FAVR authority disabled and parked |

“Implementation complete” means the typed incremental contract and its
synthetic/boundary tests exist. It is not a profitability or natural-market
authority claim.

The bounded 2023 natural-market review and the reason FAVR remains parked are
recorded in [`docs/natural_authority_2023.md`](docs/natural_authority_2023.md).
The registered outcome-blind full-eye 2023 scan is complete. It ran
Reader -> Observer -> Group 1-5 + Displacement -> lightweight statistics and
retained 353,445 in-window observations without Brain, Decision, Risk,
execution, MBO, PnL, future paths, annual Scene Graph projection or per-minute
artifacts. Its permanent outputs are:

- [`docs/evidence/eye_group1_5_natural_authority_2023_summary.json`](docs/evidence/eye_group1_5_natural_authority_2023_summary.json)
- [`docs/evidence/eye_group1_5_natural_authority_2023_cases.json`](docs/evidence/eye_group1_5_natural_authority_2023_cases.json)

It can be reproduced only through the registered profile:

```bash
.venv/bin/python scripts/run_eye_authority_scan.py \
  --profile eye_group1_5_natural_authority_2023_full_year \
  --force
```

On macOS this runner fails closed unless the interpreter is native arm64;
the repository `.venv` is the supported entry point. During a long run it
atomically refreshes a small, non-evidence `progress.json` independently of
the less frequent checkpoint.

Execution reality in this Eye-only path is `not_evaluated`, not an Eye anomaly.
The sampled transport review is also complete: 30 frozen clocks were replayed,
seven of seven exactly evaluable strict cases passed, and two dependent July
cases remain explicitly prefix-censored because a seven-day cold prefix cannot
reproduce their annual parent identities. MatureBalanceRange is retained as rare
context, while FAVR remains parked.
The earlier reproducible, lightweight Group 4-only result remains separately
frozen historical evidence in
[`docs/evidence/group4_natural_authority_2023.json`](docs/evidence/group4_natural_authority_2023.json).
It can be regenerated only through its registered outcome-blind profile:

```bash
python3 scripts/scan_mature_ranges.py \
  --profile group4_natural_authority_2023_full_year \
  --force
```

That scan executes the production Group 1–2 and Group 4 reducers with all five
enabled liquidity-pool source timeframes. It does not use Brain, Decision,
Risk, PnL, MBO, future paths or threshold search.

`ContinuousSMCEngine.from_config(..., runtime_mode=...)` requires an explicit
development or live mode. Constructing it with `runtime_mode="live"` is
rejected unless the single top-level release
readiness block, DFP/LSR Group5 input authority, economic validation, rolling
OOF and MBO stability are all explicitly complete. The current configuration
intentionally fails that gate; typed DFP/LSR development actions are not
live-trading authorization.

Internal semantic event identities remain version/hash bound where needed, but
the runtime does not select between historical product generations.

## Data boundaries

All market data stays under `data/`:

- OHLCV-1m covers 2017–2026 through the strict previous-session contract front;
- June–July 2024 MBO supplies observed spread, depth, fillability and costs;
- August–December 2024 MBO remains behind `.HOLDOUT_SEALED` until the one final
  execution reveal is explicitly authorized.

`configs/data_splits.json` binds the causal OHLCV artifact and manifest, the MBO
development partition manifest and execution artifacts, and the sealed DBN and
vendor manifest by exact SHA-256. MBO is execution reality only; it cannot
define SMC primitives or become a buy/sell label.

For roll-sensitive research, materialize the previous-session front with:

```bash
python3 scripts/prepare_causal_front.py \
  --source data/raw/nq_ohlcv_1m/glbx-mdp3-20170101-20211231.ohlcv-1m.dbn.zst \
  --source data/raw/nq_ohlcv_1m/glbx-mdp3-20220101-20251231.ohlcv-1m.csv \
  --source data/raw/nq_ohlcv_1m/glbx-mdp3-20260101-20260714.ohlcv-1m.dbn.zst
```

The default output, roll-map, and manifest paths must all be absent; the
materializer deliberately refuses to overwrite them. For a new run, supply
unused `--out` and `--roll-out` paths rather than deleting or replacing a
registered artifact.

Each Globex session uses only the highest-volume outright contract from the
strictly prior completed session. The first source session is omitted; there is
no current-session fallback.

## Runtime and replay

The development runtime API is
`ContinuousSMCEngine.from_config("configs/model.json", runtime_mode="development")`,
followed by one `on_bar` call per newly completed 1m bar. Its fixed causal order
is reader → Observer normalization/detectors → atomic event store/reducers →
`MarketSnapshot` → Scene Graph → existing development Brain → Decision → Risk.

A bounded development replay can be run with:

```bash
python3 scripts/run_continuous_replay.py \
  --source data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet \
  --start 2022-01-03 --end 2022-02-01 \
  --output outputs/development_replay
```

Normal replay writes light decision rows, an aggregate summary, progress,
checkpoints and resumable shards. `--brain-calibration` adds only typed
calibration rows inside the registered calibration window. `--shadow-outcomes`
independently records frozen, outcome-blind Shadow candidates and later derives
sharded episode/root outcomes; add `--shadow-details` only when per-candidate
challenge and motif rows are needed. The two modes can be enabled together but
neither enables the other, and aggregate Brain funnels require the separate
`--brain-diagnostics` flag. Shadow output never feeds Brain, Decision or Risk.
Shadow summary schema v4 reports raw revision rows, selected root-representative
rows and unbound rows separately; outcome-quality statistics use only the
root-representative population. Finalization keeps episode/root de-duplication
in a disposable local SQLite index and removes that index before publishing.
Detailed motif rows retain at most 40 deterministic root IDs as examples; they
are not an exhaustive motif-membership list.
`--mbo-execution` supplies
observed spread/depth/fillability to Observation, Decision and Risk;
`--simulate-execution` enables the existing next-bar position feedback path.
This retained simulator remains conservatively OHLCV-bar based. It is distinct
from the implemented Phase 8 seven-entry-method evaluator core and immutable
order FSM v1.5. The evaluator's `phase8_execution_research_v1.1` contract now
separates entry good-til-time from the later analysis horizon: post-GTT entry
fills are forbidden while an existing position's stop/target observation may
continue. A target reached before a pending remainder fills cancels that
remainder, and secondary realized-spread/path censoring does not remove an
otherwise complete primary implementation-shortfall pair. Stop/invalidation
and target prices must lie on the frozen tick grid or the research intent fails
closed. The evaluator has
not frozen exact method-price provenance or
wait/cancel/stop/target variants, and still has an incomplete, unauthorized
manifest with no empirical result. A read-only
[`check_phase8_execution_readiness.py`](scripts/check_phase8_execution_readiness.py)
validates that blocked template without opening ledgers or writing artifacts;
it reports 12 blockers, including `formal_runner_not_implemented`. There is no
formal Phase 8 study runner. The FSM is an engineering state machine and does
not submit broker orders. It remains a standalone exact-intent consumer;
`TradeIntent -> RiskApproval -> FSM` has not replaced the current
Engine/Decision/Risk/simulator execution path.

Phase 9 also provides a `phase9_shadow_live_v1.2` engineering parity runner.
It records exact causal bar/execution/account evidence, immutable evidence IDs,
frozen instrument mapping, full registered state digests, journal/failure/
gateway state, and runs behind a `NullExecutionGateway` that forbids external
submission. Cold replay and restart parity are fail-stop. A retained hash-bound,
tick-normalized schema-v2 June Week-1 cold-start input contains 6,900
clocks (6,899 real and one synthetic; SHA-256
`fd9e48850d1657cf369e3e617e3e8b464790e9823f48c79b01f65bfc111a46e4`).
A bounded real-data rehearsal checkpointed at 100 clocks, resumed to 200, and
matched an independent 200-clock cold replay exactly. It also exposed and
closed a cross-process hash-order defect in revised Scene-Graph edge IDs.
That receipt binds the pre-supplement path protocol/model bytes
(`5213b3d6…` / `4214da19…`). The new global dependency-cluster contract changes
those bindings, so it remains historical engineering evidence. The current
Foundation-enabled model has separately passed the same exact 200-row prefix
under its current model/registry bindings; that result is recorded once in the
[Foundation release table](docs/refactor/canonical_semantic_foundation_v2.md#replay-test-and-empirical-boundary)
and the portable
[machine receipt](docs/evidence/phase9_foundation_v2_prefix_200_receipt.json)
for that non-portable local evidence.
This closes only the current-prefix binding check; any complete 6,900-clock
rehearsal must still be rematerialized rather than reuse the historical output.
The bounded 200-row input has its own local `COMPLETED.json`; there is no
6,900-clock completion marker and no actual real-time multi-day shadow pilot,
so the operational Phase 9 gate is not passed.
The retained run proves deterministic parity for its bound source snapshot,
not the target operational
metrics for event/relation churn, evidence-belief consistency, signal expiry,
or DOL stability. The complete historical path also retains full-prefix and
co-resident live/cold capacity costs that must be removed or protocol-versioned
before treating a 6,900-clock rehearsal as complete.
A new read-only capacity preflight extrapolates from the completed 200-clock
prefix without creating an Engine or replaying a clock. It verifies the
`COMPLETED.json -> checkpoint manifest SHA-256` binding, then estimates a
342,420,401-byte checkpoint, 370,193,384 bytes of retained output, and a
723,016,956-byte peak-working-set lower bound for 6,900 clocks. The historical
reference bindings differ from the current model/path identities, nonlinear
consistency cost remains unresolved, and `full_6900_replay_authorized=false`;
this is capacity planning, not a completed run or pilot.
The input, checkpoint, and journal are ignored local engineering files rather
than portable repository evidence; exact paths, hashes, reproduction commands,
and the no-replace publication contract are recorded in the
[Phase 6–9 gate report](docs/refactor/phase_6_9_completion_report.md#phase-9-parity-harness-implemented-pilot-pending).

Unexplained episodes have no action authority. Normal replay keeps only their
aggregate strata plus currently open roots in checkpoint state. Use
`--include-unexplained-episode-details` only for diagnostics; it writes a
deterministic stratified sample capped at 40 cases rather than the full history.
Full thesis/playbook comparisons are likewise shadow diagnostics, not part of
the development `MarketBelief` action-candidate interface.

Visualization is opt-in and limited to preselected decision clocks. Repeat an
aware `--visualize-at` value for at most 40 sampled cases; the replay renders
each view immediately after its `step.snapshot` and writes a single index under
`<output>/visualizations/`. With no such option, the replay creates no images.

```bash
python3 scripts/run_continuous_replay.py \
  --source data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet \
  --start 2022-01-03 --end 2022-02-01 \
  --output outputs/development_replay_visuals \
  --visualize-at 2022-01-10T10:31:00-05:00 \
  --visualize-at 2022-01-14T14:06:00-05:00
```

The former frozen-packet, sealed-reveal and identity-bound AI audit stack has
been retired from the development runtime. After the vertical chain is stable,
a small sampled diagnostic can be rebuilt around only three things: a bounded
minute trace, an independent future view, and AI comments translated into
computable sequence primitives. It will not be part of annual replay output.

Replay stores bounded Parquet shards and checkpoint state so an interrupted run
can continue with `--resume`. Source, time window, warm-up, current model
configuration and execution mode must match the checkpoint. Input and current
configuration identities are recorded once per run, not repeated in every row.

## Development order

1. Use the completed v1.2 r2 diagnostic only as development evidence; do not
   relax its sparse-stage/control thresholds or reuse the historical v1.1
   result as v1.2 evidence. Foundation v2 now defines geometric nesting,
   complete Structural Leg paths, explicit generations, structural FVG expiry,
   and separate range types, but none has empirical authority yet. Preregister
   later independent validation, Protected-Swing survival, and matched
   Origin-Zone first-retest studies; do not invent fixed FVG TTLs, range
   extensions, or importance/probability claims from those definitions.
2. Keep the frozen June Week-4 temporal/branching design outcome-blind; bind a
   separate executable revision to the exact mechanism artifact, runtime
   identities, and 6,900-clock census, then materialize and run that bounded
   diagnostic once. Only after that construct gate may a separate study fit,
   calibrate, and admit path likelihoods, DOL probabilities, and Signal Policy
   outcomes. Phase 7 cannot emit a production Trade Intent before those
   artifacts pass admission.
3. Freeze method-price semantic provenance and the wait/cancel/stop/target
   variant estimands, bind a non-zero intent/minute ledger, complete the Phase 8
   manifest and formal runner, run the fixed same-intent comparisons, and audit
   the empirical result. Keep submission disabled.
4. Freeze the Phase 9 operational metrics, close the historical capacity
   residual if that rehearsal remains useful, then run the real-time multi-day
   no-order pilot and prove live/replay/restart/failure parity.
5. Only after those gates pass, run rolling OOF, stability checks, and the
   sealed holdout once under their preregistered governance.

The large hash-bound research ledgers are formal evidence, not cleanup files.
Before publishing this checkout as an ordinary Git repository, move them to
Git LFS or an immutable artifact store while preserving path, SHA-256, row
count, and retrieval identity.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/playbook_preregistration.md`](docs/playbook_preregistration.md), and
[`docs/self_review_checklist.md`](docs/self_review_checklist.md). The exact
implementation/remaining-work matrix is
[`docs/refactor/current_implementation_status.md`](docs/refactor/current_implementation_status.md).
