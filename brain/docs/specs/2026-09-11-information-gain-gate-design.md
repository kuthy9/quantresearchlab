# Information-gain gate: do the Eye's events change the future's distribution?

Status: design, awaiting review. Date: 2026-09-11.

## 1. The decision this spec serves

The redesign brief of 2026-09-11 replaces per-minute trajectory retrieval with
event-driven re-conditioning: the Eye keeps observing every minute, the Brain
is invoked only when the Eye's facts change the conditional distribution of
the future, and hypotheses are the modes of that distribution. Its own first
principle is that *historical data decides which facts have predictive value*.
So before any joint conditional model (C), hypothesis lifecycle (D) or
event-driven orchestration (E) is built, one measurement has to come back
positive:

> On out-of-sample clocks at which the Eye published a structural event, does
> a model that also sees the recent event sequence Δₜ predict the next sixty
> minutes better than the same model seeing only the raw tape and the Eye's
> state Sₜ?

This spec defines that measurement so that it is decided before the numbers
exist, and the one engineering change needed to run it in reasonable time.
C, D and E are out of scope; they start only if §5.9 returns PASS.

## 2. What was measured before writing this

All on the sealed source `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`,
sessions 2022-01-09..13, with the Eye built from `configs/model.json` by
`brain/research/trajectory_dataset.build_eye`.

- **The Brain reads Sₜ only.** `brain/core/hypothesis_proposer.observation_features`
  reads the `MarketSnapshot`'s per-scale structure, delivery, range, liquidity
  and zone state into 150 components. It reads none of the 61 `EventKind`s the
  Eye publishes in `MarketObservation.semantic_events_this_update`. Δₜ has never
  been an input.
- **Only magnitude has been predictable so far** ([brain/docs/README.md](../README.md),
  "The signal is magnitude"): on every-minute clocks, `range_60` reaches R² 0.21
  while `asymmetry_60` and `r_15/30/60` sit at or below zero under every model,
  and the Eye's 142 state components add ΔR² ≈ 0 over the raw tape. That study
  never conditioned on events; this gate does.
- **Eye per-bar cost grows with bars seen.** Real-tape timing per 500 bars:
  9.0 → 18.3 → 22.0 → 26.5 → 30.1 → 35.5 → 38.1 → 44.0 s over the first 4,000
  bars (55 → 11 bars/s). Profiled growth sits in three places, all of the form
  "re-process a retained collection on every bar":
  1. `contract/eye/interaction.py:397 validate_canonical_bindings` re-admits
     every nested DTO of the current `InteractionUpdate` on every bar. The
     update carries every closed path until `maximum_context_states` (256, a
     sealed parameter in `semantics/parameters_v1_3.yaml:149`) forces eviction;
     at bar 3,000 it held 187 paths, 169 of them closed for a median 1,640 min.
  2. `eyes/core/event_memory.py:691 sync_retained_entity_timelines` iterates
     every retained entity timeline; 2,012 at bar 3,000, of which 1,616 belong
     to entities no longer live (924 are swings kept by `history_limit`).
  3. `eyes/core/event_store.py:1771` inside `build_structural_legs` scans the
     whole available-event map for `BAR_COMPLETED` events on every call.
  Both collections are bounded by configuration, so the curve plateaus near
  0.1 s/bar rather than growing forever; 110 sessions would take 4–5 h today.
- **Event rates on the real tape** (per 1,380-bar session, any scale):
  `*_state` re-publications and `bar_completed` fire on 67–100 % of bars;
  1-minute transitions such as `sweep_confirmed` (468/session),
  `acceptance_confirmed` (414) and `level_touched` (1,132) fire on 17–33 % of
  bars each; 5-minute transitions run 10–130/session; 15-minute 2–27; 1-hour
  0.3–6. Bars carrying at least one transition event: 75.4 % at ≥1m, 17.2 % at
  ≥5m, 3.6 % at ≥15m, 0.8 % at ≥1H.

## 3. Scope

Two parts, in order.

**Part 1 — Eye per-bar cost, output unchanged.** Make the three hot paths cost
proportional to what changed this bar, not to what is retained, with the Eye's
published output byte-identical before and after. No parameter, protocol,
specification or test expectation changes; the atomic identity
`f92b24c8…` is untouched.

**Part 2 — the gate.** One continuous Eye run over 110 sessions of 2022, an
event log beside the existing state dataset, and an extension of
`brain/scripts/predictability_gate.py` that adds event clocks, Δₜ features,
first-passage targets, classification metrics and a per-kind ablation, with a
pre-registered verdict.

Not in scope: changing what the Eye retains or publishes (fix "a" of the
discussion — deferred to the Eye contract redesign, where the sequence window
L will decide it on data); rebinding `shares/core/engine.py`; any model that
is kept for runtime use.

## 4. Part 1 — Eye per-bar cost

### 4.1 Invariant

For every bar of a replay, `content_hash(to_primitive(observation))` and the
hash of every `semantic_events_this_update` element are identical before and
after the change. This is the acceptance test, not a hope: the harness in
§4.5 records the per-bar hash stream from the current code before any edit.

### 4.2 Changes

1. **`InteractionUpdate.validate_canonical_bindings`** — memoise re-admission
   per DTO instance. Every nested DTO (`EntryLocationState`,
   `ReacceptanceState`, `MicroBreakFact`, `PathSequenceState`,
   `PathSequenceStep`, …) is a frozen dataclass; an instance that passed
   `exact_values` once passes forever. Keep a module-level
   `weakref.WeakSet` (or `id`-keyed cache tied to the instance) of validated
   instances and skip the reconstruction for members already in it. The
   shape check on the update itself (`set(self.__dict__)`) stays. A DTO that
   is genuinely new (this bar's transition) is validated exactly as today.
2. **`EventMemory.sync_retained_entity_timelines`** — the future-check loop
   over all timelines and the two `retained_clocks`/`retained_event_ids`
   set comprehensions recompute from scratch each bar. Maintain them
   incrementally: track the maximum `observed_at` per timeline at `append`
   time (the tail is already the maximum, so the check becomes one
   comparison per timeline key that changed this bar), and keep the retained
   clock and event-id sets as counters updated on append/drop rather than
   rebuilt. Dictionary narrowing to `retained` stays as written.
3. **`EventStore` `BAR_COMPLETED` lookup** (`event_store.py:1771` and the
   sibling scans in `build_structural_legs`) — keep a per-scale ordered index
   of `BAR_COMPLETED` / `NORMALIZED_DATA` events maintained on admission, and
   read the eligible range from it instead of filtering
   `available_events.values()`.

Each change is one commit, each verified against the hash stream on its own,
so a regression is attributable.

### 4.3 What must not change

`maximum_context_states`, `history_limit`, eviction order, retention
semantics, `EventKind` vocabulary, any `to_primitive` payload, the semantic
registry and parameters, `configs/*`. `eyes/tests/test_eye_module_boundary.py`
must keep passing (no new downstream import in `eyes/core/`).

### 4.4 Expected result

Per-bar cost flat at roughly the current bar-0 level (≈0.02–0.03 s/bar,
35–50 bars/s) across a multi-session run; 151k bars in about one hour on one
core. Memory still grows to the configured caps exactly as today.

### 4.5 Verification

- `eyes/scripts/replay_hash_stream.py` (new, research script): runs the Eye
  over a named window and writes one line per bar —
  `asof, observation_hash, events_hash, n_events` — plus wall time per 500
  bars. Run once on the current code over 2022-01-09..13 (4,500 bars) and
  keep that file as the reference; run after each change and `diff`.
- Existing Eye suite: `.venv/bin/python -m pytest eyes/tests shares/tests --ignore-glob='* 2.py'`
  before and after; the same modules collect and the same tests pass.
- The timing table of §2 re-measured after all three changes; the spec's
  claim is met when the eighth 500-bar block costs no more than 1.5× the
  first.

## 5. Part 2 — the gate

### 5.1 Data

- Source: the OHLCV path bound in `configs/data_splits.json`
  (`nq_1m_previous_session_front_v2_3_2017_2026.parquet`, sha256
  `84c9ed4d…`), loaded through `shares.core.io.load_ohlcv` and
  `iter_completed_bars` exactly as `trajectory_dataset.build_dataset` does.
- Warm-up: sessions 2021-12-27 → 2021-12-31 (inside the `development`
  window; fed to the Eye, never sampled).
- Training: the first 90 Globex sessions of 2022, 2022-01-03 → 2022-05-09.
- Out-of-sample: the next 20 sessions, 2022-05-10 → 2022-06-06.
- 151,074 completed bars in training + OOS. All inside the `calibration`
  window of the split registry; `rolling_oof` and `sealed_holdout` untouched.

### 5.2 One Eye run, two artefacts

`brain/research/trajectory_dataset.build_dataset` is extended (not forked)
to record, beside the existing per-clock `features` / `prices` /
`future_*` arrays, an **event log** with one row per published transition
event: `known_at`, `kind`, `timeframe`, `direction`, `side`, `strength`,
`price`, `entity_id`, `lifecycle`, `event_id`. Saved as
`events.parquet` next to `dataset.npz` under `outputs/information_gain_gate/<run_id>/`
(an ignored directory). The run id is the sha256 of the model path, the
split registry identity, the window strings and the Eye's atomic identity.

A *transition* event is any `EventKind` whose value does not end in `_state`
and is not `bar_completed` or `market_epoch_reset`. `*_state` events are
lifecycle re-publications of retained entities, not changes; they are what
made "every minute has an event" true in §2.

### 5.3 Targets (horizon 60 minutes, ATR = the snapshot's one-minute ATR)

| name | type | definition |
| --- | --- | --- |
| `fp_1.0_1.0` | 3-class | first of: high ≥ +1.0 ATR, low ≤ −1.0 ATR, neither within 60 min |
| `fp_1.0_0.5` | 3-class | first of: high ≥ +1.0 ATR, low ≤ −0.5 ATR, neither |
| `fp_0.5_1.0` | 3-class | first of: high ≥ +0.5 ATR, low ≤ −1.0 ATR, neither (the mirror of the row above) |
| `asymmetry_60` | continuous | `(U − D)/(U + D)` as in the existing gate |
| `range_60` | continuous | `U + D`, the known-predictable control |

The three first-passage targets are the verdict targets. `asymmetry_60` is
reported with the existing five-condition verdict as a second directional
reading. `range_60` is a control only: a Δₜ that helps `range_60` and nothing
else says the events carry volatility, not direction, which is the result
already on record.

### 5.4 Clocks

Clock Cᵏ = the set of bars t at which at least one transition event with
timeframe ≥ k was published (`known_at == t`).

The verdict is taken over a **family of three clocks**, none of them
privileged. Which clock carries an improvement, if any, is an output of the
gate — the time scale the information lives on — not an assumption fed into
it. On the real tape (§2), per session and for the 90 + 20 split:

| clock | bars covered | training clocks | OOS clocks | what it can and cannot test |
| --- | --- | --- | --- | --- |
| **C1** | 75.4 % | ≈94,000 | ≈21,000 | the only clock that tests intra-session micro-sequences (Sweep → Reclaim → Displacement on 1m–5m) at their natural rate; differs from the every-minute clock by a quarter of the bars, so it carries most of that clock's dilution |
| **C5** | 17.2 % | ≈21,000 | ≈4,700 | the most power among the sparse clocks; still admits 5m `displacement_observed` at ≈131/session |
| **C15** | 3.6 % | ≈4,400 | ≈1,000 | closest to a few-triggers-a-day reading; thin enough that a wide interval is a live outcome |

- C60 is reported descriptively (≈11 clocks/session is too thin for a
  20-session OOS) and never enters the verdict.
- The every-minute clock is reported once, as the dilution reference, not as
  a verdict input.

The clock decides *when* M₀ and M₁ are fitted and scored, never *what* M₁
sees: on every clock, Δₜ carries all 49 transition kinds on all five scales
(§5.5). Both models are fitted and scored on the same clock set, so the
comparison is "given that an event happened here, does knowing which events
happened, and in what order, help".

### 5.5 Feature groups

- **M₀ = RAW + S** — the existing `raw` group (8 tape components) plus the
  existing `eye` group (142 state components), i.e. today's `raw+eye`.
- **M₁ = RAW + S + Δ** — M₀ plus the event sequence Δₜ over the window
  L = 120 minutes, encoded three ways, all read only from events with
  `known_at ≤ t`:
  1. per (transition kind × timeframe): minutes since the last occurrence
     (capped at L; L when none), count within L, direction of the last
     occurrence (+1 / −1 / 0). 49 transition kinds × 5 scales × 3 = 735 columns,
     mostly constant on any one clock;
  2. the ordered recent sequence, **one track per scale**: for each of the
     five scales, the last K = 4 transition events published on that scale,
     each slot carrying kind id, minutes ago, direction, strength and an
     empty flag — 5 tracks × 4 slots × 5 fields = 100 columns. Slot order is
     arrival order within the track, so *Sweep → Reclaim → Displacement*
     and *Displacement → Sweep → Reclaim* are different vectors. Tracks are
     per scale because 1-minute transitions arrive about every 1.3 minutes
     on the real tape (§2); a single shared track of any practical length
     would hold nothing but the last few 1-minute events and no 15-minute
     or 1-hour sequence would ever be visible to the model;
  3. the trigger set at t: a one-hot over kinds fired at t itself.
  No hand-set weights anywhere: every coefficient or split is fitted inside
  the training window.

L and K are fixed here so they cannot be tuned to the OOS result.

### 5.6 Models and fitting

- 3-class targets: multinomial logistic regression with L2 penalty (the
  linear reference) and LightGBM `multiclass` (the interaction-capable
  reference). Continuous targets: `RidgeCV` and LightGBM regression as in
  the existing gate.
- Penalty and early-stopping rounds selected by the existing purged, blocked
  CV inside the training window (`select_ridge_alpha`, LightGBM's purged
  tail). Nothing is selected on the OOS sessions.
- Standardisation fitted on the training rows only (`standardize_pair`).
- Purge and embargo as in `build_folds`: a training row is dropped when
  `t + 60 min` reaches the embargo; embargo 60 minutes.

### 5.7 Folds

- **Primary fold:** train 90 sessions → OOS 20 sessions (§5.1).
- **Rolling folds:** within the same 110 sessions, train 60 → holdout 10,
  step 10 → 5 folds, for the consistency and drop-best-fold conditions the
  existing verdict already applies. The Eye dataset is built once; only the
  fits repeat.

### 5.8 Metrics

- 3-class: OOS multiclass log-loss of M₁ minus M₀ (negative is better),
  per clock as the paired difference series; interval by session-block
  bootstrap (`block_bootstrap_interval`, 2,000 draws), pooled across folds
  by `pooled_interval`. Accuracy and the class prior are reported beside it.
- Continuous: OOS R² and ΔR² as today.
- **Ablation**, on each of the three family clocks and the primary fold:
  M_Full − Eᵢ for every transition kind i (drop all Δₜ columns that mention
  kind i on every scale, refit, rescore), reported as a ranked table of
  Δlog-loss per clock. It informs which events the Brain would
  later learn from; it does not enter the verdict.

### 5.9 Pre-registered verdict

The gate returns **PASS** when, for at least one cell of the family
{`fp_1.0_1.0`, `fp_1.0_0.5`, `fp_0.5_1.0`} × {C1, C5, C15}, all of the
following hold for at least one of the two model classes:

1. the primary-fold OOS log-loss of M₁ is below M₀ and the pooled
   session-block bootstrap interval of the difference lies below zero, at
   α = 0.10 after Holm correction across the 9 (target × clock) tests;
2. M₁ beats M₀ on ≥ 80 % of the rolling folds;
3. dropping the best rolling fold does not flip the mean difference.

The passing cells name the clock, and therefore the time scale, the
information was found on; that is recorded as a finding, not assumed.
Everything else — `asymmetry_60`, `range_60`, C60, the every-minute clock,
the ablation tables — is reported and not judged. A PASS starts C; a FAIL is
recorded with the full tables and stops the redesign at the measurement,
which is the outcome the brief itself allows for.

### 5.10 Outputs

- `outputs/information_gain_gate/<run_id>/dataset.npz`, `events.parquet`,
  `folds.json`, `results.csv` (one row per group × target × clock × model ×
  fold), `ablation.csv`, `verdict.csv`.
- `brain/docs/evidence/<run date>_information_gain_gate.md`, dated on the day the run completes: the receipt —
  run id, data identities, the §2 timing table re-measured, the verdict
  tables, and the decision taken.

## 6. Tests

Part 1: the hash-stream harness (§4.5) is the acceptance test; unit tests
are added only where a change introduces new state (the memoisation set, the
incremental counters, the bar index) to pin that a stale entry cannot
survive an eviction or a boundary reset.

Part 2, all on synthetic sessions from `shares/tests/helpers.session_bars`
so they run in seconds:

- `events.parquet` has one row per transition event and none for `*_state`
  or `bar_completed`; every `known_at` equals the bar it was published on.
- Clock construction: a bar with only 1m transitions is in C1 and not in C5;
  a bar with a 15m transition is in C1, C5 and C15.
- Δₜ encoding reads nothing with `known_at > t` (a planted future event
  leaves the vector unchanged), distinguishes the two orderings of the same
  three events, and keeps a 1-hour event in the 1H track when forty 1-minute
  events arrive after it.
- First-passage labels on hand-built futures: upper-first, lower-first,
  neither, and the same-bar tie resolved conservatively (both touched on one
  bar → the adverse side first, the same conservative reading `configs/model.json` names `same_bar_resolution: conservative` for the risk engine).
- The ablation drops exactly the columns of one kind.
- The verdict function on a fabricated results table returns PASS and FAIL
  for the intended cases and applies Holm across six tests.

## 7. Risks and what is deliberately not decided here

- **Thin OOS on C15** (~1,000 clocks) and **dilution on C1**. The family
  verdict accepts that the two ends of the family fail for opposite reasons;
  a cell whose interval is wide on both sides is inconclusive, not a PASS,
  and the record says so. Holm across nine tests costs some power against
  the six-test alternative with a privileged clock; that price buys not
  having chosen the time scale in advance.
- **Leakage through S.** `observation_features` reads the snapshot at t; the
  existing gate already established it carries no future. The event log is
  filtered on `known_at`, not `observed_at`, for the same reason.
- **Eye retention plateaus, not grows.** If Part 1 does not bring the eighth
  500-bar block within 1.5× of the first, the run still completes (4–5 h)
  and Part 2 proceeds; Part 1 is a cost, not a correctness, prerequisite.
- **Which events matter is not decided here.** The clock thresholds (5m,
  15m) are sampling rules, not a claim that 1-minute events are worthless;
  the ablation table is where that question gets its first data.
