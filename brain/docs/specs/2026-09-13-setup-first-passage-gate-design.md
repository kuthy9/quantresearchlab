# Setup first-passage gate: does a Group-5 Setup change where price goes first?

Status: design, approved in discussion 2026-09-13; awaiting review of this
text. Date: 2026-09-13. Branch: `brain-setup-gate` off `brain`.

## 1. The decision this spec serves

The redesign brief of 2026-09-11 stands: the Eye observes every minute, the
Brain is invoked only when a structural event changes the conditional
distribution of the future, and hypotheses are modes of that distribution
over *target versus invalidation*, not trajectory shapes. Its first
principle also stands: historical data decides which facts have predictive
value, before any machinery is built on them.

The information-gain gate of 2026-09-11 asked whether the Eye's event
*sequence* changes a sixty-minute first-passage forecast over ATR barriers.
It returned FAIL twice — run 1 void (barriers in ATR₁ₘ, "neither" 0 %),
run 2 under ATR₆₀ with non-degenerate classes (67 / 15 / 18): eighteen of
eighteen cells failed, every LightGBM interval covering zero and the linear
class 8–18 % worse with the sequence columns (receipt of 2026-09-12, run 2
tables on disk under `aa4be1c91244d0c4/`). What M₀ did beat was the class
prior, by 4–5 % on the symmetric target, and that is the known magnitude
skill: whether ±1 ATR₆₀ is reached at all, not which side first.

This spec asks the narrower question the Brain actually needs answered:

> At the moment the Eye registers or advances one of its own Setups — a
> Group-5 interaction path, `zone_return` or `pool_reversal` — does knowing
> the Setup change the probability that price reaches its liquidity target
> before its failure boundary, **beyond what the barrier geometry, the
> remaining session and the volatility already imply?**

The baseline is the part that changed. Under a driftless walk the
target-first probability is `d_f / (d_t + d_f)`: a function of two distances
and nothing else. A model that sees only the distances gets that for free.
The gate therefore measures the Setup against a geometry-only model, so a
PASS means information *about the Setup*, not information about volatility
wearing a Setup's name.

C (the conditional proposer), D (a hypothesis pool over Setup × target
nodes), E (event-driven orchestration) and the shadow sink are out of scope;
they start only if §6 returns PASS, under their own spec.

## 2. What was measured before writing this

- **The Eye already has a Setup grammar.** `contract/eye/interaction.py`
  publishes `PathSequenceState` with `context_kind ∈ {zone_return,
  pool_reversal}` (`GROUP5_CONTEXT_KINDS`), a `direction`, a lifecycle
  `active / closed / censored`, and ordered `PathSequenceStep`s whose kinds
  are drawn from `INTERACTION_PHYSICAL_PATH_STEP_KINDS` (`zone_visible`,
  `departure_confirmed`, `first_pullback`, `wick_rejection`,
  `reference_left`, `reference_reclaimed`, `reacceptance_held`,
  `reacceptance_failed`, `micro_break_observed`, `location_left`,
  `pool_swept`, `opposite_displacement`, `accepted_outside`, …). New steps
  arrive as `InteractionUpdate.milestone_transitions`, one
  `(sequence_id, step)` pair each. `brain/core/brain_entry_sequence.py`
  already interprets these paths for the Brain; nothing here invents a
  grammar.
- **Each Setup carries its own invalidation, and the Eye names its draw.**
  A `zone_return` path's `context_id` is an `EntryLocationState`
  (`InteractionUpdate.zone_interactions`): "the first completed-1m visit to
  one exact frozen 5m entry zone", with `direction`, `lower_bound /
  upper_bound / near_edge / far_edge`, `failure_boundary`, `source_zone_kind`,
  `entry_mode`, `first_penetration_fraction` and
  `nearest_visible_draw_distance_points`. A `pool_reversal` path's
  `context_id` is a Group-4 `ManipulationState`
  (`MarketObservation.manipulations`), with `side`, `timeframe`,
  `sweep_extreme`, `penetration_atr`, and the `ReacceptanceState` bound to it
  carries `reference_price`, `failure_boundary`, `reclaim_margin_atr`,
  `hold_margin_atr`.
- **Unswept liquidity per scale is on the snapshot.**
  `TimeframeState.liquidity.unswept_bsl / unswept_ssl` are the price lists
  the proposer already reads (`hypothesis_proposer.py:296`).
- **The existing gate blocks cannot serve.** `events.parquet` of run
  `aa4be1c91244d0c4` recorded `entry_path_step` with `entity_id = None` and
  no step kind, reason or geometry (38,153 step events over 23 weeks,
  ≈1,600 per week; 0 distinct entities). A new Eye pass is required; the
  Group-5 path count per week is therefore unknown until it runs, and §6
  refuses thin cells instead of assuming a count.
- **Everything else is reusable as is.** `gate_family.py`
  (`session_block_bootstrap`, `pooled_session_bootstrap`, `holm`,
  `family_verdict`), `predictability_gate.build_folds` and
  `session_labels`, the Globex-week block protocol and run-id scheme of
  `build_gate_blocks.py`, and the fit helpers of `information_gain_gate.py`
  (`select_logistic_c`, `fit_predict_proba`, `Z_CLIP`,
  `MINIMUM_CLASS_SHARE`).

## 3. Scope

One Eye pass with a path recorder, a label builder on the raw tape, two
feature sets, one gate script, one receipt.

Not in scope: any change to what the Eye retains or publishes; any model
kept for runtime; rebinding `shares/core/engine.py`; the multi-year run
(the 2022 H1 window decides whether a multi-year run is worth its ≈78 h).

## 4. The unit: one Setup at one milestone

An **instance** is one `PathSequenceState` observed at the clock a given
step kind first appears on it. Three clocks:

| clock | step kind | what the Brain would know at that moment |
| --- | --- | --- |
| **K0** | `zone_visible` (zone_return) / `pool_swept` (pool_reversal) — the path's first step | the location exists and price is there; no confirmation |
| **K1** | `reacceptance_held` | the reference was left, reclaimed and held for the protocol's bars |
| **K2** | `micro_break_observed` | a 1m break in the path's direction was bound to the context |

A path contributes at most one instance per clock (its first occurrence of
that step). Clocks are read from `milestone_transitions` at the bar the
step is published, so `known_at` is the completed bar's `asof` and nothing
is known earlier than the Eye knew it.

The verdict family is {K0, K1, K2} × {zone_return, pool_reversal}: six
cells per model class. Which clock and which context kind carries
information, if any, is an output — it is the milestone at which the
Brain should wake, and that is recorded as a finding.

## 5. Measurement

### 5.1 Data and blocks

- Source, split registry, warm-up rule, 110 sessions (2022-01-03 →
  2022-06-06), primary 90 / 20 split: exactly §5.1–5.2 of the 2026-09-11
  spec. Blocks are Globex weeks warmed seven calendar days; the tape is
  read two hours past the next open.
- `brain/research/trajectory_dataset.build_dataset` gains
  `record_paths: bool = False`. When on, for every emit-window bar it reads
  `observation.interaction_update.milestone_transitions` and appends one
  row per pair to a **path log** (`brain/research/path_log.py`), beside
  the existing `features` / `prices` / `future_*` / `events`. The Eye is
  driven once; nothing is forked.
- Blocks write `paths.parquet` next to `dataset.npz` under
  `outputs/setup_gate/<run_id>/blocks/<week>/`. The run id is the existing
  digest plus `"recorder": "paths_v1"`, so it cannot collide with the
  information-gain run. `build_gate_blocks.py` gains `--record-paths` and
  `--output-root`; with the flag off it is byte-for-byte the old builder.

### 5.2 The path log (one row per new step)

| group | columns | source |
| --- | --- | --- |
| identity | `known_at`, `sequence_id`, `context_kind`, `context_id`, `direction`, `path_formed_at`, `path_lifecycle`, `step_id`, `step_kind`, `step_reason`, `step_strength`, `step_observed_at`, `step_ordinal` | `PathSequenceState`, `PathSequenceStep` |
| zone geometry (zone_return) | `lower_bound`, `upper_bound`, `near_edge`, `far_edge`, `failure_boundary`, `source_zone_kind`, `entry_mode`, `first_penetration_fraction`, `eye_draw_distance_points` | `EntryLocationState` by `context_id` |
| pool geometry (pool_reversal) | `source_lower_bound`, `source_upper_bound`, `sweep_extreme` → `failure_boundary`, `penetration_atr`, `source_timeframe`, `reference_price`, `reclaim_margin_atr`, `hold_margin_atr` | `ManipulationState` by `context_id`; `ReacceptanceState` with the same `context_id` if present |
| tape at the clock | `close`, `high`, `low`, `atr_1m`, `rv_30`, `rv_60` (`sqrt(Σ Δclose²)` over the last 30 / 60 one-minute closes, divided by ATR₁ₘ), `minutes_since_open` (from the 18:00 New York open of the session `session_labels` assigns) | the bar and the builder's `history` |
| Eye state at the clock | the 150 `observation_features` components, named by `FEATURE_NAMES` | the same call `build_dataset` already makes |
| liquidity at the clock | `bsl_5m`, `ssl_5m`, `bsl_15m`, `ssl_15m`, `bsl_1h`, `ssl_1h` — nearest unswept level on that scale above / below `close`, NaN when none | `timeframe_states[tf].liquidity.unswept_*` |
| structure at the clock | `ext_dir_5m`, `int_dir_5m`, `last_bos_dir_5m`, and the same for 15m and 1h (+1 / −1 / 0) | `timeframe_states[tf].structure`, encoded as `hypothesis_proposer._direction` does |

Columns are fixed here; the gate reads them, it does not compute new ones
from the Eye. A step whose context cannot be found on the same observation
is logged with NaN geometry and counted; the count goes in the receipt.

### 5.3 Targets and labels (`brain/research/setup_labels.py`)

For each instance, in path direction `s ∈ {+1, −1}`:

- **failure** `F` = the row's `failure_boundary`.
- **target** `T` = the nearest unswept level in direction: for `s = +1` the
  minimum over `{bsl_5m, bsl_15m, bsl_1h}` that is `> close`; for `s = −1`
  the maximum over `{ssl_*}` that is `< close`. The Eye's own
  `eye_draw_distance_points` is recorded beside it as a cross-check, not
  used.
- Distances `d_t = s·(T − close)`, `d_f = s·(close − F)`, both required
  finite and `> 0`; otherwise the instance is **dropped** (no target in
  direction, or price already past the failure boundary). Drop shares by
  reason are reported per cell.
- **Unit**: ATR₆₀ = ATR₁ₘ · √60, as `first_passage.HORIZON_ATR_SCALE`.
- **Scan** the raw tape from the bar after `known_at` (the builder's frame,
  not the sixty-bar window). Horizon `H = min(240 minutes, session end)`,
  where session end is the last bar before the next gap `> 1 minute` in the
  frame index. At each bar: for `s = +1`, target hit if `high ≥ T`, failure
  hit if `low ≤ F`; mirrored for `s = −1`.
- **Label** ∈ {`target`, `failure`, `censored`}: the first hit; both on the
  same bar → `failure` (conservative), with the same-bar share reported.
  No hit within `H` → `censored`.
- Also recorded: `time_to_resolve` (minutes, `H` when censored), `mae`
  and `mfe` in ATR₆₀ up to resolution, `minutes_to_session_end`.

### 5.4 Feature sets (`brain/research/setup_features.py`)

- **M₀ — geometry only** (8 columns): `log(d_t / d_f)`, `d_t + d_f`,
  `d_t`, `d_f` (all in ATR₆₀), `minutes_to_session_end`, `rv_30`, `rv_60`,
  and time of day as `sin / cos` of `minutes_since_open / 1380 · 2π`
  (counted as one feature pair).
- **M₁ — geometry + Setup**: M₀ plus `context_kind` (one-hot),
  `source_zone_kind` (one-hot), `entry_mode` (one-hot), the steps reached
  so far as one column per step kind carrying the step's strength (0 when
  not reached) and one-hot of each reached step's reason, `path_age`
  (minutes since `path_formed_at`), zone width `(upper − lower) / ATR₆₀`,
  `first_penetration_fraction`, `penetration_atr`, `reclaim_margin_atr`,
  `hold_margin_atr`, `source_timeframe` (one-hot), and structure alignment
  `s · ext_dir`, `s · int_dir`, `s · last_bos_dir` at 5m / 15m / 1h.
  Roughly thirty to forty columns; NaN for a group that does not apply to
  the context kind is imputed to 0 after standardisation.
- **Reported, not judged**: the analytic driftless `P_target =
  d_f / (d_t + d_f)` scored as a two-class log-loss on resolved instances;
  and **M₂ = M₁ + the 150 Eye-state components** (the path log's own
  copy, so no instance is lost to `dataset.npz`'s sixty-minute-future
  filter), so the Eye state's increment over the Setup is on record without
  entering the verdict.

### 5.5 Models and fitting

As the 2026-09-11 gate: multinomial logistic (C by blocked purged CV on
the primary training rows; `select_logistic_c`) and LightGBM `multiclass`
(400 trees, early stop on a purged tail). Standardisation on training rows
only; ±10 σ clip. The fit helpers move from `brain/scripts/information_gain_gate.py`
to `brain/research/gate_models.py` unchanged and both scripts import them;
`brain/tests/test_information_gain_gate.py` keeps passing.

### 5.6 Folds

Sessions from `session_labels(known_at)`. Primary fold: train 90 → OOS 20.
Rolling: train 60 → holdout 10, step 10 → five folds. Purge and embargo via
`build_folds` with `embargo_minutes = 240`, the label horizon. Folds are
built per cell on that cell's instances (`folds_for`).

### 5.7 Metrics

Per cell: OOS 3-class log-loss of M₁ minus M₀ as a paired per-instance
series; session-block bootstrap (2,000 draws) on the primary fold, pooled
across folds by `pooled_session_bootstrap`. Accuracy, the class shares, the
analytic reference and M₂ beside it. Per cell, the distributions of
`time_to_resolve` and `mae` by label (quantiles 10 / 50 / 90) — the shape
a Setup × target node would later carry.

## 6. Pre-registered verdict

A cell is **judged** only if, on its primary training rows, every class
holds ≥ 5 % (`MINIMUM_CLASS_SHARE`) and its primary OOS rows number
≥ 200; otherwise it is reported and not judged. Among judged cells, one
model class at a time, the gate returns **PASS** when at least one cell
meets all of:

1. primary-fold OOS log-loss of M₁ below M₀, with the pooled session-block
   bootstrap interval of the difference below zero at α = 0.10 after Holm
   across the judged cells of that model class;
2. M₁ beats M₀ on ≥ 80 % of the rolling folds;
3. dropping the best rolling fold does not flip the mean difference.

These are the §5.9 conditions of the 2026-09-11 spec, applied by
`family_verdict` unchanged. A PASS names the clock and context kind; that
is the milestone the Brain wakes at and the Setup it forms nodes for, and
it starts the machinery spec. A FAIL stops the redesign at the measurement:
the Brain's demonstrable content is then geometry and magnitude, and no
Setup × target hypothesis pool is built on this evidence.

The horizon (240 min), the clocks, the target rule (≥ 5m union, nearest),
the same-bar rule, the 200-row floor and the feature lists are fixed here
so none of them can be chosen on the OOS result.

## 7. Outputs

- `outputs/setup_gate/<run_id>/` (ignored): `run.json`, `blocks/<week>/`
  (`dataset.npz`, `events.parquet`, `paths.parquet`), `instances.parquet`
  (one row per labelled instance with its features), `results.csv`,
  `verdict.csv`, `descriptives.csv`, `gate.log`.
- `brain/docs/evidence/<date>_setup_gate.md`: the receipt, with run id,
  clock coverage, drop and same-bar shares, class shares, the verdict
  table, the reported-not-judged tables and the sha256 of every csv.
- `brain/docs/README.md`: one paragraph under "What actually predicts
  what" pointing at the receipt.

## 8. Code and tests

| unit | file | test |
| --- | --- | --- |
| path log rows from an `InteractionUpdate` | `brain/research/path_log.py` | `brain/tests/test_path_log.py` — synthetic update with one zone_return and one pool_reversal step; geometry joined; missing context → NaN and counted |
| labels on the tape | `brain/research/setup_labels.py` | `brain/tests/test_setup_labels.py` — target-first, failure-first, same-bar → failure, 240-min censor, session-end censor, drop when no target / past failure, mirrored short case, `mae`/`time_to_resolve` values |
| features | `brain/research/setup_features.py` | `brain/tests/test_setup_features.py` — column names and order fixed, analytic P, alignment sign flips with direction, NaN-group imputation |
| fit helpers | `brain/research/gate_models.py` | existing `test_information_gain_gate.py` |
| block builder flag | `brain/scripts/build_gate_blocks.py` | `test_gate_blocks.py` — run id changes with the flag, unchanged without |
| the gate | `brain/scripts/setup_gate.py` | `brain/tests/test_setup_gate.py` — synthetic instances with a planted Setup effect → PASS on that cell; no effect → FAIL; thin cell and < 5 % class → refused |

Verification before the receipt: `.venv/bin/python -m pytest brain/tests`
green; one block built end to end on a two-session window and its
`paths.parquet` inspected by hand; the full run in the background with
`gate.log` cited in the receipt.

## 9. Correction of 2026-09-13, before the first fit

The smoke block the plan required (two sessions, 2022-01-03/04: 649 steps,
195 paths, `context_found` 100 %, every clock populated) showed the §5.3
target rule degenerate on the real tape. The nearest unswept ≥ 5m level in
direction is, at the median, 0.19 ATR₆₀ from the close — inside a bar's
range — so the median instance resolved in two minutes and the `censored`
class held 2 % of instances (K0 2 %, K1 1 %, K2 1 %). Under §6 every cell
would have been refused; the gate could not have returned an answer by
construction. No model had been fitted when this was found.

Two rules change, both fixed here before any fit:

1. **Target = the first unswept level in direction at least 1R away**
   (`d_t ≥ d_f`, the failure distance), over the same 5m ∪ 15m ∪ 1h union,
   nearest such level wins. A level inside 1R is not a draw a Setup is
   traded to. On the smoke block this leaves 374 of 497 labelled instances,
   median resolution 3 minutes, 90th percentile 48.
2. **The verdict outcome is binary**: `hit_target` = 1 when the target is
   reached before the failure boundary within the horizon, 0 otherwise
   (failure first, or censored). This is P(Target < Invalidation) as the
   Brain would use it; a censored instance is a claim that did not pay.
   The three-way label of §5.3 stays in the descriptives with its
   time-to-resolve and MAE distributions. The §6 class guard applies to the
   two outcomes. On the smoke block, `hit_target` = 1 on 21 % of instances
   (K0 17 %, K1 23 %, K2 25 %).

Everything else — clocks, horizon, same-bar rule, feature sets, folds,
bootstrap, Holm, the 200-row floor — is unchanged. The change was chosen on
label shares of two sessions, not on any model's out-of-sample loss, and it
is recorded so it cannot be mistaken for a pre-registered choice.
