# Trading Brain

The Brain interprets the Eye's published facts. It owns no market state and no
history: it reads `MarketSnapshot` and the canonical physical interaction facts,
and it produces belief, admission and action.

**The Brain forms its hypotheses from history rather than from a named
taxonomy.** A trajectory-mode library is fitted offline from what the market
actually did in the sixty minutes after each past bar; at runtime the Brain
keeps at most three of those modes alive as competing hypotheses, plus a
residual for "none of these". The frozen six-path `PathKind` competition set
that preceded it was retired on 2026-09-09.

Three is a working-set bound, not a claim about how many futures exist. The
library may hold any number of modes, and which three are live changes minute to
minute.

## Core modules — `brain/core/`

| module | owns |
| --- | --- |
| `brain_entry_sequence.py` | Brain-side interpretation of canonical physical interaction facts |
| `hypothesis_proposer.py` | the context vector `X_t`, and k-NN retrieval of the modes that historically followed contexts like it |
| `hypothesis_pool.py` | the live working set and its five lifecycle operations |
| `belief_updater.py` | the realized path, its likelihood against a mode, and log-space normalization |
| `forecast.py` | the published surface: one `MarketBeliefState` per completed bar |
| `decision.py` → `risk.py` | the sole runtime action authority |

### Per-clock flow

```text
MarketObservation (Eye)
 └─ observation_features(snapshot)      X_t: Eye state + tape + session + cross-TF
     └─ HypothesisProposer.propose      which modes followed the nearest past contexts
         └─ HypothesisPool.advance      extend → score → retire → merge → split → spawn
             └─ HypothesisForecaster    MarketBeliefState (asof, hypotheses,
                                        residual_probability, uncertainty,
                                        revision_id)
```

### Lifecycle

| operation | when |
| --- | --- |
| `SPAWN` | a proposed mode is not live, its retrieval prior clears `spawn_minimum_prior`, and there is a free slot (or it beats the weakest live hypothesis by `spawn_displacement_margin`) |
| `UPDATE` | every clock: each live hypothesis is re-scored against its realized path |
| `SPLIT` | a hypothesis older than `split_minimum_age_bars` sits between two child modes of its own mode, within `split_maximum_imbalance`. If the pool is full it may evict the weakest rival, exactly as a spawn may — otherwise a pool that is usually full could never split |
| `MERGE` | two live hypotheses' *remaining* expected trajectories are within `merge_maximum_distance`; they become their lowest common ancestor. When one is already the other's ancestor, the descendant is absorbed into it — the finer distinction stopped paying for itself |
| `RETIRE` | age reached `retire_maximum_age_bars`, divergence exceeded `falsification_divergence`, probability fell under `retire_minimum_probability`, or a better-supported proposal displaced it |

The evidence weight is recomputed in full on every clock rather than
accumulated. Accumulating would count the same realized minute once per
subsequent bar; recomputing keeps the score a pure function of the path so far,
which is also what makes a replay reproduce every `revision_id`.

`residual_probability` is never renormalized away. It competes as an ordinary
term in the log-sum-exp, is floored at `residual_floor`, and equals one when
nothing is live — the Brain is never forced to explain the whole future with
whichever modes it happens to be holding.

`uncertainty` reads that residual correctly. The residual is "some mode I am
not naming", not one named outcome, so `belief_uncertainty` spreads it across
the slots the Brain is not using before taking the entropy. Treating it as a
single outcome would make total ignorance — an empty pool, residual one — score
zero uncertainty, the same as perfect confidence.

## Research — `brain/research/`

| module | owns |
| --- | --- |
| `trajectory_dataset.py` | driving the Eye over a window and pairing each bar with its realized sixty minutes |
| `mode_discovery.py` | Ward + K-Medoids mode fitting, the Ward hierarchy, and the algorithm comparison harness |

Both read the future by construction. Nothing here may become a runtime
authority, which is why `observation_features` lives in `brain/core/` and this
package imports it, not the other way round.

`trajectory_vector` replays the future through the same `RealizedPath` the
runtime updater uses, so a mode's medoid and a live hypothesis's partial path
can never be measured by different arithmetic.

### Why Ward, not HDBSCAN

HDBSCAN was the first choice, for good reasons: density-based, never told how
many modes exist, and free to call a trajectory *noise* rather than forcing it
into a cluster. On real data it abstains almost completely.

Over 5,813 observation points (2022-01-02 → 01-06):

| algorithm | parameter | clusters | noise | η²(r60) |
| --- | --- | --- | --- | --- |
| HDBSCAN | `min_cluster_size=15` | 2 | **98.2%** | 0.038 |
| HDBSCAN | 25 / 50 / 100 / 200 | **0** | 100% | — |

ATR-normalized sixty-minute trajectories are one continuous cloud with no
density gaps — which is what a near-continuous return distribution looks like.
This is not a tuning failure; there is no density structure to find.

So the modes are quantization bins of a continuum, and the honest way to cut a
continuum is a partitional method. On the temporally disjoint sample:

| algorithm | k | silhouette | η²(r60) | block-resample ARI |
| --- | --- | --- | --- | --- |
| **ward** | 6 | 0.175 | **0.576** | 0.487 |
| kmeans | 6 | 0.243 | 0.514 | **0.540** |
| gmm | 6 | 0.119 | 0.446 | 0.220 |

Ward and K-Means are close; the Gaussian mixture is decisively worse on
stability. Ward wins because it produces a hierarchy, and `SPLIT`/`MERGE` move
along it.

Note that η² rises monotonically with `k` while silhouette falls. The two
criteria disagree precisely because there is no natural `k`; six is where both
were still reasonable, and `--n-modes` exposes the choice.

Two sampling cautions are built into the harness. Consecutive observation
points share fifty-nine of their sixty future minutes, so metrics on the full
sample measure autocorrelation as much as structure — hence the decimated
"disjoint" sample, which for a three-day window is only ~97 points. And
stability uses contiguous-block resampling, because an i.i.d. bootstrap over
near-duplicate rows reports a stability the data does not have.

## Protocols — `brain/configs/`

`hypothesis_protocol.json` only. It declares `shadow_only` /
`development_unvalidated` / `action_authority_ready: false`, and
`load_hypothesis_protocol` fails closed if any of that changes.
`configs/model.json` records its path and its `hypothesis_protocol_fingerprint`
(`sha256` over the whole file), so a content edit must recompute that
fingerprint in the same change.

Every threshold in it is a development default chosen so all five lifecycle
operations are reachable on a short replay. None has been fitted.

## Scripts — `brain/scripts/`

| script | does |
| --- | --- |
| `build_hypothesis_modes.py` | drives the Eye once over a window, compares four clustering families, fits and writes the mode library |
| `replay_hypothesis_belief.py` | replays the forecaster over the cached window and reports the rebuild verdict |

Artifacts land under `outputs/hypothesis_modes/` (gitignored and governed):
`dataset.npz`, `mode_library.json`, `clustering_comparison.csv`,
`stability.json`.

## Tests — `brain/tests/`

`test_hypothesis_forecast.py` covers the contract, the updater, all five
lifecycle operations, retrieval and the published surface.
`test_risk.py` covers the surviving Risk surface.

## Authority documents

The Brain's current implementation-versus-plan authority is
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md);
the facts it consumes are defined in
[eyes/docs/smc_semantic_specification_v1.3.md](../../eyes/docs/smc_semantic_specification_v1.3.md).
No Brain surface carries economic or live-trading authority: `configs/model.json`
holds `release_readiness.live_execution_allowed = false`, and every
`MarketBeliefState` refuses construction if it claims otherwise.

**A fitted mode library is not evidence of predictive skill.** It records what
followed similar contexts in the fitting window and nothing more. A library fit
and replayed on the same window is circular by construction, and says only that
the machinery runs.

## Current gaps

1. `shares/core/engine.py` imports `brain.core.playbooks`,
   `brain.core.playbook_registry`, `brain.core.dol_probability` and
   `brain.core.signal_policy`, and binds `model.path_hypotheses`. All are gone.
   Until that block is rebound to `brain/core/forecast.py`, 8 test modules
   (135 tests) cannot be collected and `shares.ContinuousSMCEngine` is
   unavailable. The forecaster is deliberately usable without the Engine — it
   takes one `ForecastInput` per bar — so the rebinding is orchestration work,
   not a redesign.
2. `decision.py` reads `MarketBelief.thesis_candidates` and
   `position_management_candidates`. `MarketBeliefState` does not fill those;
   Decision needs either an adapter or a new input contract.
3. The mode library has only ever been fitted on a three-day in-sample window.
   Nothing has been fitted out-of-sample, and no threshold in
   `hypothesis_protocol.json` has been calibrated.
4. `SPLIT` and `MERGE` churn: over three sessions the replay recorded 587 splits
   and 1,644 merges against 4,132 updates. The pool consolidates and refines far
   more often than it retires, which is defensible — the context genuinely moves
   between specific and ambiguous — but the rates have not been tuned, and a
   quieter pool may well be preferable.
5. `brain/core/validation.py`, `calibration.py`, `brain_calibration.py` and
   `calibration_replay.py` were removed on 2026-09-08. Four scripts still import
   `brain.core.validation`
   (`eyes/scripts/run_eye_authority_scan.py`, `eyes/scripts/scan_mature_ranges.py`,
   `shares/scripts/audit_market_clock.py`,
   `execution/scripts/materialize_mbo_execution.py`) and cannot run.
