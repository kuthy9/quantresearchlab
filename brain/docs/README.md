# Trading Brain

The Brain interprets the Eye's published facts. It owns no market state and no
history: it reads `MarketSnapshot` and the canonical physical interaction facts,
and it produces belief, admission and action.

**The Brain forms its hypotheses locally, per clock, from a conditional future
cloud.** It retrieves the historical contexts nearest to the current one, reads
what those contexts actually led to over the following sixty minutes, and
extracts at most three representative trajectory nodes carrying meaningful
probability mass — plus a residual for everything they do not cover.

Three is a working-set bound, not a claim about how many futures exist.

## What this replaced, and why

| retired | date | why |
| --- | --- | --- |
| the frozen six-path `PathKind` competition set | 2026-09-09 | a taxonomy the module asserted about the market rather than one the market produced |
| the global `ModeLibrary` and its Ward hierarchy | 2026-09-09 | a global fit cannot say what is likely *from here*; it can only say what is common overall |

The second retirement is the substantive one. A global library answers "what
shapes exist in this market?"; the Brain needs "what follows a context like this
one?". Those are different questions, and only the second is conditional on the
present.

## Identity is path geometry

A trajectory is the ATR-normalized cumulative return curve over the next sixty
completed minutes — `r_1 … r_60`, the whole curve. That curve, projected onto a
globally fitted principal basis (`PC1…PC5`), decides whether two futures are the
same claim.

Realized volatility is an **attribute**, never an identity dimension. Two paths
that arrive in the same place by the same shape are the same claim regardless of
how noisily they got there.

`PathAttributes` describes a trajectory once it is already identified:

| group | fields |
| --- | --- |
| returns | `r_5`, `r_15`, `r_30`, `r_60` |
| incremental excursions | `mfe_0_15`, `mfe_15_30`, `mfe_30_60`, and the `mae_*` mirror |
| path shape | `time_to_mfe`, `time_to_mae`, `path_efficiency` |
| attributes only | `rv_30`, `rv_60` |

Excursions are **incremental**: `mfe_15_30` is how much further the path ran
beyond its first-fifteen-minute high, not the high over the first thirty. Nested
cumulative excursions restate one extreme three times and leave the later
windows nearly collinear with the earlier ones.

`r_1` and `r_10` are not attributes — but no information is lost, because the
full `r_1…r_60` curve remains the identity input. They were simply too noisy to
be worth naming separately.

## Core modules — `brain/core/`

| module | owns |
| --- | --- |
| `brain_entry_sequence.py` | Brain-side interpretation of canonical physical interaction facts |
| `trajectory.py` | the curve and the path attributes, defined once for runtime and research |
| `hypothesis_proposer.py` | the context vector `X_t`, retrieval, and local extraction of the conditional cloud |
| `hypothesis_pool.py` | the working set, maintained by association |
| `belief_updater.py` | curve-versus-realized-path likelihood and log-space normalization |
| `forecast.py` | the published surface: one `MarketBeliefState` per completed bar |
| `decision.py` → `risk.py` | the sole runtime action authority |

### Per-clock flow

```text
MarketObservation (Eye)
 └─ observation_features(snapshot)     X_t: Eye state + tape + session + cross-TF
     └─ k-NN over historical contexts  the nearest 200
         └─ their realized futures     the conditional future cloud
             └─ local K-Means in PC space; keep nodes with mass >= threshold
                 └─ associate with the live set (Hungarian, gated)
                     └─ MarketBeliefState: H1/H2/H3 + residual
```

### Association is where the lifecycle comes from

Nodes are re-extracted every clock and carry no identity of their own across
time, so persistence is established by *matching* this clock's nodes against the
live hypotheses in the principal basis. The five operations fall out of that
matching rather than being separate rules:

| association outcome | operation |
| --- | --- |
| one live ↔ one node, within the gate | `UPDATE` — keeps identity, age and realized path |
| a node matched nothing live | `SPAWN` |
| a live hypothesis matched no node | `RETIRE` |
| a second node sits next to a live claim | `SPLIT` |
| two live claims collapse onto one node | `MERGE` |

Matching uses the Hungarian assignment: deterministic and globally optimal. A
greedy nearest-first pass would make the operation depend on iteration order,
and the whole point of recording `SPLIT` and `MERGE` is to tell a real change
from an artefact.

`association_max_distance` is the gate. Beyond it the geometry has moved far
enough that calling it the same claim would be a fiction.

The evidence weight is recomputed in full on every clock rather than
accumulated. Accumulating would count the same realized minute once per
subsequent bar; recomputing keeps the score a pure function of the path so far,
which is also what makes a replay reproduce every `revision_id`.

### Uncertainty has three components, kept separate

| component | measures |
| --- | --- |
| `entropy` | how evenly the probability is spread over what is named |
| `distribution_ambiguity` | how far apart the named claims are from each other |
| `coverage` | how much of the conditional cloud nothing named covers |

Three tightly agreeing hypotheses and three wildly opposed ones can carry
identical entropy and mean completely different things — which is why ambiguity
is its own number. `coverage` is read from the cloud, not from the posterior: it
is the share of what actually followed similar contexts that no live claim
speaks for.

`combined` is their mean, offered as one sortable number. The three components
are the authoritative reading; the mean claims no principled aggregation.

The residual is "some future I am not naming", not one named outcome. Treated as
a single outcome its entropy would be zero, so total ignorance — an empty pool,
residual one — would score as perfect confidence. `entropy_uncertainty` spreads
it across the unused slots first, so an empty pool scores one.

## Research — `brain/research/`

| module | owns |
| --- | --- |
| `trajectory_dataset.py` | driving the Eye and caching each bar's **raw** sixty-minute future |
| `forecast_index.py` | the retrieval index and the globally fitted principal basis |
| `cluster_study.py` | the k sweep, cross-window centroid reproduction, algorithm comparison |
| `churn_diagnostics.py` | telling a real change of claim from clustering jitter |

Everything here reads the future by construction. Nothing may become a runtime
authority, which is why `trajectory.py` and `observation_features` live in
`brain/core/` and this package imports them, not the other way round.

The dataset caches raw future bars rather than derived features. Deriving a
representation is cheap; re-driving the Eye to change it costs about an hour per
window. Every curve, attribute and principal score is computed from those bars.

### Why K-Means, and why not HDBSCAN

HDBSCAN was tried first and abstains on real data. Over 5,813 observation points
it labelled 98.2% of trajectories noise at `min_cluster_size=15` and found no
cluster at all at 25 or above: ATR-normalized sixty-minute trajectories are one
continuous cloud with no density gaps, which is what a near-continuous return
distribution looks like. That is not a tuning failure — there is no density
structure to find.

So the nodes are quantization bins of a continuum, and the honest way to cut a
continuum is a partitional method. K-Means is the runtime extractor;
`cluster_study.py` scores it against Ward and a Gaussian mixture on the same
vectors. DTW would add time-warping tolerance and is a deliberate non-goal for
now.

### Choosing the resolution

There is no natural `k`: silhouette falls with it while explained variance
rises, and the two never agree. `study_cluster_selection.py` therefore reports
four things per (algorithm, `k`) and leaves the trade-off visible —
`silhouette`, `eta2_r60`, block-resample `stability_ari`, and **centroid
reproduction on a window the basis was not fitted on**.

Reproduction is the criterion that matters most. A resolution whose
representative shapes do not reappear out of sample is describing one window's
noise, however tidy its silhouette.

### Churn: real change or jitter?

Every `SPAWN`, `SPLIT`, `MERGE` and `RETIRE` has two possible causes: the cloud
genuinely moved, or the same cloud landed in a different local optimum. A
threshold tuned without separating those is tuned against noise.

The separation is measurable. `cluster_jitter` re-clusters the *same* cloud
under different initializations — whatever changes under that is jitter, because
nothing about the market changed between the runs. `cloud_drift` measures the
other half, how much the retrieved neighbourhood itself turned over. An operation
firing on a still cloud with an unstable clustering is an artefact; one firing on
a moving cloud with a stable clustering is information.

## Protocols — `brain/configs/`

`hypothesis_protocol.json` only. It declares `shadow_only` /
`development_unvalidated` / `action_authority_ready: false`, and
`load_hypothesis_protocol` fails closed if any of that changes.
`configs/model.json` records its path and `hypothesis_protocol_fingerprint`
(`sha256` over the whole file), so a content edit must recompute that
fingerprint in the same change.

Every threshold in it is a development default. None has been fitted.

## Scripts — `brain/scripts/`

| script | does |
| --- | --- |
| `build_forecast_index.py` | drives the Eye once, caches the dataset, fits the retrieval index and principal basis |
| `study_cluster_selection.py` | sweeps k with cross-window centroid reproduction |
| `replay_hypothesis_belief.py` | replays the forecaster, reports the rebuild verdict and the churn diagnostics |

`_windows.py` holds the shared window slicing. Fit and holdout windows are named
in exchange-local time because trading sessions are: a session runs 18:00 to
17:00 New York, so "the three sessions 2022-01-03/04/05" is the half-open range
`2022-01-02T18:00` to `2022-01-05T17:00`.

Artifacts land under `outputs/hypothesis_v2/` (gitignored and governed):
`dataset.npz`, `forecast_index.npz` plus its `.manifest.json`,
`cluster_selection.csv`, `belief_replay.parquet`.

## Tests — `brain/tests/`

`test_hypothesis_forecast.py` covers the trajectory geometry, the contract, the
three uncertainty components, the updater, all five association outcomes, the
retrieval and extraction path, the study surfaces and the churn diagnostics.
`test_risk.py` covers the surviving Risk surface.

## Authority documents

The Brain's current implementation-versus-plan authority is
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md);
the facts it consumes are defined in
[eyes/docs/smc_semantic_specification_v1.3.md](../../eyes/docs/smc_semantic_specification_v1.3.md).
No Brain surface carries economic or live-trading authority: `configs/model.json`
holds `release_readiness.live_execution_allowed = false`, and every
`MarketBeliefState` refuses construction if it claims otherwise.

**A fitted index is not evidence of predictive skill.** It records what followed
similar contexts in the fitting window and nothing more. An index fitted and
replayed on the same window is circular by construction and says only that the
machinery runs.

## Current gaps

1. `shares/core/engine.py` imports `brain.core.playbooks`,
   `brain.core.playbook_registry`, `brain.core.dol_probability` and
   `brain.core.signal_policy`, and binds `model.path_hypotheses`. All are gone.
   Until that block is rebound to `brain/core/forecast.py`, 8 test modules
   (135 tests) cannot be collected and `shares.ContinuousSMCEngine` is
   unavailable. The forecaster is deliberately usable without the Engine — it
   takes one `ForecastInput` per bar — so this is orchestration work, not a
   redesign.
2. `decision.py` reads `MarketBelief.thesis_candidates` and
   `position_management_candidates`. `MarketBeliefState` does not fill those;
   Decision needs either an adapter or a new input contract.
3. No threshold in `hypothesis_protocol.json` has been calibrated, including the
   association gate and `cluster_count`. `churn_diagnostics` and
   `study_cluster_selection` exist to inform those and have not yet been used to
   set anything.
4. `brain/core/validation.py`, `calibration.py`, `brain_calibration.py` and
   `calibration_replay.py` were removed on 2026-09-08. Four scripts still import
   `brain.core.validation`
   (`eyes/scripts/run_eye_authority_scan.py`, `eyes/scripts/scan_mature_ranges.py`,
   `shares/scripts/audit_market_clock.py`,
   `execution/scripts/materialize_mbo_execution.py`) and cannot run.
