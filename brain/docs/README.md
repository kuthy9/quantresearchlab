# Trading Brain

The Brain interprets the Eye's published facts. It owns no market state and no
history: it reads `MarketSnapshot` and the canonical physical interaction facts,
and it produces belief, admission and action.

**The Brain forms its hypotheses locally, per clock, from a conditional future
cloud.** It retrieves the historical contexts nearest to the current one, reads
what those contexts actually led to over the following sixty minutes, and
extracts the representative trajectory nodes carrying meaningful probability
mass — keeping at most three of them live, plus a residual for everything they
do not cover.

Three is a working-set bound, not a claim about how many futures exist. The
cloud is allowed to surface more than three, because a `SPLIT` is made of a node
the pool has no slot for yet: capping extraction at the number of slots would
make it unreachable by construction.

## What this replaced, and why

| retired | date | why |
| --- | --- | --- |
| the frozen six-path `PathKind` competition set | 2026-09-09 | a taxonomy the module asserted about the market rather than one the market produced |
| the global `ModeLibrary` and its Ward hierarchy | 2026-09-09 | a global fit cannot say what is likely *from here*; it can only say what is common overall |
| principal components of the **raw** trajectory curve | 2026-09-09 | PC1 absorbed 80.7% of the variance and was essentially "where did it end", so distance degenerated into a quantization of direction |

The second retirement is the substantive one. A global library answers "what
shapes exist in this market?"; the Brain needs "what follows a context like this
one?". Those are different questions, and only the second is conditional on the
present.

## Identity is path geometry, in two channels

A trajectory is the ATR-normalized cumulative return curve over the next sixty
completed minutes. How that curve is *represented* decides what counts as the
same claim, and the first attempt got it wrong in a way worth recording.

Fitting principal components directly to the raw curve put **80.7% of the
variance on one axis**. That axis was, in substance, the endpoint: a sustained
move shifts every point of a cumulative curve in the same direction, so the
leading component measured where the path finished. With `Var(PC1) ≫ Var(PC2)`,
distance is `d² ≈ (ΔPC1)² + small`, and the pool degenerates into five bins of
direction — strong down, mild down, flat, mild up, strong up. "Fell, came back,
rallied" and "rallied straight" landed in the same bin. In a market where the
endpoint is close to unpredictable, the part that was crushed is the part worth
having.

So the representation is split into two channels that answer different
questions:

| channel | dims | answers |
| --- | --- | --- |
| **Direction** | 13 | where it went and how far: `r_5`, `r_15`, `r_30`, `r_60`, the incremental `mfe_*`/`mae_*`, `time_to_mfe`, `time_to_mae`, `path_efficiency` |
| **Shape** | 5 | what the path did on the way, with the endpoint trend removed |

Shape is built by subtracting the straight line from the origin to the endpoint,

```text
shape_k = c_k - c_60 * k / 60
```

which is zero at both ends by construction, then rescaling to unit RMS —
magnitude belongs to Direction, and leaving it here would let it back in through
the side door. A path that is already straight has no shape and comes back as
zeros rather than as amplified rounding noise. The principal basis is then
fitted to those detrended shapes, not to raw curves.

The two channels are standardized separately and gain-scaled so each carries
half the representation's variance: without that, thirteen Direction axes would
outvote five Shape axes on count alone. The assembled 18-dimensional vector is
what association, matching and ambiguity are all measured in.

Realized volatility is an **attribute**, never an identity dimension. Two paths
that arrive in the same place by the same shape are the same claim regardless of
how noisily they got there.

`PathAttributes` describes a trajectory once it is already identified:

| group | fields | in Direction? |
| --- | --- | --- |
| returns | `r_5`, `r_15`, `r_30`, `r_60` | yes |
| incremental excursions | `mfe_0_15`, `mfe_15_30`, `mfe_30_60`, and the `mae_*` mirror | yes |
| path shape | `time_to_mfe`, `time_to_mae`, `path_efficiency` | yes |
| attributes only | `rv_30`, `rv_60` | **no** |

Excursions are **incremental**: `mfe_15_30` is how much further the path ran
beyond its first-fifteen-minute high, not the high over the first thirty. Nested
cumulative excursions restate one extreme three times and leave the later
windows nearly collinear with the earlier ones.

`r_1` and `r_10` are not attributes — the full curve still enters through the
Shape channel, and `r_1` was too noisy to be worth naming.

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
             └─ choose k_t from the cloud's own separation, then K-Means
                 └─ keep nodes with mass >= threshold, with their support sets
                     └─ associate with the live set (Hungarian, gated)
                         └─ MarketBeliefState: H1/H2/H3 + residual
```

### How many pieces the cloud is cut into is decided per clock

`k_t` is not a constant. Some minutes the retrieved futures agree and the cloud
is one thing; some minutes there are two dense regions; some minutes it keeps
spreading. A fixed cut manufactures structure on the quiet clocks and hides it
on the contested ones, so every `k` from two to `max_cluster_count` is scored by
mean silhouette and the best wins — but only if it clears `separation_floor`.
Below the floor the cloud is declared a single mode and `k_t = 1`.

### Identity is decided by support, not by proximity

Nodes are re-clustered from scratch every minute and carry no identity of their
own across time, so persistence has to be established somehow. Matching on
geometry alone is not enough: the same coordinates can be produced by a
completely different set of historical samples, and that is a different
assertion about the market wearing the previous one's clothes.

Every node therefore records the **row indices of the observation points that
support it**, and the lifecycle reads those sets:

| what happened | operation |
| --- | --- |
| matched within the gate, and still resting on the same support | `UPDATE` |
| matched within the gate, but the support was replaced | `RETIRE` + `SPAWN` — not an update |
| a node that inherited nothing live | `SPAWN` |
| a live claim's support **divided** between two nodes far enough apart to be separate claims | `SPLIT` |
| two live claims' supports **converged** on one node, and the information gap between them closed | `MERGE` |
| a live hypothesis matched no node, or lost the competition | `RETIRE` |

`SPLIT` and `MERGE` are exact duals. A split needs each side to take at least
`split_minimum_inheritance` of the parent's support *and* the two nodes to be
separated by more than the gate. A merge needs the absorbed claim's support to
have flowed into the surviving node *and* `information_gap` — the separation of
the two curves measured in units of their own pooled dispersion — to have fallen
below `merge_information_floor`. Neither fires on proximity alone.

Matching uses the Hungarian assignment: deterministic and globally optimal. A
greedy nearest-first pass would make the operation depend on iteration order,
and the whole point of recording `SPLIT` and `MERGE` is to tell a real change
from an artefact.

Distance is measured on the centroid **and the cluster's spread together**. Two
clouds can share a centre and be nothing alike — one a tight knot, the other a
diffuse ring — so `component_spread` enters the metric as one more coordinate
instead of being ignored. `association_max_distance_scale` is the gate,
expressed as a **fraction of the typical distance between two unrelated
futures** (`component_scale`) rather than an absolute number, so a threshold
keeps its meaning across volatility regimes.

Nodes match on the cluster **centroid** and publish the **medoid**. A medoid is a
discrete choice, so a small shift in the cloud can jump it to a different
historical curve even when the cluster barely moved; the centroid is what stays
comparable, and the medoid is what stays real.

The evidence weight is recomputed in full on every clock rather than
accumulated. Accumulating would count the same realized minute once per
subsequent bar; recomputing keeps the score a pure function of the path so far,
which is also what makes a replay reproduce every `revision_id`.

### Uncertainty has three components, kept separate

| component | measures | remedy if it is bad |
| --- | --- | --- |
| `mode_ambiguity` | how much the claims that *are* published disagree with each other | none — the future is genuinely contested |
| `representation_coverage` | how much of the local conditional cloud the working set speaks for | more slots, or a finer cut |
| `retrieval_confidence` | whether the present state has enough close historical precedent at all | none — wait for a state the Brain has seen before |

They are kept separate because they have different meanings and different
remedies. A belief can read `H1 = 0.82`, residual `0.05` — sharp, well covered —
and still rest on twelve neighbours at extreme distance. That is the case the
third number exists to surface, and neither of the other two can see it.

`mode_ambiguity` is the expected distance between two futures drawn
independently from the published claims, saturated against `component_scale`. It
is deliberately *not* a mean over pairs: normalizing by the pair weights cancels
the probabilities out entirely whenever there are two claims, which is exactly
the case the weighting exists for.

`retrieval_confidence` multiplies a count term by a proximity term rather than
averaging them, because a full complement of remote analogues is no better
supported than a handful of close ones. The proximity term reads against
`context_scale`: the median distance between two unrelated contexts in the
fitted window.

`combined` is `(ambiguity + (1 − coverage) + (1 − confidence)) / 3`, offered as
one sortable number. The three components are the authoritative reading; the
mean claims no principled aggregation.

## Research — `brain/research/`

| module | owns |
| --- | --- |
| `trajectory_dataset.py` | driving the Eye and caching each bar's **raw** sixty-minute future |
| `forecast_index.py` | the retrieval index and the globally fitted principal basis |
| `cluster_study.py` | the k sweep, cross-window centroid reproduction, algorithm comparison |
| `churn_diagnostics.py` | telling a real change of claim from clustering jitter |
| `design_study.py` | the four measurements the v3 design rests on: representation, support identity, adaptive `k`, retrieval skill |
| `event_log.py` | the Eye's transition events, one row each, and the per-block cache the gates read |
| `event_sequence.py` | the causal Δₜ encoding of the event sequence, one track per scale |
| `first_passage.py` | which ATR barrier the next sixty minutes reach first |
| `gate_family.py` | session-block bootstrap, Holm, and the family verdict every gate applies |
| `gate_models.py` | classifier fitting shared by the gates, purged by clock when rows are not minutes |
| `path_log.py` | one row per new Group-5 path step with the geometry a Setup is labelled from |
| `setup_labels.py` | which of a Setup's two levels the tape reaches first, and whether the target paid |
| `setup_features.py` | the geometry-only and geometry-plus-Setup feature sets |

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

### The four questions the design has to answer

| # | question | measured by |
| --- | --- | --- |
| 1 | does Direction + detrended Shape beat raw-curve PCA? | `study_representation.py` — prototype geometry, out-of-sample stability, leading-hypothesis flip rate, cluster jitter |
| 2 | when a hypothesis survives a clock, is it the same hypothesis? | `study_lifecycle.py` — weighted support overlap between consecutive clocks |
| 3 | does the local cut need to adapt? | `study_lifecycle.py` — the `k_t` distribution against every fixed alternative |
| 4 | **does retrieval have any out-of-sample skill at all?** | `study_representation.py` — the conditional cloud against a random cloud and against climatology, paired per clock |

Question four decides whether the rest matters. If the cloud retrieved by `X_t`
is no closer to what actually happened than a random slice of history, then no
amount of well-behaved hypothesis lifecycle has trading meaning, and the honest
response is to say so rather than to admire the machinery.

### Choosing the resolution

There is no natural `k`: silhouette falls with it while explained variance
rises, and the two never agree. `study_cluster_selection.py` therefore reports
four things per (algorithm, `k`) and leaves the trade-off visible —
`silhouette`, `eta2_r60`, block-resample `stability_ari`, and **centroid
reproduction on a window the basis was not fitted on**.

Reproduction is the criterion that matters most. A resolution whose
representative shapes do not reappear out of sample is describing one window's
noise, however tidy its silhouette.

### What the v2 calibration run found, and what it changed

Measured on the fit window 2022-01-03/04/05 against the holdout 2022-01-07/10/11,
under the retired raw-curve representation:

| finding | number | consequence |
| --- | --- | --- |
| PC1 alone explains | 80.7% of curve variance | the representation was split into Direction and Shape |
| associating on the medoid's coordinates | median consecutive-clock distance 3.37 | nodes now match on the centroid |
| associating on the cluster **centroid** | median 1.94 | — |
| K-Means vs Ward vs GMM | K-Means wins on silhouette and stability at every k | K-Means kept as the local extractor |
| centroid reproduction out of sample | 0.50 at k=2 falling to 0.17 at k=6 | the fixed cut was replaced by a per-clock one |
| k=4 vs k=6 | stability 0.840 vs 0.545 | `max_cluster_count` set to 6 as a ceiling, not a cut |

The association gate was originally an absolute distance of 2.0 while the
coordinates scale with the window's volatility, so it admitted almost nothing
and the pool churned completely every clock — 7,012 spawns over 4,135 clocks.
Expressing it as a fraction of the representation's own spread cut that to
~2,100.

### What the v3 run found: the representation works and the retrieval does not

Fitted on 2022-01-03…14 (13,788 observation points, ten sessions) and judged on
2022-01-17…28 (13,550 points, ten sessions).

**The two-channel split did what it was designed to do.** The detrended shape
basis spreads its variance instead of collapsing:

| | raw curve PCA | Direction + Shape |
| --- | --- | --- |
| leading component's share of variance | **0.810** | **0.213** |
| spread of prototype endpoints (`direction_span`) | 10.84 | 6.57 |
| spread of prototype shapes (`shape_span`) | 11.71 | 11.67 |
| out-of-sample reproduction distance (ATR/point) | 1.145 | 1.429 |
| prototypes that reappeared out of sample | **0.000** | **0.000** |
| clustering jitter ARI | 0.995 | 0.674 |

Raw PCA's better reproduction and near-perfect jitter are not a point in its
favour. A representation that puts 81% of its variance on one axis is close to a
scalar, and a scalar is trivially stable and trivially reproducible. The
two-channel version pays for carrying real shape information with a
correspondingly harder clustering problem. What matters is the last row but one:
**under both representations, not one prototype shape reappeared out of sample
within the gate.**

**And the decisive test failed.** The conditional future cloud retrieved by
`X_t` is not closer to what actually happened than a random slice of history —
it is reliably *further*:

| pool → scored | clocks | neighbour distance | context scale | correlation | sign | optimal α |
| --- | --- | --- | --- | --- | --- | --- |
| fit → holdout | 904 | 24.86 | 8.32 | +0.012 | 49.2% | −0.119 |
| Jan 3–12 → Jan 13–14 (adjacent) | 184 | 12.12 | 8.24 | −0.020 | 54.3% | −0.064 |
| Jan 17–26 → Jan 27–28 (adjacent) | 184 | 9.35 | 8.14 | −0.057 | 50.5% | −0.066 |
| holdout → fit (reversed) | 920 | 14.20 | 8.13 | +0.015 | 52.5% | +0.010 |
| fit → fit (in-sample, **leaks**) | 920 | 7.30 | 8.32 | **+0.580** | **68.4%** | **+1.575** |

`correlation` is between the predicted and realized sixty-minute return, and
`α` is the least-squares optimal scaling of the conditional mean: at or below
zero, the best available use of the prediction is to ignore it. Against a random
cloud of the same size the conditional mean was worse by 0.280 ATR per point
(block-bootstrap interval [+0.133, +0.430]), and better on only 41.5% of clocks.

Three things follow, and none of them is a tuning problem.

1. **There is no signal, not a mis-scaled one.** RMSE alone could not have
   settled that — a prediction of zero scores well by not committing — but
   correlation, sign agreement and the optimal scaling all agree: the
   conditional mean carries magnitude (1.32 ATR RMS against climatology's 0.28)
   and that magnitude is uncorrelated with what follows.
2. **It is not the regime break.** The mid-January selloff does show up — the
   fit → holdout row retrieves neighbours at 24.86 against a context scale of
   8.32, meaning the "nearest" historical contexts are three times further away
   than two contexts picked at random. But the two adjacent trials retrieve
   genuinely close analogues (12.12 and 9.35) and have no more skill than the
   one that does not.
3. **The in-sample row is leakage, and its size measures the trap.** Adjacent
   observation points share 59 of their 60 future minutes, so retrieving from a
   pool containing a clock's own temporal neighbours retrieves the answer. The
   gap between +0.580 in-sample and ≈0.00 out-of-sample is the whole story: any
   evaluation that does not hold out by *time* will report skill that is not
   there.

The Brain is therefore machinery that runs, is bounded, deterministic and
honestly instrumented — and has no demonstrated predictive content on this
window. `retrieval_confidence` exists precisely to publish that fact per clock
rather than let it hide behind a confident-looking `H1`.

### What the v3 run found about the lifecycle itself

Replayed over the same ten holdout sessions, 13,550 clocks. These measure the
machinery, not its value: everything below describes a Brain that the previous
section showed has no predictive content.

**Support inheritance was worth building.** Of 13,302 clocks where at least one
hypothesis survived, the weighted overlap of its supporting samples was:

| | overlap |
| --- | --- |
| mean / median | 0.615 / 0.626 |
| p10 / p90 | 0.347 / 0.875 |
| **share below 0.50** | **29.4%** |
| share below 0.25 | 4.4% |

Nearly a third of the survivals rested on evidence that had been more than half
replaced, and 4.4% on evidence that was three-quarters gone. A geometry-only
rule reports every one of those as a clean `UPDATE` of the same claim. That is
the mislabelling the support test was built to catch, and it is not a rare
corner: it is a third of the working set's history.

The lifecycle over the same window:

| operation | count |
| --- | --- |
| `SPAWN` | 6,620 |
| `UPDATE` | 32,765 |
| `SPLIT` | **1,754** |
| `MERGE` | 21 |
| `RETIRE` | 8,283 |

`SPLIT` fired 1,754 times against **zero** under the v2 geometry rule — the
support-inheritance definition is one the data can actually meet. `MERGE` stays
rare because it is the strict dual: it needs the absorbed claim's support to
have flowed into the surviving node *and* the information gap between the two
claims to have closed, and both conditions holding at once is genuinely
uncommon.

**The adaptive cut helps, but less than the ceiling does.**

| `k_t` | clocks | share |
| --- | --- | --- |
| 2 | 2,318 | 17.1% |
| 3 | 288 | 2.1% |
| 4 | 496 | 3.7% |
| 5 | 2,028 | 15.0% |
| 6 | 8,420 | 62.1% |

Mean `k_t` 5.03 against a mean published working set of 2.68. On separation the
adaptive rule beat every fixed alternative — 0.2330 against 0.2267 for the best
of them — but the best fixed alternative *is* the ceiling, and 62% of clocks sat
on that ceiling, so the margin is 0.006 of silhouette and the rule is behaving
mostly as a k=6 chooser. Two honest consequences:

1. **`max_cluster_count = 6` binds.** The clouds want a finer cut than the
   ceiling allows at most clocks. Whether that is real structure or the
   silhouette's known preference for more clusters is not settled here.
2. **`k_t = 1` never fired.** Every local cloud in ten sessions cleared the 0.15
   separation floor, so the "this cloud is one mode" branch has no real-data
   exercise — it is covered by unit tests only.

**The rebuild verdict on the same window.** All four checks pass over 13,550
clocks: every clock published a well-formed state, the working set never
exceeded three, probabilities and residual summed to one to 2.2e-16, and a
second pass reproduced every `revision_id`. All five lifecycle operations fired
for the first time. Clustering jitter sits at ARI ≈ 0.74 across every operation
with an artefact suspicion of ≈ 0.09, so the churn is mostly the cloud moving
rather than the algorithm re-initializing.

The published uncertainty is worth reading against the skill result:

| | mean | min | max |
| --- | --- | --- | --- |
| `mode_ambiguity` | 0.135 | 0.000 | 0.336 |
| `representation_coverage` | 0.696 | 0.000 | 1.000 |
| `retrieval_confidence` | **0.390** | 0.121 | **0.595** |

`retrieval_confidence` never rose above 0.6 and averaged 0.39, because the
retrieved neighbours sat at a mean distance of 24.88 against a context scale of
13.71. The instrument was telling the truth the whole time: on every one of
these clocks the Brain reported that the present state had no close historical
precedent, and the skill measurement then confirmed there was nothing to be
learned from the analogues it did find. Two independent readings, one
conclusion.

### SPLIT, and why it now has a definition it can meet

Under the v2 pool, `SPLIT` never fired on real data — and widening the gate to
make it fire would have relabelled a distinct future as a refinement of an
unrelated claim. Measured over 3,192 unmatched nodes, the median distance to the
nearest live claim was 15.2 against a gate of 5.7, and of the 1.8% inside the
gate none had a matched nearest claim.

The v3 definition does not depend on that geometry at all. A split is one
claim's *support* dividing between two nodes, which is a question about which
historical samples went where; the geometric separation only confirms the two
halves are worth telling apart. Measured on the ten holdout sessions it fires
1,754 times, so the operation is now reachable by the data rather than by the
unit tests alone.

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

### What actually predicts what

`predictability_gate.py` removes the retrieval machine and asks the flat
supervised question: can Ridge or LightGBM beat the climatological mean, out of
sample, on seven targets? Nine rolling folds across 2022 — 60 training sessions,
purge, embargo, 20 holdout sessions — over 366,033 observation points, on the
`raw` OHLCV feature group.

| target | model | OOS R² | pooled CI | α | corr | passes |
| --- | --- | --- | --- | --- | --- | --- |
| `r_15` | ridge | −0.004 | [+0.018, +0.052] | −0.01 | +0.00 | no |
| `r_30` | ridge | −0.007 | [+0.070, +0.204] | −0.10 | −0.00 | no |
| `r_60` | ridge | −0.013 | [+0.303, +0.803] | −0.16 | −0.01 | no |
| `shape_pc1` | ridge | −0.002 | [+0.006, +0.099] | +0.29 | +0.02 | no |
| `shape_pc2` | ridge | −0.002 | [+0.011, +0.034] | −0.04 | −0.00 | no |
| **`mfe_60`** | **ridge** | **+0.016** | **[−0.510, −0.159]** | **0.99** | **0.14** | **yes** |
| **`mae_60`** | **ridge** | **+0.026** | **[−0.976, −0.344]** | **0.97** | **0.19** | **yes** |

A positive interval means the model is significantly **worse** than predicting
the training mean, and every directional and shape target is there. Their
per-fold correlations alternate sign and their optimal scalings swing between
+0.45 and −0.73: that is what noise looks like when it is fitted.

The two excursion targets are different in every respect. Eight of nine folds
positive, correlation positive on every single fold, and an optimal scaling of
0.97–0.99 — the prediction does not merely correlate, it is already on the right
scale and needs no shrinkage. Both clear all five conditions: beats the
baseline, pooled block-bootstrap interval excludes zero, direction consistent
across the rolling windows, survives dropping its best fold, and does not
collapse under rescaling.

**The Brain has been predicting the wrong quantity.** Where price goes is not
predictable from this data; how far it can travel is. That is the quantity
position sizing and stop placement need, and it is not what a trajectory-shape
hypothesis pool is built to express.

#### Decomposing the excursion pair, and testing it against harder baselines

`mfe_60` and `mae_60` are not independent — both grow with volatility — so the
next question is whether the predictable part is magnitude or direction. Split
them into `range_60 = U + D` (how far the path travelled in total),
`asymmetry_60 = (U − D)/(U + D)` (which side it favoured) and `time_to_touch`
(how long until price first moved one ATR either way), and add the two baselines
a magnitude signal actually has to beat:

| target | time_of_day | volatility_only | RAW ridge | LightGBM |
| --- | --- | --- | --- | --- |
| `range_60` | **0.2144** | 0.0392 | 0.0809 | 0.2033 |
| `mae_60` | 0.0825 | 0.0110 | 0.0255 | −0.0325 |
| `mfe_60` | 0.0504 | 0.0123 | 0.0156 | −0.0135 |
| `time_to_touch` | 0.0155 | 0.0047 | 0.0131 | 0.0312 |
| `asymmetry_60` | −0.0129 | −0.0025 | −0.0083 | −0.0751 |
| `r_15` / `r_30` / `r_60` | negative | negative | negative | negative |
| `shape_pc1` / `shape_pc2` | negative | negative | negative | negative |

Two things fall out, and they settle the question the retrieval study opened.

**The signal is magnitude, and only magnitude.** `range_60` — pure size, no
direction — reaches R² 0.214 with a correlation of 0.46. `asymmetry_60` — the
same two numbers rearranged to carry only direction — is negative under every
model with a correlation of 0.015. `mfe_60` and `mae_60` looked predictable
because each of them *contains* the range; separate the two and every bit of
the signal follows the magnitude and none follows the side.

**And the rich features lose to the session clock.** A time-of-day lookup table
— the training mean of the target per minute of the session — scores 0.214 on
`range_60` against 0.081 for Ridge on all twenty-four RAW features. Ridge
recovers barely a third of what knowing the time of day gives for free.
LightGBM matches the clock on `range_60` and is worse than useless everywhere
else. Against the volatility-persistence baseline the ranking is the same:
time-of-day adds 0.183, Ridge adds 0.044.

So the predictable component is the **intraday volatility seasonality** — the
open moves more than lunch — plus a little volatility persistence. That is not
an edge; it is the shape of the trading day, and every participant has it. What
matters for the Brain is the negative half: after conditioning on the clock and
on volatility, no feature set tested here says anything about *direction*.

#### The Eye's own state, alone and as an increment

The Eye run for this arm stalled at 82,747 bars under memory pressure on a
9 GB machine (the liquidity candidate set is append-only by design and the
dataset builder held every row in RAM); the 63,738 accumulated rows were
rescued from the live process by debugger injection and scored at a reduced
protocol — 30 training sessions, 10 held out, stepping by 5, two folds over
2022-01-03…2022-03-08 — with RAW re-run on the same folds so the increment is
like for like. Two folds is enough to read a sign, not to satisfy the
"consistent across windows" condition properly.

| target | time_of_day | RAW ridge | **EYE ridge** | RAW+EYE ridge | RAW lgbm | **EYE lgbm** | RAW+EYE lgbm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `range_60` | 0.071 | 0.077 | **0.099** | 0.096 | 0.139 | **0.076** | 0.100 |
| `time_to_touch` | 0.124 | 0.047 | **0.073** | 0.104 | 0.150 | **0.130** | 0.151 |
| `mae_60` | 0.014 | 0.038 | 0.017 | 0.027 | 0.046 | 0.019 | 0.013 |
| `mfe_60` | 0.025 | 0.022 | 0.010 | 0.016 | 0.045 | 0.022 | 0.014 |
| `asymmetry_60` | −0.038 | 0.001 | −0.009 | −0.008 | 0.001 | 0.001 | 0.002 |
| `r_15` / `r_30` / `r_60` | ≤ −0.02 | ≤ 0.004 | ≤ −0.006 | ≤ −0.005 | ≈ 0 | ≈ 0 | ≈ 0 |
| `shape_pc1` / `pc2` | ≤ −0.04 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

**The Eye alone predicts what RAW predicts, and nothing more.** Its 142
structural components reach 0.099 on `range_60` and 0.130 on `time_to_touch`
— the two magnitude targets — and sit within ±0.01 of zero on every
directional and shape target, exactly RAW's profile. The session block that
carries `session_elapsed_minutes` and `session_realized_volatility` is
re-encoding the clock and the volatility state; the structural blocks add
nothing on top.

**And the increment is not there.** `ΔR²(RAW+EYE − RAW)`:

| target | ridge | LightGBM |
| --- | --- | --- |
| `range_60` | +0.019 | −0.039 |
| `time_to_touch` | +0.057 | +0.001 |
| `mae_60` | −0.011 | −0.032 |
| `mfe_60` | −0.005 | −0.030 |
| direction and shape | −0.009 … +0.004 | ≈ 0 |

Opposite signs on the same target under the two models is the signature of
noise the ablation criterion was written to reject. The one increment with the
same sign under both models, `time_to_touch`, is smaller than the time-of-day
lookup table alone scores on it (0.124).

A note on method, because it changed the answer. The first pass reported EYE
R² down to −2.3, which is not "no signal" — a regularized linear model cannot
get there on its own. `RidgeCV` had selected α = 0.1 by leave-one-out, which on
minute data is no test at all, and fitted coefficients of −39.7, +27.9 and
+19.7 on the collinear `unswept_ssl`/`unswept_bsl` counts; they cancelled in
training and not out of sample. The penalty is now chosen by blocked, purged
cross-validation inside the training window, and LightGBM early-stops on a
purged tail of it. Hyperparameter selection has to honour the same leakage
rules as everything else, or it becomes the leak.

LightGBM lost to Ridge on every cell. It reached *higher* raw correlation on the
excursion targets (0.21 and 0.25) and still scored a negative R², which is
overfitting and miscalibration rather than signal.

#### The Setup gate: Group-5 paths against their own geometry

The event-sequence gate of 2026-09-12 (FAIL on all eighteen cells, twice;
`evidence/2026-09-12_information_gain_gate.md`) asked whether the Eye's
transition *statistics* change a first-passage forecast over ATR barriers.
The Setup gate asks the narrower question the Brain needs: at the moment
the Eye registers or advances one of its own Setups — a Group-5
`zone_return` or `pool_reversal` path, at the location (K0), the held
reacceptance (K1) or the aligned micro-break (K2) — does knowing the Setup
change the probability that price reaches the first unswept liquidity
level at least 1R away before it reaches the Setup's failure boundary,
beyond what the two distances, the time left in the session and the
recent volatility already imply? The baseline is deliberately geometry
only, because under a driftless walk that probability is `d_f/(d_t+d_f)`
and a model that sees the distances gets it for free.

23 warmed Globex-week blocks over 2022 H1, 38,153 path steps, 26,046
labelled instances, ten judged cells. **FAIL.** No cell's session-block
interval for Δ log-loss (M₁ − M₀) lies below zero; the best cell,
K0:zone_return under LightGBM, is −0.0024 [−0.0055, +0.0005] on a baseline
of 0.56, beating geometry on two rolling folds of five; Holm rejects
nothing. The fitted geometry model is itself within 0.01 of the
zero-parameter ratio on four cells, and on the sweep bar (K0:pool_reversal,
7,698 instances, failure boundary a median 0.05 ATR₆₀ away) neither beats
the class prior. Adding the 150 Eye-state components on top makes every
cell but one worse. What the Setups do carry is conditional shape: a target
that pays does so with a median adverse excursion of 0.02–0.13 ATR₆₀, a
failure has already run 0.11–0.48 against the claim, and only the
post-break clock lives for hours. Receipt: `evidence/2026-09-14_setup_gate.md`.

Run 2 against the repaired Eye (`main` merged 2026-09-15) has no verdict
yet: 19 of its 23 weeks build, four raise inside the Eye (a base-origin
core pruned before its order block qualifies; a 15m dealing range
confirming balance on one upper touch), and 96 sessions cannot form the
primary fold. On the 19 common weeks the repairs leave Group 5 bit-for-bit
unchanged — same 31,772 steps, same 8,878 paths, same milestones and
boundaries — and move only the target inventory through candidate
retirement; every finding of the run-1 diagnosis survives. The pass itself
is nine minutes a block instead of an hour. Status and the two
reproductions: `evidence/2026-09-15_setup_gate_run2_eye.md`.

### The Eye's throughput, and why the long windows are expensive

Driving the Eye degrades within a single run: 33.4 bars/s over the first five
hundred bars, 10.4 by two thousand, 6.8 by four thousand. Two hypotheses were
tested and both were wrong.

*It is not garbage collection.* Freezing the accumulated objects periodically
(`gc.freeze()` plus a raised gen-0 threshold) drove GC-tracked objects from
438,000 to zero and moved throughput by 6% — 11.2 bars/s against 10.6. Resident
memory is flat at 247 MB throughout, which should have ruled memory pressure out
from the start.

*Restarting the Eye per chunk does not help either.* It measured 7.7 emitted
bars/s against 10.6 for one continuous pass. The Eye needs about 3,840 bars of
warmup (sixteen completed 4H bars) but degrades within about a thousand, so the
warmup always costs more than the degradation saves.

The cause is a per-bar rebuild. `dataclasses.replace` is called **5,338 times
per bar**, and three sites account for 87% of it:

| calls/bar | site | rebuilds |
| --- | --- | --- |
| 2,208 | `market_state.py:1662` `_candidate_range_membership` | `range_role`, `normalized_location_in_range` |
| 1,852 | `market_state.py:2094` | distance-to-close |
| 560 | `market_state.py:2280` | distance-to-close |

Each one maps `replace` across the *entire* liquidity candidate collection to
refresh fields derived from the current bar, and that collection grows with
market history — so per-bar cost grows linearly and total cost quadratically.
`_liquidity_state` is called 27 times per bar and accounts for 32% of runtime on
its own; `replace` itself is 47%.

Fixing it means computing those derived fields at publication time instead of
folding them into the stored candidate on every event. That is a design change
to `eyes/core/market_state.py` and has not been made.

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
| `study_representation.py` | raw PCA against Direction + Shape, and whether retrieval beats random history |
| `study_lifecycle.py` | support inheritance across clocks, and the adaptive cut against every fixed one |
| `replay_hypothesis_belief.py` | replays the forecaster, reports the rebuild verdict and the churn diagnostics |
| `predictability_gate.py` | the flat supervised question: does anything beat the climatological mean out of sample |
| `build_gate_blocks.py` | drives the Eye over warmed Globex-week blocks; `--record-paths` adds the path log |
| `information_gain_gate.py` | the event-sequence gate on event clocks (FAIL, receipt of 2026-09-12) |
| `setup_gate.py` | the Setup first-passage gate on K0/K1/K2 × context kind against a geometry baseline |

`_windows.py` holds the shared window slicing. Fit and holdout windows are named
in exchange-local time because trading sessions are: a session runs 18:00 to
17:00 New York, so "the three sessions 2022-01-03/04/05" is the half-open range
`2022-01-02T18:00` to `2022-01-05T17:00`.

Artifacts land under `outputs/hypothesis_v3/` (gitignored and governed):
`dataset.npz`, `forecast_index.npz` plus its `.manifest.json`,
`cluster_selection.csv`, `belief_replay.parquet`, and the study CSVs.

## Tests — `brain/tests/`

`test_hypothesis_forecast.py` covers the trajectory geometry and both channels,
the contract, the three uncertainty components, the updater, all five
association outcomes including support-inheritance `SPLIT` and information-gap
`MERGE`, the retrieval and extraction path, the study surfaces and the churn
diagnostics.
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

Gap 5 is the one that matters. The others are orchestration.

1. `shares/core/engine.py` imports `brain.core.playbooks`,
   `brain.core.playbook_registry`, `brain.core.dol_probability` and
   `brain.core.signal_policy`, and binds `model.path_hypotheses`. All are gone.
   Until that block is rebound to `brain/core/forecast.py`, 8 test modules
   cannot be collected and `shares.ContinuousSMCEngine` is unavailable. The forecaster is deliberately usable without the Engine — it
   takes one `ForecastInput` per bar — so this is orchestration work, not a
   redesign. The Eye-to-forecaster half is already exercised end to end by
   `brain/tests/test_eye_to_brain_link.py` on real `MarketSnapshot`s from the
   registered Eye; what is missing is only the Engine calling it.
2. `decision.py` reads `MarketBelief.thesis_candidates` and
   `position_management_candidates`. `MarketBeliefState` does not fill those;
   Decision needs either an adapter or a new input contract.
3. No threshold in `hypothesis_protocol.json` has been calibrated. The studies
   have now measured how several of them behave — `max_cluster_count` binds on
   62% of clocks, `separation_floor` never binds at all — but nothing has been
   *set* from an out-of-sample outcome, because there is no out-of-sample
   outcome to set it from (see gap 5).
4. `brain/core/validation.py`, `calibration.py`, `brain_calibration.py` and
   `calibration_replay.py` were removed on 2026-09-08. Four scripts still import
   `brain.core.validation`
   (`eyes/scripts/run_eye_authority_scan.py`, `eyes/scripts/scan_mature_ranges.py`,
   `shares/scripts/audit_market_clock.py`,
   `execution/scripts/materialize_mbo_execution.py`) and cannot run, and three
   test modules (`shares/tests/test_v2_protocols.py`,
   `eyes/tests/test_v3_displacement_replay.py`, plus the eight blocked by gap 1)
   fail at collection for the same reason.
5. **The retrieval has no demonstrated predictive skill.** On 2022-01, the
   conditional cloud is not closer to the realized path than a random slice of
   history: correlation +0.012, sign agreement 49.2%, optimal scaling −0.119,
   and the same result holds on adjacent windows where the retrieval does find
   close analogues. Until that changes, every threshold above is a parameter of
   machinery rather than of a forecast, and the lifecycle's good behaviour is
   not evidence of anything tradeable. The most likely next thing to try is a
   much longer retrieval history: ten sessions cannot cover the context space,
   and the fit→holdout neighbour distance of 24.86 against a context scale of
   8.32 says the holdout's states had no true precedent in the pool.
6. **The Eye's events do not add information either (2026-09-12).** The
   information-gain gate of
   [2026-09-11-information-gain-gate-design.md](specs/2026-09-11-information-gain-gate-design.md)
   asked, on 110 sessions of 2022 read in warmed weekly blocks, whether the
   recent event sequence Δₜ — the 49 transition kinds as recency, count,
   direction, one ordered track per scale and the trigger set, over two
   hours — improves a sixty-minute first-passage forecast that already sees
   the raw tape and the Eye's state, on clocks where a ≥1m, ≥5m or ≥15m
   event fired. It does not: none of the eighteen pre-registered cells
   passes. The boosted class gains at most 0.3 % of its log-loss on one
   cell and not across the rolling folds; the linear class loses 1.5–2.5 %
   on every cell; the ablation ranks noise. The receipt is
   [evidence/2026-09-12_information_gain_gate.md](evidence/2026-09-12_information_gain_gate.md);
   the event-driven joint conditional model, the distribution-mode
   hypothesis lifecycle and the event-driven orchestration the brief
   proposed do not start on this result.
