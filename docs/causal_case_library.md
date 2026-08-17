# EntryEpisode causal case library

The causal case library is an optional downstream artifact of the existing
continuous replay. It does not run a second replay and has no path back into
Eye, Scene Graph, Brain, Decision, Risk, or a playbook. Enable it with
`--causal-case-library --shadow-outcomes`.

## Grain and clocks

One `case_id` identifies one exact `(market_epoch_id, context_thesis_id,
entry_episode_id)`. A case contains sparse decision-time revisions, not one row
per minute. The recorder's first observed snapshot is the admission clock; a
historical playbook `formed_at` is never promoted to an admission clock.
The exported market epoch is a deterministic namespace over canonical source
SHA-256, symbol, instrument ID, and the runtime epoch ID; the process-local raw
epoch remains visible inside the typed Context JSON but cannot collide across
different sources or contracts.

The only revision stages are:

- `context_formed`, only when the Context was actually formed at the current
  decision clock;
- `episode_created`;
- `context_changed`, keyed by a canonical material evidence signature;
- `zone_registered`;
- `first_pullback`;
- `trigger`;
- `plan_formed`;
- `terminal`.

Unchanged evidence produces no row. Set-like authority/support/opposition IDs
are canonicalized before comparison. `plan_formed` is the first non-empty plan
only. Its identity hashes stable Episode custody (setup, zone/path, entry
geometry, stop, deadline and playbook-specific LSR/range provenance). A target,
draw or route that is explicitly provisional before Risk approval, and live R
diagnostics, may be re-evaluated without creating a second revision. Once an
LSR Episode first reaches `executable`, `entered`, or `delivering`, the recorder
also checkpoint-freezes separate complete execution-plan and selected-trigger
identities at one owner clock. Every retained field must match exactly,
including after the phase moves on. A typed terminal snapshot may boundedly
omit plan and/or trigger, but never clears the frozen identities; any retained
terminal field is still compared in full. Non-terminal disappearance or any
custody mutation fails closed. `available_trigger_kinds` is an append-only
evidence-family diagnostic rather than trigger custody: later qualified
alternative families may appear without creating a new stage. Trigger
strength, ID, kind, clock, setup, zone/path, direction and source custody remain
frozen. This independently audits, but does not weaken or replace, the
playbook/model owner invariant.

The recorder still observes every replay update between two sparse rows. For
each admitted Episode it accumulates the complete outcome-blind typed
Observation transition and SceneGraphDelta ledger, then freezes and clears that
ledger at the next revision. Both JSON views carry `coverage_start_at`,
`coverage_start_exclusive`, `coverage_end_at`, replay-update ordinal bounds,
`observed_update_count`, and `complete`. The runner-owned replay-update ordinal
must advance contiguously on every call, which makes `gap_free` verifiable
across checkpoint/resume. This coordinate is deliberately separate from the
canonical source-row ordinal: synthetic no-trade updates advance the former but
do not consume the latter. Empty updates
advance the coverage clock and count without being retained; only eventful
typed/graph updates appear in the ordered `updates` array and
`eventful_update_count`. Thus a long quiet Episode uses constant ledger memory
while still proving its interval complete. If the typed observer cannot prove
an interval complete, `complete=false`; target builders must mask that interval
rather than treating the next case stage as a market event.

Admission is fail-closed. A newly visible episode carrying an earlier
first-pullback, trigger, or terminal clock is excluded and counted in
`summary.skipped_quality`. LSR additionally rejects a new Context/Episode that
claims an already formed physical zone/path or whose exact manipulation root,
displacement, direction, contract, location, and zone-return path are not owned
by that Context. This LSR check does not reject the valid DFP pattern in which a
pre-existing zone is bound later by its own causal mechanism.

A first-seen pullback, trigger, or terminal timestamp must equal the current
Episode revision clock; historical milestone backfills stop capture rather than
manufacturing hindsight rows. Contract/data resets close every old-epoch case
with one explicit terminal revision at the observable boundary. That row uses
only the last old-epoch OHLCV prefix and never indexes the new epoch.

## Separate Arrow contracts

`causal_case_input_shards.manifest.json` binds the outcome-blind input stream.
It contains the current Observation transition, SceneGraphDelta, Context,
EntryEpisode, compact Brain response, authority/scale/draw/blocker/ambiguity
views, evidence IDs, plan geometry, source/model versions and OHLCV prefix
references.

Every changed Scene edge also carries an outcome-blind relation descriptor:
relation kind, lifecycle, and each endpoint's node kind, semantic role,
timeframe, structural scale, and lifecycle. Absolute node/edge IDs remain for
custody audits, while representation code can tokenize topology rather than
learning IDs or reducing a graph delta to counts. Descriptors are frozen from
the same-asof Scene Graph revision and contain no price or outcome data.

The case projection preserves the causal order of the retained Scene updates.
Within each update, the five set-valued added/revised node, added/revised edge,
and resolution-event ID arrays are unique and lexicographically sorted. Their
aggregate arrays are the sorted union across those ordered updates. Relation
descriptors retain added-edge before revised-edge grouping and are sorted by
edge ID plus their complete canonical payload within each group. An edge cannot
be both added and revised in the same update, and descriptor edge IDs cannot be
duplicated. Aggregate descriptors concatenate those canonical per-update
groups in update order, so the same edge may correctly recur in later updates.
The case-level added/invalidated event-ID summaries are likewise set-valued,
sorted and unique. Typed transition collections remain causal sequences and
are not reordered by this rule. Consequently, an upstream set permutation has
byte-identical `scene_graph_delta_json`, `input_fingerprint`, and `revision_id`,
while a real descriptor-content change retains a different identity.

`causal_case_outcome_shards.manifest.json` binds the future-label stream. It
contains target/invalidation/deadline ordering, MFE/MAE, 0.5R/1R/2R, draw
delivery, fill/expiry and terminal resolution derived from the already frozen
Shadow output. There is exactly one outcome row per admitted case at replay
finalization. Outcome columns do not exist in the input Arrow schema.
Shadow rows join only when Context, Episode, frozen location, path, and (for
LSR) displacement/zone custody all match exactly; missing, ambiguous, or sibling
custody is counted and cannot win by outcome priority. Shadow rows belonging to
warmup/rejected/never-admitted Episodes are discarded at their causal boundary
instead of accumulating across epochs or checkpoints.

When more than one exact-custody Shadow candidate can label a case, selection
uses the highest causal event specificity, then the earliest candidate
`observed_at`, then `candidate_id`. It never compares `resolved_at`, resolution,
MFE/MAE, or any other future result. A Shadow row marked `censored` is validated
before projection: its target/invalidation flags must both be null and it cannot
claim a same-bar collision. It is then exported as `first_event=right_censored`;
the exact cause, such as `draw_consumed_before_entry`,
`invalidation_before_entry`, or `entry_target_same_bar_order_unknown`, remains
in `resolution`. This keeps an unavailable path label out of target/stop heads
without erasing why it was unavailable.

Same-bar target/invalidation collisions are always exported under the frozen
conservative policy as filled, non-censored, invalidation-first rows with
`target_first=false` and `invalidation_first=true`. Expiry is not inferred from
substrings. Only `deadline_no_delivery`, `entry_unfilled_deadline`, and
`deadline_inside_completed_bar_censored` exhaust the frozen Shadow deadline;
the last remains right-censored because its completed bar crosses the deadline.
The censored family is also explicit and closed:
`observation_boundary`, `data_gap_boundary`, `window_right_censored`,
`contract_boundary`, `deadline_inside_completed_bar_censored`,
`activation_geometry_invalid`, `invalidation_before_entry`,
`draw_consumed_before_entry`, and `entry_target_same_bar_order_unknown`.
An unknown censored resolution fails closed and requires a protocol version.
For non-censored rows, the exact Shadow resolution family is validated in both
directions against `first_event`, the nullable target/invalidation pair, fill,
expiry and collision. A target flag paired with a deadline or invalidation
resolution therefore fails closed instead of being accepted by flag priority;
unknown non-censored resolution families require a new protocol version.

`causal_case_library.manifest.json` hash-binds both stream manifests and the run
manifest while preserving their schema separation. Finalization reopens the
serialized Arrow shards and verifies their hashes, exact schemas, case
identities, sparse stage cardinality, and one-outcome-per-case coverage before
publishing this manifest.

The pair manifest cannot upgrade or launder an older run identity. Before it
is published, the finalizer parses the bound canonical `run_manifest.json` and
requires the exact current schema-7 causal identity: the complete protocol
1.6 mapping, fixed input/outcome stream names, separate future stream,
`future_visible_to_input=false`, and `output_affects_model=false`. The run must
also identify `continuous_replay`, enable `output.causal_case_library`, and
list each case stream family exactly once. Representation training independently repeats
that run-manifest hash and identity check. It reads the committed outcome
manifest only for finalization metadata and never opens outcome rows.

## OHLCV references

Bars are not copied into case rows. Every row carries the absolute canonical
source path and SHA-256 plus five timeframe prefix references.

The timezone-aware `start_at`/`end_at` boundaries are authoritative for
reloading the canonical source and resampling each timeframe. The
`replay_view_1m_row_*` values are ordinals only within the run-bound filtered
view. `frame_row_*` values are local observer counters and must never be used as
global indices into an external multi-timeframe table. The public
`case_input_to_representation_mapping()` adapter preserves this distinction.

Normalization statistics are restricted to source bars ending strictly before
the decision `asof`. A planned deadline may be after `asof`; it is a known
horizon, not an observed market clock.

## Outcome-blind regime metadata

`observable_regime` is derived only from contemporaneous typed state under the
versioned `asof_only_v1` rule:

- balanced global mode or FAVR episode: `balance`;
- LSR episode: `sweep_failure`;
- DFP episode: `continuation`;
- insufficient evidence: `unknown`.

`mechanism_label` is the typed playbook identity. Neither field uses a future
path or Shadow result.

## Leakage gates

`validate_case_input_row`, `validate_case_library_rows`, and
`validate_episode_disjoint_splits` enforce:

- observed feature/event/graph clocks do not exceed `asof`;
- named JSON clocks must parse and be timezone-aware, including allowed future
  planning deadlines;
- normalization uses a strict prior prefix;
- future outcome fields cannot enter the input schema or representation
  adapter;
- one episode-stage identity is emitted once (material Context changes use a
  distinct signature);
- case identity cannot cross a market epoch;
- one episode cannot be shared by train and validation/test splits;
- feature-identical input rows cannot be duplicated according to a later
  outcome;
- every admitted case has exactly one separate outcome row after finalization.

The materialized-library validator also checks that every outcome resolves no
earlier than its last input revision, numeric labels are finite, booleans keep
their exact nullable/non-nullable types, R-hit levels are monotonic, and
first-event/deadline/censor/expiry flags do not contradict one another. It also
requires exact stop-first collision semantics and rejects pairwise first-event
flags on unfilled or censored paths.

Checkpoint/resume pickles the recorder inside the existing replay checkpoint.
The run identity, exact Arrow schema fingerprints, shard hashes, row counts and
contiguous shard indices are all verified before resume/finalization.
Recorder schema 7 and causal-case protocol 1.6 bind the deterministic Scene
projection together with the outcome-selection and censoring rules above;
schema-6 checkpoints and run manifests fail closed rather than resuming under
changed case identities.
