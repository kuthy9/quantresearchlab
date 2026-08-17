# Causal multi-timeframe market representation v1

This is an offline, outcome-blind learning component. It does not add a replay
loop and it has no write path into Eye, playbooks, Brain, Decision, Risk, or
execution. The current primary input is one sparse neutral `MarketEpisode`
revision. The older `EntryEpisode` case-library path remains compatibility-only
until its consumers are migrated.

## Neutral MarketEpisode input-only path

`market_case_input_to_representation_mapping()` and
`representation_case_from_market_case_input_row()` consume the 18-field
`market_case_input_shards` schema plus its bound run manifest. Source path,
source hash, symbol, instrument, tick size and config identity come only from
the run manifest; they are not repeated in each row. This path does not read a
case library, Shadow artifact, outcome, PnL, Decision, Risk or playbook label.

The one overloaded Group-5 MicroBOS field named `outcome` is not an economic
result. After the market-row validator proves its exact collection path and
four-value enum, the adapter deep-copies it into a
`micro_bos_reference_alignment:*` relation token and removes the overloaded
key. Every other nested outcome/future key remains forbidden.

`build_neutral_market_revision_targets()` groups only by
`(market_epoch_id, market_episode_id)`. It enables two targets supported by the
sparse input stream:

- next sparse lifecycle of the same physical episode;
- contemporaneous active cross-scale direction alignment.

The legacy next-event-time, displacement and draw targets, and the generic
next-event-family target, are explicitly `-100` and receive zero supervision.
Direct scale-relation tokens are marked for the shortcut-mask probe.

The trainer's neutral branch remains deliberately fit-free. Its only required
identities are the input-shard manifest and its bound run manifest. It derives
their content identities directly, verifies source/config identity once per
load, and verifies shard hashes/rows and the exact 18-column Arrow schema. It
does not require caller-supplied duplicate manifest hashes, an outcome manifest
or a causal-case library. A run manifest may carry the newer exact
`repository: {commit: <40-lowercase-hex>}` identity; older immutable smoke runs
without that field remain readable, and the repository value is never
tokenized.

Audit mode validates all rows, builds the neutral targets and reports coverage.
Single-batch smoke additionally reconstructs the canonical five completed-bar
views from the run-bound 1m source, resolves every prefix, collates one batch,
and executes one evaluation-mode forward/loss pass. It creates no optimizer,
performs no backward pass and writes no checkpoint or embedding/head artifact.
Both modes reject fit/export options before loading the dataset. A future fit
requires a new preregistered time-block profile with purge/embargo boundaries
and a separately materialized training corpus.

`MarketEpisodeCaseIndex` is a thin, outcome-free retrieval facade over the
shared cosine-distance, local-density and OOD routing core. It admits only the
first online occurrence of an explicitly selected physical milestone, requires
strictly prior neighbours from the same epoch and a different MarketEpisode,
and accepts ensemble disagreement only from the active `next_lifecycle` and
`scale_direction_alignment` heads. Missing independent ensemble members routes
to abstention; no historical outcome distribution is joined.

`encode_market_episode_records()` is the neutral artifact handoff. It preserves
the canonical MarketEpisode, location, path, full same-clock transition-kind
set, causal clocks and encoder content identity. Its companion
`encode_market_episode_active_head_records()` exports exactly the two active
neutral heads; the four disabled legacy heads are absent rather than emitted as
untrained probabilities. These APIs are ready for a future eligible fit, but
the current audit-only profile produces no embeddings or trained retrieval
index.

## Input contract

For the legacy compatibility path, `case_input_to_representation_mapping()` is
the supported adapter for a materialized causal-case input row.
`representation_case_from_case_input_row()` uses that adapter directly;
callers do not rename recorder fields. Recursive guards reject future/outcome
fields anywhere in the mapping before event tokenization.

OHLCV is never copied into a case. Every scale carries source identity and time
bounds. Recorder `frame_row_*` values are snapshot-local observer counters and
are never used as external-table coordinates. The representation store resolves
external rows from timezone-aware `start_at/end_at` against an explicitly
declared bar-completion clock.

Every external five-timeframe view is bound by
`MARKET_EPOCH_ID:TIMEFRAME:case.source_sha256` and a pre-registered,
content-hashed lineage
manifest. The trainer first verifies the manifest SHA and original
`case.source_path` bytes, then matches epoch, parent source SHA/path,
symbol/instrument, timeframe, aggregation protocol/version, availability
semantics, view path and view SHA. It hashes every view before loading it,
rejects generic/unqualified bindings, and fails closed when a wrong-but-past
file is paired with the right case.

Real training accepts only strict causal-case input rows from committed Parquet
shards. It verifies the complete input-stream manifest, exact field schema,
shard order/row counts/content hashes and bound run-manifest hash. It also
requires the separately pre-registered final `causal_case_library.manifest.json`;
this proves library finalization and cross-row validation completed before
training. Generic mappings/JSONL cannot bypass the recorder contract. Outcome
shards are never opened by the trainer. It verifies only the bound outcome
stream manifest's hash, complete status, exact outcome schema, row count and
same run binding, plus contiguous shard metadata whose row total is conserved,
so a forged/partial pair cannot masquerade as finalized.

The store deliberately refuses to guess whether a DataFrame index is bar-open
or bar-close time. A caller must provide either:

- a timezone-aware availability/completion column; or
- `__index_is_completed_bar_end__`, only for a view whose index is certified to
  be completed-bar time.

Canonical multi-timeframe views must be built by the existing causal aggregation
semantics. An ordinary Pandas resample is not an accepted substitute. ATR and
relative-volume denominators are shifted, rolling values from strictly earlier
completed candles. Current absolute OHLC prices never enter the returned feature
matrix.

The candle feature schema contains returns, ATR-normalized gaps/body/range and
distances, tick body/range, body/wick/range ratios, close position, relative
volume, time delta, and baseline-validity flags. Event input contains stable
typed-event, lifecycle, graph-relation and scale tokens plus log age, log
duration, relation count and direction. IDs identify case rows but are not
embedded as semantic tokens.
Changed Scene Graph relations are not reduced to counts: complete
`relation_descriptors` contribute relation kind, lifecycle and typed
source/target kind, role and scale. Edge/node IDs remain audit identities only;
an incomplete descriptor delta fails closed.

`brain_response` remains in the case artifact for audit but is never tokenized.
Decision/Risk/action/hard-gate fields, playbook/regime/mechanism labels and
absolute timestamp strings are recursively excluded from market tokens. Time
enters only as causal age, duration or delta features. Contrastive authority
uses the contemporaneous `authority_json.authority_direction`; a missing value
is unknown (`0`) and is never copied from thesis direction.

## Model

The optional PyTorch model has five independent two-layer GRU encoders:

```text
4H GRU  ─┐
1H GRU  ─┤
15m GRU ─┼─ fusion MLP ─ L2-normalized 128d embedding
5m GRU  ─┤
1m GRU  ─┤
event/graph GRU ─┘
```

Construction enforces fewer than 5,000,000 trainable parameters. PyTorch is an
optional `representation` dependency. In a replay-only environment without it,
preprocessing and leakage audits remain available, while model construction and
training raise `TorchUnavailableError`; there is no silent NumPy substitute.

## Learning objectives

All labels are separate from `RepresentationBatch` and therefore inaccessible
to `encode()`:

- masked candle reconstruction;
- masked typed-event reconstruction;
- next typed market-event family (fixed vocabulary, including none/ambiguous);
- next typed lifecycle transition (fixed vocabulary, including none/ambiguous);
- next typed-event time bucket;
- displacement continues versus exhausts;
- draw consumed within the fixed observable horizon;
- contemporaneous cross-scale direction alignment;
- contrastive pull for different episodes under the same Context;
- contrastive separation across market epochs or opposite authority.

`build_observable_revision_targets()` derives labels only from later input
revisions of the same `(market_epoch_id, entry_episode_id)`. The recorder uses
the existing replay to accumulate every observable transition between sparse
case revisions. Next-event, next-lifecycle and next-time labels are supervised
only when `observation_transition.coverage` is complete and exactly spans the
two revision clocks and its replay-update ordinals are contiguous; otherwise
all three are `-100` (ignored), never a guessed
`none`. The earliest typed/scene update supplies the label and simultaneous
different types become `ambiguous`. The builder records the latest label clock
and never reads the separate Shadow outcome stream. Final or right-censored
revisions retain `-100` for unavailable targets. Masked reconstruction and
contrastive masks are produced only during batch collation.

Displacement continuation/exhaustion is supervised only by an exact typed
lifecycle/invalidated transition for the frozen displacement ID in complete
coverage; an EntryEpisode/zone terminal is not a proxy. Draw consumption is
likewise bound to the exact frozen draw ID and requires complete coverage
through the consumption clock or fixed horizon. A supplied external target file
is only a cache: every identity, input/label clock, allowed label source and
class must exactly equal the internal observable builder.

## Splits and duplicate control

`assign_leakage_safe_splits()` computes connected components over both
`entry_episode_id` and the future-blind causal-input fingerprint. Consequently:

- one episode cannot cross train/validation/test;
- an identical decision-time input cannot cross splits under another episode;
- epoch resets remain distinct;
- future results cannot manufacture duplicate model inputs.

## Evaluation and retrieval handoff

`evaluate_outcome_blind_embedding_space()` reports:

- nearest-centroid separation of continuation, sweep failure, balance and
  unknown;
- the majority baseline;
- direction/regime group counts and collapse diagnostics;
- independent-episode, cross-date same-mechanism retrieval@K and precision@K.

Completion requires the preregistered minimum count for both long and short in
each of continuation, sweep failure, balance and unknown, plus non-collapse in
every group in both reference/train and held-out validation queries. Regime
accuracy must clear the majority baseline by a fixed margin;
cross-date retrieval must clear both a fixed hit threshold and its label-chance
rate by a fixed lift. All six registered classification heads must have labels,
model metrics and majority baselines; none can disappear through a shared-key
intersection. Validation-task NLL must clear fixed per-task and mean relative
improvement thresholds. Candle-zero and event-uniform reconstruction baselines
are separately gated by a fixed relative improvement. These are immutable v1
development defaults, not thresholds to tune after seeing a run.

The quality probe recomputes embeddings after removing direct label-source
tokens (notably case scale-relation tokens). `criteria_met` is forcibly false
for an unmasked probe even if raw geometry looks separable. It reports
`insufficient_evidence` when any required group or cross-date neighbour set is
too small or collapsed. It never turns a synthetic smoke result into a
validation pass and never claims trading edge.

For retrieval export, first use `select_first_causal_stage_revisions()`. It
chooses the minimum online `revision_index` for an explicitly requested stage,
which is order-invariant and cannot prefer a later/deeper or successful path.
`encode_decision_time_records()` then requires one row per episode and a
homogeneous explicit stage. Exported records include `revision_stage`,
`revision_index`, `stage_identity`, `stage_occurrence=0`, decision/feature
clocks, split role, mechanism/regime and `outcome_fields_used=false`.
Every vector also carries `embedding_checkpoint_id`, the content SHA-256 of the
exact encoder weights. `validate_single_embedding_checkpoint()` rejects an
index that mixes vectors from different encoders; model version alone is not a
sufficient vector-space identity.

Embedding/head export accepts only a batch collated with
`mask_probability=0.0`; stochastic candle or event masks are rejected. Export
runs under `eval()`/`no_grad()` and restores the prior training mode. Records
bind `inference_unmasked_v1`, the exact checkpoint, case/revision/episode and
decision/feature clocks. The training CLI writes atomic JSONL plus a sidecar
manifest containing the artifact hash, schema, explicit stage, checkpoint IDs,
fixed six-head shapes and causal input/library manifest identities. The first
ensemble member is the single reference embedding space; all members export
independent probability heads.

For OOD work, `--ensemble-size 3` trains three independently initialized
checkpoints with distinct seed and content identities. The script exposes
`decision_time_head_probabilities(output)` for disagreement measurement; it
does not average heads into a buy/sell recommendation. Metrics include
cross-seed mean/population-standard-deviation/min/max and require every member
to meet the fixed representation gates before ensemble stability can pass.

## Commands

Install in a separate training environment:

```bash
uv sync --extra representation --extra test
```

Run a no-artifact synthetic smoke test:

```bash
.venv/bin/python scripts/train_market_representation.py \
  --synthetic-smoke --ensemble-size 3 --epochs 1
```

Audit a completed neutral input-only run:

```bash
.venv/bin/python scripts/train_market_representation.py \
  --neutral-dataset-audit-only \
  --market-case-input-manifest RUN/market_case_input_shards.manifest.json \
  --market-case-run-manifest RUN/run_manifest.json \
  --market-embedding-kind episode_created
```

Exercise the complete neutral read/feature/target/model boundary as one batch:

```bash
.venv/bin/python scripts/train_market_representation.py \
  --neutral-single-batch-smoke \
  --market-case-input-manifest RUN/market_case_input_shards.manifest.json \
  --market-case-run-manifest RUN/run_manifest.json
```

These commands validate and report; neither fits nor exports a model.
Supplying `--epochs`, `--checkpoint`, `--embedding-output` or `--head-output`
with either neutral mode fails closed.

The legacy EntryEpisode training path requires causal-case input shards plus pre-registered input,
run and finalized-library manifest hashes, canonical completed-bar views,
trusted view content hashes, tick sizes, and explicit completion-clock bindings.
Each
`--canonical-view MARKET_EPOCH_ID:TIMEFRAME:SOURCE_SHA256=PATH` must have a
matching epoch-qualified `--canonical-view-sha`; bare timeframe/source
fallbacks are rejected and reset epochs cannot share an implicit view. The
trusted JSON lineage file is supplied with
`--canonical-lineage-manifest` and its independently pre-registered digest with
`--canonical-lineage-manifest-sha`. A separate target cache may be supplied,
but cannot override internally built labels. `--embedding-output` and
`--head-output` require an explicit `--embedding-stage`; each produces an
atomic artifact and hash/identity sidecar. Checkpoint and metrics files are
written atomically only when their output arguments are provided.

Success is not defined as profitability. A release requires measured validation
loss better than registered simple baselines, non-collapsed direction/regime
groups, adequate cross-date causal-mechanism retrieval, and no leakage-audit
failure. The v1 code only provides and enforces those measurements; it does not
pre-claim that a dataset or checkpoint has passed them.
