# Episode case retrieval and novelty/OOD

> The EntryEpisode/outcome material below documents the legacy compatibility
> path. New neutral work uses `MarketEpisodeCaseIndex`: input-stream and run
> manifest hashes plus an explicit physical milestone select the first online
> occurrence, and retrieval is limited to strictly prior, same-epoch,
> different-MarketEpisode embeddings. Its result has no outcome distribution.
> Ensemble disagreement accepts only `next_lifecycle` and
> `scale_direction_alignment`; missing independent members abstains. The
> January 2024 neutral corpus is audit-only, so no trained neutral retrieval
> index is claimed yet.

The neutral handoff is produced by `encode_market_episode_records()` and
`encode_market_episode_active_head_records()`. It retains the canonical
MarketEpisode/location/path identities and the complete same-clock material
kind set, binds the unmasked encoder checkpoint, and contains no outcome
payload. The current audit-only CLI deliberately never calls these exporters.

The retrieval layer is a read-only empirical-prior service over the unified
causal case library. It does not replay the market, alter a playbook gate, or
own a Decision, Risk, Eye, or execution action.

## Input contract

The index grain is one independent `EntryEpisode`. It consumes the mapping
returned by `DecisionTimeEmbeddingRecord.as_dict()` in
`smc_trader.market_representation`:

- a 128-dimensional, L2-normalized `decision_embedding`;
- `embedding_asof` at the exact decision clock;
- `feature_max_at <= embedding_asof`;
- `outcome_fields_used == false`;
- one lowercase `embedding_checkpoint_id` content SHA-256 for the exact
  encoder weights that define the vector space;
- `embedding_input_protocol == inference_unmasked_v1`;
- stable case, epoch, Context Thesis, EntryEpisode, direction, regime, split,
  revision-stage, and embedding-model identities.

The export sidecar also binds the finalized causal-case library manifest and
its input-stream manifest by SHA-256. All embedding artifacts used to build an
index must share that lineage. The pickle-free index checkpoint persists it;
the query embedding and ensemble-head sidecars must match it exactly.
Synthetic smoke artifacts may carry both hashes as `null`, but real and
synthetic lineages cannot be mixed.

The adapter accepts structural mappings, while importing the representation
module's fixed inference protocol and six-head schema so the handoff cannot
silently drift. A row explicitly marked with an `embedding_clock` other than
`decision_time` is ignored.
`select_first_causal_stage_revisions` in the representation module selects the
first already-observed occurrence of an explicitly requested stage; it never
selects the deepest, terminal, strongest, or best-looking later revision. The
handoff binds `revision_index`, `stage_identity`, and `stage_occurrence == 0`.
The retrieval index compares only the exact same `revision_stage`. Explicit
later occurrences are ignored; if multiple material rows are marked as the
first occurrence, the smallest online `revision_index` is retained independent
of input order. Two different identities carrying the same online
`revision_index` fail closed. This permits multiple
stage rows per episode without hindsight while every K result remains
episode-independent.

Embedding input names and optional `embedding_inputs` are scanned for future
outcome fields. Outcome/future fields, a post-decision feature clock, a zero or
non-finite vector, a changed model version, or a different encoder checkpoint
content identity are rejected. An index cannot mix encoder checkpoints.

## Leakage boundaries

Neighbour admission requires all of the following:

1. the episode belongs to an explicitly allowed prior reference split;
2. its market epoch equals the query epoch;
3. it is not the query EntryEpisode;
4. its decision timestamp is strictly earlier than the query timestamp;
5. its causally selected revision stage exactly equals the query stage.

Validation and test queries default to the `train` reference corpus and cannot
name validation/test/holdout roles as references. Repository
`brain_validation`, rolling OOF, and holdout queries default to the frozen
`calibration` corpus. Production/live callers can explicitly bind historical
or reference corpora. These defaults are confidence-data routing; they do not
change the model's registered validation protocol.

Frozen outcomes remain in the separate causal-case outcome artifact. They are
joined by `case_id` only after cosine distances and the K neighbours have been
frozen. A neighbour outcome is included only when its timezone-aware
`resolved_at` is no later than the query decision clock; later outcomes are
reported as unavailable and cannot enter the empirical prior. The input-only
index checkpoint does not persist outcomes. Changing a
target/invalidation result, MFE/MAE, R milestone, draw delivery, fill, expiry,
or terminal reason can therefore change only the aggregate historical outcome
distribution—not vectors, similarities, neighbour identities, density, or OOD.

## Result and OOD policy

The result contains:

- K nearest independent prior EntryEpisodes and cosine similarity/distance;
- neighbour month, direction, and regime distributions;
- a separate frozen-path distribution over first terminal event, resolution,
  fill/expiry, draw delivery, R milestones, terminal reason, and MFE/MAE
  summaries;
- neighbour sufficiency and local density;
- ensemble size and per-head/maximum disagreement;
- one confidence-routing policy.

The only policy values are `continue_evaluation`, `increase_uncertainty`, and
`abstain`. They never encode long/short or a trade recommendation.

The default novelty check combines mean/nearest cosine distance, density in a
fixed cosine radius, and RMS probability disagreement across independent model
checkpoints. A missing or undersized deep ensemble fails closed to `abstain`.
Every ensemble-head row must match the query's case, revision, EntryEpisode,
model version, decision clock, and feature cutoff exactly; each member must
carry a distinct checkpoint content SHA-256 and `inference_unmasked_v1`.
Members must expose exactly the six registered outcome-blind heads with their
fixed probability-vector widths; an arbitrary result/profit head is rejected.
Heads from another episode or clock are rejected rather than treated as
uncertainty evidence. A bulk head artifact is filtered to the exact query
identity only after its full content and sidecar manifest have been verified.
Sparse but nearby cases or elevated disagreement increase uncertainty. Clear
distance/density OOD or extreme ensemble disagreement abstains. Thresholds are
configured through `OODThresholds`; they do not loosen any causal hard gate.

## CLI

Build a pickle-free, hash-bound, input-only checkpoint:

```bash
.venv/bin/python scripts/query_causal_cases.py build \
  --cases decision_embeddings.jsonl \
  --case-manifest decision_embeddings.jsonl.manifest.json \
  --case-manifest-sha "$EMBEDDING_MANIFEST_SHA256" \
  --output episode_case_index.npz
```

Query one current decision-time embedding and join independent outcomes only
for the returned distribution:

```bash
.venv/bin/python scripts/query_causal_cases.py query \
  --index episode_case_index.npz \
  --query decision_embeddings.jsonl \
  --query-case-id "$CASE_ID" \
  --query-revision-id "$REVISION_ID" \
  --query-manifest decision_embeddings.jsonl.manifest.json \
  --query-manifest-sha "$EMBEDDING_MANIFEST_SHA256" \
  --ensemble ensemble_probabilities.jsonl \
  --ensemble-manifest ensemble_probabilities.jsonl.manifest.json \
  --ensemble-manifest-sha "$HEAD_MANIFEST_SHA256" \
  --outcomes causal_case_outcomes.parquet \
  --k 10
```

The CLI reads JSON, JSONL, or Parquet and never invokes a replay or engine.
Embedding and head artifacts require complete sidecar manifests plus their
independently pre-registered SHA-256. This binds artifact bytes, row count,
stage, first-occurrence selection contract, record-identity digest, encoder
checkpoints, deterministic input protocol, fixed head schema, and finalized
case-library lineage.

## Required regression checks

`tests/test_case_retrieval.py` covers future-outcome mutation invariance,
decision-revision uniqueness, representation-export compatibility,
train/validation/epoch isolation, self/future-neighbour exclusion, sparse and
high-disagreement routing, obvious OOD, the sufficient-neighbour path, outcome
aggregation, checkpoint integrity, and absence of trade-action output.
