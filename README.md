# SMC Continuous Trader

Version: **2.1.0-managed-policy.2**

This repository now implements an event-driven SMC decision process rather
than optimizing static labels on a frozen candidate table.

```text
completed 1m bar
    ↓
causal 4H / 1H / 5m / 1m observer + event memory + execution reality
    ↓
playbook-specific beliefs and continuous phase machines
    ↓
enter / wait / hold / protect / exit / abstain utility comparison
    ↓
independent structural-risk, liquidity-target, cost and data vetoes
```

The observer only describes the market. It cannot place trades. The brain
maintains a belief for three preregistered playbooks in both directions. The
decision layer compares net utilities, and acts only when the best action has a
clear advantage. The risk layer can veto any optimistic model output.

Current status is research-only. The 2022 belief calibration improves the
registered target-before-invalidation proposition, while the separate 2023
managed-policy calibration fails closed for two playbooks and exposes only a
small gross-value map for liquidity-sweep reversal. In June-July 2024
reconstructed-MBO diagnostics, the behavior policy proposed 1,676 entries, but
the managed layer approved none: every supported LSR value was negative after
observed costs, and DFP/FAVR had no usable managed-value discrimination. The
eyes→brain→decision link and real MBO observation path are active, but the
entry→risk→fill→position-management chain is not empirically validated because
no v2.1 entry reached it. This is underfit/overconstrained rather than evidence
of profitability, and the model is not ready for capital. See
[`reports/validation_2026-07-25/v2_1_june_july_mbo_diagnostic.md`](reports/validation_2026-07-25/v2_1_june_july_mbo_diagnostic.md)
for the evidence and claim boundary.

The complete earlier research lineage is preserved under
[`archive/versions/1.0.0`](archive/versions/1.0.0/README.md). Its original
experiment identifiers remain intact, but none of its fixed candidate tables
are an input to v2.

## Data

All source data remains under `data/`:

- NQ OHLCV-1m raw sources cover 2017-2026.
- The strict previous-session continuous-front parquet covers
  2017-01-03 through 2026-07-13.
- June-July 2024 MBO parquet is materialized.
- August-December 2024 MBO DBN files are physically present but remain behind
  `.HOLDOUT_SEALED`; loaders reject them before content access unless the final
  reveal is explicitly acknowledged.

## Runtime

The production API is `smc_trader.engine.ContinuousSMCEngine.on_bar`. It accepts
one newly completed 1m bar and returns a complete, auditable decision snapshot.

The historical replay CLI is:

```bash
python3 scripts/run_continuous_replay.py \
  --source data/processed/nq_1m_previous_session_front_2017_2026.parquet \
  --start 2025-01-02 --end 2025-01-03 \
  --simulate-execution \
  --output outputs/v2_replay
```

Replay materializes observations, beliefs, decisions, vetoes, and separated
decision/reveal charts. With `--simulate-execution`, an approved entry is first
eligible on the following bar and the resulting position is fed back to the
brain for hold/protect/exit decisions. The trade ledger is evidence only after
a separate validation protocol; running the CLI is not itself a profitability
claim.

Use this general runner for bounded research/visual diagnostics only. Its
retired `--gross-policy-calibration` flag fails before source or output access.

The registered 2023 managed-policy calibration uses a separate bounded-memory
runner:

```bash
python3 scripts/run_managed_policy_calibration.py \
  --source data/processed/nq_1m_previous_session_front_pre_holdout_2017_20260331.parquet \
  --output outputs/v2_1_managed_policy_calibration_2023

python3 scripts/run_managed_net_calibration.py ...
python3 scripts/run_action_clock_calibration.py ...
```

Managed-policy, managed-net, and action-clock are the canonical annual runners.

Decision rows are atomically streamed to bounded Parquet shards. Immutable
content-addressed checkpoints include the causal reader, observer, brain,
portfolio, pending/open execution state, counters, last committed source row,
and shard inventory. With `--resume`, the same binding-identical command resumes an incomplete run,
verifies all committed hashes, reports completion/throughput/ETA, and writes
`COMPLETED.json` only after every final artifact is committed. It does not
calculate a full serialized snapshot hash every minute; visual/path-audit
replays retain that stronger per-decision hash.

Resume is explicit: repeat the exact command with `--resume`. Source, window,
warm-up, config, code, protocol, execution input, and storage parameters are
hash-bound. If any binding changes, start a fresh output rather than reusing
stale checkpoint state.

For roll-sensitive causal research, first build the previous-session front:

```bash
python3 scripts/prepare_causal_front.py \
  --source data/raw/nq_ohlcv_1m/glbx-mdp3-20170101-20211231.ohlcv-1m.dbn.zst \
  --source data/raw/nq_ohlcv_1m/glbx-mdp3-20220101-20251231.ohlcv-1m.csv \
  --source data/raw/nq_ohlcv_1m/glbx-mdp3-20260101-20260714.ohlcv-1m.dbn.zst
```

Selection for each Globex session uses only the highest-volume outright from
the strictly prior completed session. The first source session is omitted and
there is no current-session fallback.

See [`docs/architecture.md`](docs/architecture.md) for the contracts and
[`docs/playbook_preregistration.md`](docs/playbook_preregistration.md) for
playbook admission rules. See
[`docs/self_review_checklist.md`](docs/self_review_checklist.md) for the
mandatory pre-test review gate.
