# SMC Continuous Trader

Version: **1.0.0**

This repository implements a causal, continuously updated multi-timeframe SMC
trading model. It does not predict a complete future path and then choose a
strategy. Each newly completed 1m bar advances one shared vertical chain:

```text
causal 4H / 1H / 5m / 1m observation + ordered event memory
    ↓
Temporal Market Scene Graph + typed DFP / LSR / FAVR beliefs
    ↓
enter / wait / hold / protect / exit / abstain utility comparison
    ↓
structural invalidation, liquidity target, cost and execution vetoes
    ↓
next-bar execution and position feedback
```

The eyes describe what has happened. The Brain maintains typed causal
hypotheses and continuous stages. The Decision layer acts only when one action
has a clear utility advantage, and Risk can veto any optimistic output.

This is research software. A connected software path is not evidence of market
edge. Brain calibration, rolling OOF, MBO stability and the sealed holdout are
separate later stages defined in [`configs/data_splits.json`](configs/data_splits.json).

## Current configuration

Only schema version 1 is active:

- [`configs/model.json`](configs/model.json): runtime wiring and risk/decision settings;
- [`configs/playbooks.json`](configs/playbooks.json): typed DFP, LSR and FAVR definitions;
- [`configs/primitives_structure_liquidity.json`](configs/primitives_structure_liquidity.json): candle, swing, BOS and liquidity inventory;
- [`configs/primitives_displacement.json`](configs/primitives_displacement.json): incremental displacement episodes;
- [`configs/primitives_zones.json`](configs/primitives_zones.json): FVG and order-block zones;
- [`configs/primitives_range.json`](configs/primitives_range.json): accumulation, dealing range and manipulation;
- [`configs/primitives_entry.json`](configs/primitives_entry.json): entry location, first pullback, reacceptance, micro BOS and path sequence.

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

Each Globex session uses only the highest-volume outright contract from the
strictly prior completed session. The first source session is omitted; there is
no current-session fallback.

## Runtime and replay

The runtime API is `ContinuousSMCEngine.from_config("configs/model.json")`,
followed by one `on_bar` call per newly completed 1m bar. Its fixed causal order
is reader → observer/scene graph → Brain → Decision → Risk.

A bounded development replay can be run with:

```bash
python3 scripts/run_continuous_replay.py \
  --source data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet \
  --start 2022-01-03 --end 2022-02-01 \
  --output outputs/development_replay
```

Normal replay writes light decision rows, an aggregate summary, progress,
checkpoints and resumable shards. `--brain-calibration` adds typed calibration
rows only inside the registered calibration window. `--mbo-execution` supplies
observed spread/depth/fillability to Observation, Decision and Risk;
`--simulate-execution` enables the existing next-bar position feedback path.
Pending fills remain conservatively OHLCV-bar based until the MBO queue/depth
fill simulator is completed.

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

1. Freeze or park semantic primitives without PnL-driven threshold search.
2. Establish natural mature-range/FAVR authority or keep FAVR parked.
3. Calibrate typed Brain dimensions on the calibration split.
4. Obtain natural ENTER decisions, then stream only their MBO windows through
   risk, fill and position feedback.
5. Rebuild one simple stratified trace/future-view diagnostic if the stable
   vertical chain still needs case-level investigation.
6. Run rolling OOF, MBO stability and the sealed holdout once the vertical chain
   is stable.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/playbook_preregistration.md`](docs/playbook_preregistration.md), and
[`docs/self_review_checklist.md`](docs/self_review_checklist.md).
