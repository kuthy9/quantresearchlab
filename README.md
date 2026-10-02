# SMC Trader

A causal, multi-timeframe research system for intraday futures (NQ on CME,
1-minute bars). A deterministic **Trading Eye** turns raw bars into Smart Money
Concepts (SMC) facts. An **LLM Brain** reasons over those facts, a **Risk gate**
sizes or vetoes its plans, and an **order machine** acts on them through either
a simulated broker or an IBKR *paper* account.

> **Research status.** This repository is for research only. It makes no claim
> of profitability and gives no trading advice. Live execution is disabled
> (`configs/model.json`: `live_execution_allowed: false`), and the IBKR adapter
> refuses any account that is not a paper (`DU…`) account. The latest
> pre-registered study of the Eye's structural events on NQ 2023 found no
> candidate edge
> ([eyes/docs/evidence/2026-09-23_event_edge_2023.md](eyes/docs/evidence/2026-09-23_event_edge_2023.md)).

---

## The Eye: an encapsulated market perception layer

The Eye is the only component that reads market data. Everything downstream
sees the market through what the Eye publishes. It is built as a sealed unit
with one entry, one history authority and one output:

```text
Bar (1m OHLCV)
 └─ CausalMarketReader         eyes/core/causal.py        normalize to the tick grid, session clock, scale registry
     └─ CausalObserver         eyes/core/observation.py   drives the six detectors on every scale
         ├─ structure.py        swings, BOS / MSS, structural legs, protected swings
         ├─ liquidity.py        levels, equal-high/low pools, sweeps, reference levels, target inventory
         ├─ displacement.py     impulsive delivery, scored per scale
         ├─ zone.py             fair value gaps and order blocks, with their lifecycles
         ├─ range_auction.py    dealing ranges, balance, manipulation
         ├─ interaction.py      how price is interacting with a zone right now (entry paths)
         ├─ semantic_event_emitter.py   the sole emitter of canonical events + ancestry
         ├─ event_store.py      EventStore: the complete atomic history
         └─ market_state.py     MarketSnapshotPublisher: reduces history to one current view
             └─ MarketObservation   contract/eye/observation.py — the Eye's sole output
```

Each primitive runs on every active scale (1m, 5m, 15m, 1H, 4H). The Eye is
built in one place, `shares/core/eye_factory.build_eye`, from
`configs/model.json`.

### What "encapsulated" means here

| property | how it is enforced |
| --- | --- |
| **One-way boundary** | No `eyes/core/` module imports `brain`, `execution`, `risk` or the orchestration half of `shares`. `eyes/tests/test_eye_module_boundary.py` checks every import. |
| **Strict causality** | A component sees time only through the bar's `known_at`. Every fact carries both `formed_at` (when it formed) and `known_at` (when it became knowable), so confirmation lag is explicit. Contracts reject any clock later than the payload's `asof`, and `shares/tests/test_no_wall_clock.py` keeps every `*/core` module off the wall clock. |
| **Single authority** | `EventStore` owns the complete history and `MarketSnapshot` owns the current view. No second history, state or lifecycle authority may exist. |
| **Sealed definitions** | The semantic registry (`semantics/`) and the eight protocol files in `configs/` are hashed into `atomic_definition_identity` (pinned in `configs/model.json`). Changing a definition changes the identity, so silent drift is not possible. |
| **Fail-closed contracts** | Every payload is a frozen dataclass that validates itself at construction. Prices are integer ticks, timestamps must be timezone-aware, and the provenance namespaces (event ancestry, raw data ids, entity ids) are kept separate. |
| **Two clocks of events** | `semantic_events_this_update` carries the ≥5m scales; the 1m tape travels separately as `microstructure_events_this_update`. Nothing is dropped; the two channels partition the update. |
| **Bounded memory** | Trackers retain only live state. Cold events spill to an append-only journal and remain provable by digest. |

### How the Brain sees the Eye

`brain/core/eye_view.py` turns each `MarketObservation` into an `EyeContext`:

- Every Eye object gets an **alias**, such as `FVG_5m_3`. The alias is stable
  within an episode and maps back to the Eye's entity id
  (`object_registry.py`).
- Each object is placed relative to price (`above_price` / `below_price` /
  `contains_price`), with a signed distance in ATRs.
- The context also carries per-scale structure, delivery phase, range and
  liquidity summaries, plus this bar's evidence events.

The LLM **names objects and never writes a price**. Code turns the objects it
names into an entry, stop and target (`brain/core/opportunity_geometry.py`),
and the reply parser refuses any unknown alias or price field.

---

## Workflow: one completed 1-minute bar

```text
1m bar
 └─ Eye ─────────────► MarketObservation
     └─ BrainRuntime (SLEEP ↔ ACTIVE)                      brain/core/runtime.py
         ├─ Sleep Controller → WAKE / STAY_ASLEEP / UPDATE / TICK / EVENT_SLEEP
         └─ on WAKE / UPDATE:
             EyeContext → LLMInput → DeepSeek → LLMUpdate  brain/core/main_brain.py
             └─ reducer: BrainState_t + evidence + update → BrainState_t+1
     └─ TradingStack                                        execution/core/stack.py
         ├─ ACTIONABLE opportunity → TradePlan              execution/core/plan.py
         ├─ RiskGate: size or veto against the account      risk/core/gate.py
         └─ OrderMachine → Broker                           execution/core/order_fsm.py
              ├─ SimulatedExecutor  (replay, tests)
              └─ IBKRBroker         (paper account only)
 └─ hash-chained JSONL journal ─► replay_journal.py / summarize_run.py
```

1. **Perceive.** The Eye ingests the bar, updates every detector on every
   scale, emits canonical events and publishes a `MarketObservation`.
2. **Decide whether to think.** The Sleep Controller reads only the Eye's
   *transition* events. A structural reaction on 15m/1H/4H (or MSS, BOS, sweep
   or displacement on 5m) wakes the Brain. While active, a reaction or a
   watched object crossing price triggers an `UPDATE`; otherwise the bar is a
   `TICK` with no LLM call. Scheduled releases (CPI, NFP, FOMC) put the Brain
   to sleep through a fixed window.
3. **Reason.** The Main Brain sends an `LLMInput` (the aliased Eye view, prior
   state, new evidence and the executor's view) to DeepSeek. The prompt holds a
   14-step framework (`brain/configs/prompts/main_brain_system.md`).
   `parse_update` accepts only a strict JSON reply. The pure **reducer** applies
   deterministic rules — bias scale, thesis scale, entry side, bias decay — and
   either accepts or rejects the update with recorded reasons.
4. **Plan.** An `ACTIONABLE` opportunity becomes a `TradePlan`: three object
   aliases, their Eye entity ids, and the geometry resolved on the current bar.
5. **Gate.** The Risk gate checks reward-to-risk, the risk fraction per thesis
   grade, exposure, leverage, staleness, the daily stop and the drawdown halt.
   It returns a quantity or named `VetoCode`s. It never moves a price.
6. **Act.** The order machine places at most one bracket per intent and
   handles expiry, cancel/replace, invalidation and structural-reversal exits,
   and event-sleep flattening. A `ThesisBook` limits how often one thesis may
   be expressed.
7. **Record.** Every step goes to a hash-chained journal.
   `brain/scripts/replay_journal.py` proves that a run reproduces from the Eye
   alone; `brain/scripts/summarize_run.py` writes calls, tokens, cost, vetoes,
   the order lifecycle and the account into one `summary.json`.

Every payload that crosses a subsystem boundary is defined in `contract/`,
which is strictly layered:
`market → execution → eye → brain → decision → risk → research`.

---

## Repository layout

| path | owns |
| --- | --- |
| `eyes/` | the Trading Eye: normalization, the six detectors, event emission, the event store, market-state reduction |
| `brain/` | the LLM Brain: Eye view, Sleep Controller, Main Brain, reducer, journal, runtime |
| `risk/` | the Risk gate and `risk/configs/risk.json` |
| `execution/` | the order machine, the `Broker` protocol, the simulated executor and the IBKR paper adapter |
| `shares/` | data access, session clock, scale registry, the Eye factory, study projections |
| `contract/` | every cross-boundary payload, one package per boundary |
| `configs/`, `semantics/` | the sealed protocol files and semantic registry (hashed into the Eye's identity — do not move) |

Each subsystem has its own `core/`, `tests/`, `configs/`, `docs/` and
`scripts/`. The full engineering guide is [AGENTS.md](AGENTS.md); each
subsystem's design notes and evidence are in its `docs/` directory.

---

## Getting started

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

Run the daily test suite. The minutes-long real-tape tests are excluded by
default and run with `-m research_orchestration`.

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

**Market data is not included.** `data/` is gitignored. The runners expect a
1-minute NQ front-month tape at
`data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`, or pass
`--source`.

**API key.** The Brain reads `DEEPSEEK_API_KEY` from the environment, or falls
back to the gitignored file `brain/configs/deepseek.key`. Never commit a key.

A smoke run needs no LLM: `--client echo` answers every call with a
contract-valid reply.

```bash
.venv/bin/python -m brain.scripts.run_llm_brain --client echo \
    --warmup-start 2021-12-30 --emit-start 2022-01-04 --end "2022-01-04 12:00"
```

A DeepSeek run with the Risk gate and the simulated executor:

```bash
.venv/bin/python -m brain.scripts.run_llm_brain --client deepseek --broker sim \
    --warmup-start 2021-12-30 --emit-start 2022-01-04 --end "2022-01-04 12:00" \
    --max-llm-calls 60
```

Journals are written under `outputs/brain_journal/`. `--broker ibkr` targets
a TWS / IB Gateway paper session (`execution/configs/ibkr.json`).
`.venv/bin/python -m execution.scripts.ibkr_paper_check` is a read-only check
of that session and never places an order.

---

## License

[Apache License 2.0](LICENSE)
