# Causal continuous architecture

Schema: **1**

## Runtime flow

```text
newly completed 1m OHLCV bar
        ↓
causal 1m / 5m / 15m / 1H / 4H aggregation
        ↓
descriptive primitives + ordered event memory
        ↓
Temporal Market Scene Graph + GlobalMarketContext
        ↓
FocusState
        ↓
Belief(t) = update(Belief(t-1), Observation(t), SceneDelta(t))
        ↓
typed DFP / LSR / FAVR episodes and phases
        ↓
enter / wait / hold / protect / exit / abstain utility comparison
        ↓
independent structural, cost, deadline, data and fillability vetoes
        ↓
next-bar conservative execution → position feedback
```

There is one runtime path. Replay calls the same engine used for incremental
operation; it does not reproduce reader → observer → Brain → Decision → Risk in
a second implementation. Multi-timeframe bars update only when complete.

## Eyes and Scene Graph

Each completed minute updates one shared observation across five enabled
frames. A higher-timeframe frame changes only when a bar for that frame has
completed:

| Frame | Descriptive state |
|---|---|
| 4H | confirmed structure/BOS and external liquidity; directional displacement, efficiency and range position remain explicitly descriptive proxies |
| 1H | confirmed swings/BOS, support/resistance and typed dealing range; rolling acceptance/rejection remains an explicitly descriptive proxy |
| 15m | bridge-scale candle, structure and liquidity context |
| 5m | displacement episode, raw/displacement-linked FVG, qualified order block; rolling compression remains an explicitly descriptive proxy |
| 1m | candle description, manipulation resolution, exact first return, qualified entry-zone reacceptance, micro BOS and ordered path steps |

Every semantic event carries identity, formation/confirmation/invalidation
clocks, lifecycle, direction, strength and source IDs. Event memory retains
order and duration, not just current scores. In particular:

- first pullback is the first return to one frozen qualified FVG/OB entry
  zone; a mature range is context, not the entry zone itself;
- qualified entry-zone reacceptance requires departure, reclaim, hold and
  explicit failure; manipulation reacceptance remains the separate Group4
  multi-bar lifecycle;
- path sequence is ordered event identity, not swing progression;
- value comes from a mature dealing range, not 4H range position;
- planned entry is a frozen zone price and need not equal the current close.

The scene graph links events, zones, draws, invalidations and competing
interpretations. `GlobalMarketContext` incrementally summarizes structural
authority, cross-scale relations, external draw candidates, path blockers and
identity-bound conflicts from the current graph delta. It does not choose a
draw or action. Focus identifies what the Brain should inspect next while
preserving ambiguity and unknown authority. The eyes and graph cannot choose an
action.

MBO adds only execution reality: bid/ask, displayed size/depth, spread, cost,
freshness and fillability. Missing MBO stays missing and is not replaced by a
constant market assumption.

## Typed Brain

Exactly three mechanisms remain registered:

1. DFP: directional structure → continuation BOS → displacement zone → first
   pullback → aligned 1m trigger;
2. LSR: visible liquidity pool → sweep → opposite displacement/MSS → frozen
   reversal location → aligned trigger;
3. FAVR: mature accumulation/dealing range → failed outside auction → re-entry
   and displacement back inside → first pullback/trigger → midpoint or opposite
   boundary liquidity.

FAVR remains parked whenever a mature range and value cannot be established
with natural market evidence. A general rejection is not a FAVR substitute.

Each hypothesis exposes separate typed dimensions:

- `thesis_strength`;
- deterministic `sequence_progress`;
- `location_quality`;
- `entry_readiness`;
- `delivery_quality`;
- `uncertainty`.

The common phase vocabulary is:

`inactive → forming → armed → waiting_location → waiting_trigger → executable
→ entered → delivering/weakening → completed/invalidated`

Episode, terminal and rearm semantics prevent a later event from silently
rewriting the active thesis. A stable evidence revision is assimilated once,
not repeatedly every minute.

LSR source authority is tiered without deleting 1m observations: H4/H1 or
typed external/intermediate liquidity may establish Tier A; connected 15m/5m
internal liquidity may establish Tier B; isolated or nested 1m internal
liquidity is Tier C trigger/refinement evidence only. A mature balance range is
rare optional context for LSR, never its hard gate, and FAVR remains parked.

## Decision, risk and execution

Flat states compare enter, wait and abstain. Open states compare hold, protect,
exit and abstain. An unclear best-versus-second-best advantage resolves to
abstain.

Risk is a separate hard boundary. Entry freezes planned entry, structural
invalidation, selected draw/targets, deadline and maximum risk. Later extrema
cannot rewrite the original thesis. Cost, stale or anomalous data, insufficient
depth, consumed draw, invalid stop provenance and deadline can veto entry.

An approved order becomes eligible only at the next tradable clock. An attempt
must become filled, expired or cancelled. A fill creates one position, which
ends completed, invalidated or at a data boundary. Same-bar stop/target
ambiguity is adverse-first, and position/risk feedback enters the next Brain
update.

## Replay, audit and validation

Normal historical replay emits only light decision rows, aggregate summaries,
progress, checkpoints and resumable shards. Brain calibration mode adds typed
calibration rows. The prior frozen-packet, sealing, full-trace and audit-record
stack is intentionally absent from the current runtime.

Once the longitudinal chain is stable, case-level diagnostics may be rebuilt as
a bounded sampled minute trace plus a separate future view. Any AI comment must
still be translated into a computable sequence primitive and can never become
an action label.

[`../configs/data_splits.json`](../configs/data_splits.json) separates
development, calibration, rolling OOF and sealed OHLCV, plus MBO development
and sealed execution holdout. It binds causal artifacts and their manifests by
exact SHA-256.
