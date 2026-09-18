# Contract

Every payload that crosses a subsystem boundary is defined here, and nowhere
else. This package replaced `shares/core/model.py`, an 11,232-line module that
held the Eye's facts, the Brain's beliefs, the Decision verdict, the Risk
verdict and the execution snapshots in one file, and that all four subsystems
therefore had to import in full.

## Layering

A module may import from a package earlier in this chain and never from a later
one. `eyes/tests/test_eye_module_boundary.py` enforces it per package.

```
market  ->  execution  ->  eye  ->  brain  ->  decision  ->  risk  ->  research
```

| package | module | owns |
| --- | --- | --- |
| `market` | `primitives.py` | `Direction` `Timeframe` `Playbook` `PlaybookPhase` `MarketMode` `ScaleRelation`; the price-grid arithmetic (`price_to_ticks`, `ticks_to_price`, `ohlc_to_ticks`); `Bar` `Candle` `BarCoverage`; `LiquidityLevel` `StructuralLevel`; `FrozenDict` `to_primitive` `content_hash` `aware_timestamp` `clamp` |
| `execution` | `reality.py`, `account.py` | `ExecutionObservation` `execution_not_evaluated` `PositionSnapshot` `AccountState`; (2026-09-16) `AccountSnapshot` `Position` `OrderState` `OrderStatus` `OrderRole` `Fill` `BracketIntent` `BrokerEvent` — what a broker reports and what the executor submits |
| `eye` | `vocabulary.py` | the lifecycle enums and frozen reason strings the detectors emit |
| | `entities.py` | one immutable dataclass per market object — swings, structure, BOS, S/R, pools, ranges, manipulations, FVGs, order blocks, entry locations, reacceptance, inventory |
| | `interaction.py` | `MicroBreakFact` `MicroBOSReference` `PathSequenceStep` `PathSequenceState` `InteractionUpdate` |
| | `observation.py` | `MarketEvent` `FrameObservation` `DisplacementObservation` and **`MarketObservation`** — the Eye's sole output |
| `brain` | `state.py` | **`BrainState`** — the LLM Brain's persistent state: `ActiveExpectation` `EvidenceItem` `EvidenceLedger` `WatchItem` `Opportunity` `RegisteredObject` `LastUpdate` and the enums `BrainStatus` `OpportunityState` `TradeDirection` `Confidence` `Verdict` |
| | `llm.py` | **`LLMInput`** (the exact payload sent, hashed as `input_sha`) and **`LLMUpdate`** (the one JSON object the model may reply with), `EvidenceVerdict`, `parse_update` — the gate that refuses any reply with an unknown alias, a missing key or a price — `LLM_UPDATE_EXAMPLE`, `MalformedReply`, `canonical_json` |
| | `vocabulary.py` | conflict roles, the neutral-state schema version (neutral-projection contracts, kept for `shares/core/scene_graph.py`) |
| | `plan.py` | `DrawSelection` `LiquidityRoute` `TradePlan` `PlanFeasibility` and the frozen trigger/context objects |
| | `hypothesis.py` | `Evidence` `HypothesisSequenceState` **`HypothesisBelief`** |
| | `context.py` | `AuthorityLayer` `BalanceContext` obstructions, `OpenMarketThesis`, the Context/Episode lifecycles, `GlobalMarketContext` `NeutralMarketState` |
| | `belief.py` | **`MarketBelief`** `FrozenThesis` |
| `decision` | `opportunity.py` | **`OpportunityGeometry`** (entry / stop / target price and reward-to-risk, resolved by code from the objects the LLM named), `OpportunityProposal` (= `Opportunity`), `GeometryError` |
| | `action.py` | `Action` `ActionUtility` `Decision` — the retired typed vertical's vocabulary, kept for `shares/core/visualization.py` and the shares test helpers |
| `risk` | `assessment.py`, `plan.py` | `VetoCode` (with `EXPOSURE` / `WORKING_ORDER` / `POSITION_SIZE` since 2026-09-16), `RiskAssessment` (inert, same status as `decision/action.py`); `ObjectRef` `TradePlan` (aliases + Eye entity ids + geometry, `signature`) `RiskVerdict` — what `risk/core/gate.py` reads and returns |
| `research` | `snapshot.py` | `EngineSnapshot` `NeutralEngineSnapshot` — same status; consumed by `shares/core/visualization.py` |

`brain/plan.py`, `hypothesis.py`, `context.py` and `belief.py` are the
neutral-projection contracts of the typed Brain retired on 2026-09-07 and
2026-09-16; they stay because `shares/core/scene_graph.py`,
`shares/core/market_cases.py`, `shares/core/visualization.py` and their tests
read them as inert dataclasses. No runtime module produces them.
`brain/forecast.py` (the kNN Brain's `MarketBeliefState`) was deleted on
2026-09-16 with the mechanical Brain.

`execution` sits before `eye` because `MarketObservation` carries an
`ExecutionObservation`: the engine scores execution reality and the Eye only
transports the result.

## How to import

Import from the layer that owns the type, so a consumer's imports state which
boundaries it actually depends on:

```python
from contract.eye.observation import MarketObservation
from contract.brain.state import BrainState
```

Each package's `__init__.py` re-exports its own modules, so
`from contract.eye import MarketObservation` also works. `contract/__init__.py`
deliberately re-exports nothing — importing the root must not pull every layer.

## What these contracts guarantee

The dataclasses are frozen and validate at construction: 66 `__post_init__`
methods carry 576 fail-closed checks. The rules that matter across boundaries:

- **Causality.** Any clock later than the payload's `asof` is rejected. A
  producer cannot hand a consumer a fact it could not have known yet.
- **Timezone.** Every timestamp passes `aware_timestamp`; naive input raises.
- **Exact types.** A published `MarketObservation` requires a real
  `MarketSnapshot`, not a duck-typed stand-in whose aliases could disagree.
- **Identity.** A mapping key must equal the identity field of its value.
- **Provenance namespaces.** `source_ids`, `source_event_ids`, `source_data_ids`,
  `source_entity_ids` and `context_event_ids` stay separate; event ancestry
  cannot be confused with raw-data or entity identity.
- **Price grid.** OHLC is normalized to integer ticks with `Decimal`, so an
  off-grid vendor bar is refused before it reaches any state.

## Remaining upward dependency

One contract module still imports a subsystem:

- `contract/eye/entities.py` imports `FOUNDATION_VERSION` from
  `eyes.core.foundation_registry` (one comparison in `StructuralLegState`).

`contract/brain/belief.py` used to import `PathBeliefUpdateRecord` and
`PathCompetitionSetState` from `brain.core.market_belief`. Retiring the six-path
competition set on 2026-09-09 removed that import, and with it the second
upward edge.

`contract/eye/observation.py` names `MarketSnapshot` under `TYPE_CHECKING` with
two deferred local imports, which is the pattern the remaining one should
follow.
