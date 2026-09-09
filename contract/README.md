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
| `execution` | `reality.py` | `ExecutionObservation` `execution_not_evaluated` `PositionSnapshot` `AccountState` |
| `eye` | `vocabulary.py` | the lifecycle enums and frozen reason strings the detectors emit |
| | `entities.py` | one immutable dataclass per market object — swings, structure, BOS, S/R, pools, ranges, manipulations, FVGs, order blocks, entry locations, reacceptance, inventory |
| | `interaction.py` | `MicroBreakFact` `MicroBOSReference` `PathSequenceStep` `PathSequenceState` `InteractionUpdate` |
| | `observation.py` | `MarketEvent` `FrameObservation` `DisplacementObservation` and **`MarketObservation`** — the Eye's sole output |
| `brain` | `vocabulary.py` | conflict roles, the neutral-state schema version |
| | `plan.py` | `DrawSelection` `LiquidityRoute` `TradePlan` `PlanFeasibility` and the frozen trigger/context objects |
| | `hypothesis.py` | `Evidence` `HypothesisSequenceState` **`HypothesisBelief`** |
| | `context.py` | `AuthorityLayer` `BalanceContext` obstructions, `OpenMarketThesis`, the Context/Episode lifecycles, `GlobalMarketContext` `NeutralMarketState` |
| | `belief.py` | **`MarketBelief`** `FrozenThesis` |
| `decision` | `action.py` | `Action` `ActionUtility` `Decision` |
| `risk` | `assessment.py` | `VetoCode` `RiskAssessment` |
| `research` | `snapshot.py` | `EngineSnapshot` `NeutralEngineSnapshot` |

`execution` sits before `eye` because `MarketObservation` carries an
`ExecutionObservation`: the engine scores execution reality and the Eye only
transports the result.

## How to import

Import from the layer that owns the type, so a consumer's imports state which
boundaries it actually depends on:

```python
from contract.eye.observation import MarketObservation
from contract.brain.belief import MarketBelief
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

## Remaining upward dependencies

Two contract modules still import a subsystem. Both were repository-wide before
the split and are now confined to one module each:

- `contract/eye/entities.py` imports `FOUNDATION_VERSION` from
  `eyes.core.foundation_registry` (one comparison in `StructuralLegState`).
- `contract/brain/belief.py` imports `PathBeliefUpdateRecord` and
  `PathCompetitionSetState` from `brain.core.market_belief` (four isinstance
  checks in `MarketBelief`).

`contract/eye/observation.py` names `MarketSnapshot` under `TYPE_CHECKING` with
two deferred local imports, which is the pattern the other two should follow.
