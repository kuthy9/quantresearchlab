"""Formal inter-module data contracts.

Every payload that crosses a subsystem boundary is defined here, and nowhere
else. The seven packages are strictly layered: a module may import from a
package earlier in this list and never from a later one.

    market  ->  execution  ->  eye  ->  brain  ->  decision  ->  risk  ->  research

| package | owns |
| --- | --- |
| `market` | shared vocabulary, price-grid arithmetic, bars and candles |
| `execution` | execution reality, position and account snapshots |
| `eye` | detector lifecycles, entity state, interaction facts, `MarketObservation` |
| `brain` | plans, hypotheses, context, theses, `MarketBelief` |
| `decision` | the action vocabulary and the utility comparison result |
| `risk` | veto codes and the independent risk verdict |
| `research` | per-clock engine snapshots for replay and study |

Import from the layer that owns the type -- ``from contract.eye.observation
import MarketObservation`` -- so a consumer's imports state which layers it
actually depends on. This package deliberately re-exports nothing itself.
"""
