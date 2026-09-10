"""Classifying the candidate collection must cost one pass per bar.

``range_role``, ``normalized_location_in_range`` and the swing ``rank`` are
snapshot projections: the registry binds them as "snapshot-derived only ... no
canonical membership event is claimed".  The reducer nevertheless folded them
back into the stored candidate at the tail of *every* reduced event, so a bar
that reduced ten events walked the whole collection ten times and re-sorted it
ten times.

The collection is effectively append-only -- a sweep disarms a level and keeps
it so a later re-approach is recognisably the same level -- so that per-event
fold made the cost of one bar proportional to accumulated market history, and
the cost of a run quadratic in its length.  Over 2022-02 the Eye created 15,000
levels and retired 84.

A projection is a function of the state it is read from, so it belongs where
the state is published, once, and not on every event that reaches the reducer.
"""
from __future__ import annotations

from eyes.core import market_state
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"


def _eye() -> tuple[CausalMarketReader, CausalObserver]:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            displacement_protocol=DISPLACEMENT_PROTOCOL,
            zone_protocol=GROUP3_PROTOCOL,
            range_auction_protocol=GROUP4_PROTOCOL,
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    return reader, observer


def _candidate_population(observer: CausalObserver) -> int:
    states = observer.market_snapshot_publisher._event_reducer.states
    return sum(len(state.liquidity.candidates) for state in states.values())


def test_one_bar_classifies_each_candidate_at_most_once(monkeypatch) -> None:
    """The pass count follows the collection, never the bar's event count."""

    reader, observer = _eye()
    bars = session_bars(2)
    for bar in bars[:-1]:
        observer.observe(reader.on_bar(bar))

    before = _candidate_population(observer)
    assert before, "the fixture produced no liquidity candidates"

    calls = 0
    real = market_state._candidate_range_membership

    def counted(candidate, range_state):
        nonlocal calls
        calls += 1
        return real(candidate, range_state)

    monkeypatch.setattr(market_state, "_candidate_range_membership", counted)
    observer.observe(reader.on_bar(bars[-1]))
    after = _candidate_population(observer)

    assert calls <= max(before, after), (
        f"one bar classified {calls} candidates against a collection of "
        f"{max(before, after)}; the projection is still folded per event"
    )
