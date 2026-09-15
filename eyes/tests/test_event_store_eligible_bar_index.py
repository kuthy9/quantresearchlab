"""The indexed eligible-bar sequence equals the scanned one, bar for bar.

``_validate_structural_leg_contract`` derived a leg's pivot and path bars by
scanning every event the store held for eligible ``BAR_COMPLETED`` roots of
the leg's scale -- a walk over the lifetime event count on every structural
leg, +8.6 s per 500 bars at bar 2,000 on the real tape.  The store now keeps
that sequence per (semantic version, scale, contract) as bars commit, and the
contract reads the index plus the bars staged in its own batch.
"""
from __future__ import annotations

from contract.eye import EventKind, EventOrigin
from contract.market import Timeframe, bar_evidence_coverage
from eyes.core.causal import CausalMarketReader
from eyes.core.event_store import EventStore
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


def _scanned(store: EventStore, event) -> tuple:
    symbol = event.evidence.get("symbol")
    instrument_id = event.evidence.get("instrument_id")
    return tuple(
        sorted(
            (
                candidate
                for candidate in store._by_id.values()
                if candidate.kind is EventKind.BAR_COMPLETED
                and candidate.origin is EventOrigin.NORMALIZED_DATA
                and candidate.semantic_version == event.semantic_version
                and candidate.timeframe is event.timeframe
                and candidate.event_time == candidate.known_at
                and bar_evidence_coverage(candidate.evidence).admits_definitional_path
                and candidate.evidence.get("symbol") == symbol
                and candidate.evidence.get("instrument_id") == instrument_id
            ),
            key=lambda candidate: (candidate.known_at, candidate.event_id),
        )
    )


def test_index_matches_scan_for_every_structural_leg_event() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    for bar in _noisy(session_bars(1)[:400]):
        observer.observe(reader.on_bar(bar))
    store = observer.audit_store
    legs = [e for e in store._by_id.values() if e.kind is EventKind.STRUCTURAL_LEG_CREATED]
    assert legs, "the replay produced no structural leg"
    scales = {leg.timeframe for leg in legs}
    assert len(scales) > 1, "the replay produced legs on one scale only"
    for leg in legs:
        indexed = store._eligible_bars(leg, staged_bars=())
        assert tuple(indexed) == _scanned(store, leg), leg.timeframe


def test_leg_contract_reads_only_the_bars_around_its_endpoints(monkeypatch) -> None:
    """The contract bisects the eligible sequence instead of walking it.

    With the index in hand it still derived the pivot clocks with linear
    ``next`` scans and filtered every bar at or before the start pivot for
    the ATR window -- one ``bar_evidence_coverage`` call per bar of the
    scale's lifetime on every leg, 17 ms per leg at bar 2,000.  Now it reads
    the fourteen strict-prior bars and the path, wherever they sit.
    """
    from eyes.core import event_store as store_module

    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    for bar in _noisy(session_bars(1)[:400]):
        observer.observe(reader.on_bar(bar))
    store = observer.audit_store
    legs = [e for e in store._by_id.values() if e.kind is EventKind.STRUCTURAL_LEG_CREATED]
    leg = max(
        (
            item
            for item in legs
            if item.timeframe is Timeframe.M1
            and item.evidence.get("foundation_version") is not None
        ),
        key=lambda item: item.known_at,
    )
    eligible = store._eligible_bars(leg, staged_bars=())
    assert len(eligible) > 200
    calls = []
    original = store_module.bar_evidence_coverage

    def counting(evidence):
        calls.append(evidence)
        return original(evidence)

    monkeypatch.setattr(store_module, "bar_evidence_coverage", counting)
    parents = tuple(store._by_id[parent] for parent in leg.source_event_ids)
    EventStore._validate_structural_leg_contract(
        leg,
        source_parents=parents,
        available_events=store._by_id,
        eligible_bars=eligible,
    )
    duration = int(leg.evidence["duration_bars"])
    # Fourteen ATR bars, the path, and the bars the contract inspects
    # directly -- never the scale's whole history.
    assert len(calls) <= 3 * (14 + duration) + 4, (len(calls), len(eligible))


def test_eligible_view_is_the_append_ordered_index_plus_the_staged_tail(monkeypatch) -> None:
    """The view concatenates two already-ordered sequences; it never re-sorts.

    The store reserves one normalized BAR root per (scale, clock) and commits
    in ``known_at`` order, so each index list is strictly increasing in
    ``known_at`` and every bar staged in a batch sits after the committed
    ones.  Sorting the whole scale on every structural leg cost
    O(N log N) per leg -- +5.9 s per 500 bars at bar 20,000 on the 2022-02
    tape -- for an order the index already had.
    """
    from eyes.core import event_store as store_module

    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    for bar in _noisy(session_bars(1)[:300]):
        observer.observe(reader.on_bar(bar))
    store = observer.audit_store
    leg = max(
        (
            item
            for item in store._by_id.values()
            if item.kind is EventKind.STRUCTURAL_LEG_CREATED
            and item.timeframe is Timeframe.M1
        ),
        key=lambda item: item.known_at,
    )
    key = EventStore._eligible_bar_key(
        next(
            bar
            for bar in store._by_id.values()
            if bar.timeframe is Timeframe.M1
            and EventStore._eligible_bar_key(bar) is not None
        )
    )
    full = list(store._eligible_bar_index[key])
    assert len(full) > 100
    committed, staged = full[:-3], full[-3:]

    sorts: list[object] = []

    def spy(*args, **kwargs):
        sorts.append(args)
        return sorted(*args, **kwargs)

    monkeypatch.setattr(store_module, "sorted", spy, raising=False)
    merged = EventStore._indexed_eligible_bars(
        leg,
        eligible_bar_index={key: committed},
        staged_eligible_bars={key: staged},
    )
    assert list(merged) == full
    alone = EventStore._indexed_eligible_bars(
        leg,
        eligible_bar_index={key: committed},
        staged_eligible_bars={},
    )
    assert list(alone) == committed
    assert sorts == []
