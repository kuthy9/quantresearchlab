"""The emitter locates bar roots by bisection, never by walking a scale's history.

Each scale's ``(clock, event id)`` bar-root lists grow for the life of the
run -- 642 entries at bar 500 and 25,752 at bar 20,000 of 2022-02 -- and the
emitter walked them from the front to find a confirmed swing's pivot bar,
rebuilt an id-to-clock dictionary over all of them for every swing window,
and scanned for the bar after a swing crossing and for an exact clock root.
Together those scans grew from 0.01 s to 1.9 s per 500 bars.  The lists are
kept in clock order, so every lookup is a bisection.
"""
from __future__ import annotations

from contract.eye import SwingLifecycle
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig

from eyes.tests.test_facts_on_every_scale import _noisy
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


class _ScanCounting(list):
    scans: list[str] = []

    def __iter__(self):
        type(self).scans.append("iter")
        return super().__iter__()

    def __reversed__(self):
        type(self).scans.append("reversed")
        return super().__reversed__()


def _observer() -> tuple[CausalMarketReader, CausalObserver]:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    return reader, observer


def _guard(observer: CausalObserver) -> None:
    emitter = observer._emitter
    for name in ("_bar_event_ids_by_timeframe", "_real_bar_event_ids_by_timeframe"):
        table = getattr(emitter, name)
        for timeframe, entries in table.items():
            table[timeframe] = _ScanCounting(entries)
    _ScanCounting.scans = []


def test_bar_root_lookups_never_iterate_a_scale_history() -> None:
    reader, observer = _observer()
    bars = _noisy(session_bars(2), seed=5)
    for bar in bars[:500]:
        observation = observer.observe(reader.on_bar(bar))
    _guard(observer)
    for bar in bars[500:700]:
        observation = observer.observe(reader.on_bar(bar))
    assert _ScanCounting.scans == [], _ScanCounting.scans[:5]
    emitter = observer._emitter
    swing = max(
        (
            item
            for item in observation.frames[Timeframe.M1].swings
            if item.lifecycle is SwingLifecycle.CONFIRMED
        ),
        key=lambda item: item.confirmed_at,
    )
    window = emitter._swing_window_event_ids(swing)
    geometry = emitter._swing_window_geometry(window, Timeframe.M1)
    assert len(window) == 2 * emitter._structure_config.span_for(Timeframe.M1) + 1
    assert geometry["window_end"] == swing.confirmed_at.isoformat()
    last_clock = observation.frames[Timeframe.M1].cutoff
    assert emitter._clock_root_event_id_at(Timeframe.M1, last_clock)
    assert emitter._bar_event_id_at(Timeframe.M1, last_clock)
    assert emitter._latest_bar_event_id_at_or_before(Timeframe.M1, last_clock)
    assert _ScanCounting.scans == [], _ScanCounting.scans[:5]


def test_bar_roots_stay_in_clock_order_across_scales() -> None:
    reader, observer = _observer()
    for bar in _noisy(session_bars(1)[:400], seed=5):
        observer.observe(reader.on_bar(bar))
    emitter = observer._emitter
    for table in (
        emitter._bar_event_ids_by_timeframe,
        emitter._real_bar_event_ids_by_timeframe,
    ):
        for timeframe, entries in table.items():
            clocks = [clock for clock, _ in entries]
            assert clocks == sorted(clocks), timeframe
            assert len(set(clocks)) == len(clocks), timeframe
