"""The Eye must publish the first re-entry into an FVG as its own frozen fact.

``FVG_PARTIALLY_FILLED`` and ``FVG_FULLY_FILLED`` are revisable fill
observations: a gap can be partially filled several times and the fraction only
ratchets.  The *first* re-entry is a different, non-revisable fact -- it happens
once per gap and its context (how deep, how old, how fast price arrived, which
displacement produced the gap) is only true at that instant.  v1.3 registers it
as ``FVG_FIRST_RETEST``.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import pytest

from eyes.core.causal import CausalMarketReader
from contract.market import (
    Bar,
    Timeframe,
    ticks_to_price,
)
from contract.eye import EventKind
from eyes.core.observation import CausalObserver, ObserverConfig

from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"

WARMUP_BARS = 600


def _grid(value: float) -> float:
    coordinate = (Decimal(str(value)) / Decimal("0.25")).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    return ticks_to_price(int(coordinate), 0.25)


def _minute(previous: Bar, *, offset: int, o: float, h: float, l: float, c: float) -> Bar:
    return Bar(
        start=previous.start + pd.Timedelta(minutes=offset),
        open=_grid(o),
        high=_grid(h),
        low=_grid(l),
        close=_grid(c),
        volume=120,
        symbol=previous.symbol,
        instrument_id=previous.instrument_id,
    )


def _bullish_fvg_then_retest() -> list[Bar]:
    """Warm up on registered bars, then form one bullish M5 gap and re-enter it.

    Each crafted M5 candle is five identical one-minute bars so the M5
    geometry is exactly the numbers written here.
    """

    warmup = session_bars(1)[:WARMUP_BARS]
    last = warmup[-1]
    base = float(last.close)

    # (open, high, low, close) per M5 candle, as offsets from the warmup close.
    plan = (
        (0.0, 0.5, -0.5, 0.0),       # c1: high = base + 0.5 -> far edge
        (0.0, 10.0, 0.0, 10.0),      # c2: the impulse
        (10.0, 12.0, 5.0, 11.0),     # c3: low = base + 5.0 -> near edge
        (11.0, 11.5, 4.0, 5.0),      # first re-entry: shallow, above the midpoint
    )
    crafted: list[Bar] = []
    offset = 1
    for o, h, l, c in plan:
        for step in range(5):
            crafted.append(
                _minute(
                    last,
                    offset=offset,
                    o=base + (o if step == 0 else c),
                    h=base + h,
                    l=base + l,
                    c=base + c,
                )
            )
            offset += 1
    return [*warmup, *crafted]


def _observer() -> tuple[CausalMarketReader, CausalObserver]:
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


def _replay() -> CausalObserver:
    reader, observer = _observer()
    for bar in _bullish_fvg_then_retest():
        observer.observe(reader.on_bar(bar))
    return observer


def test_first_reentry_into_a_gap_is_published_once_with_frozen_entry_context() -> None:
    observer = _replay()
    events = observer.audit_store.events()

    created = [
            e for e in events if e.kind is EventKind.FVG_CREATED and e.timeframe is Timeframe.M5
        ]
    assert created, "the crafted bars did not form an FVG"
    gap = created[-1]
    gap_id = gap.details["fvg_id"]

    retests = [
        e
        for e in events
        if e.kind is EventKind.FVG_FIRST_RETEST
        and e.details["fvg_id"] == gap_id
    ]
    assert len(retests) == 1
    retest = retests[0]

    fill = retest.details["fill_depth_at_entry"]
    assert 0.0 < fill < 1.0
    assert retest.details["age_bars"] >= 1
    assert retest.details["age_seconds"] > 0
    assert retest.details["entry_lifecycle"] == "partial"
    assert retest.details["qualification"] == gap.details["qualification"]
    assert retest.details["session"]
    # Price closed the last of its distance to the near edge on the entry bar,
    # so the approach is a positive number of ATRs per bar.
    assert retest.details["approach_speed_atr"] >= 0.0
    assert gap.event_id in retest.source_event_ids


def test_first_retest_precedes_the_fill_observation_it_shares_a_bar_with() -> None:
    observer = _replay()
    events = observer.audit_store.events()
    order = {event.event_id: index for index, event in enumerate(events)}

    retest = next(
        e
        for e in events
        if e.kind is EventKind.FVG_FIRST_RETEST and e.timeframe is Timeframe.M5
    )
    fill = next(
        e
        for e in events
        if e.kind is EventKind.FVG_PARTIALLY_FILLED
        and e.details["fvg_id"] == retest.details["fvg_id"]
    )
    assert order[retest.event_id] < order[fill.event_id]
    assert retest.known_at == fill.known_at
