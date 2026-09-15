"""The live-entity harvest names every entity a retained state still cites.

The emitter's per-entity memories are read for as long as some tracker or
reducer retains the entity -- or retains another entity that cites it (a
BOS names its target swing, an FVG its source candles, a leg its ATR bars).
The harvest therefore walks every retained state and collects every
identifier field, in both its namespaced and its bare form, so a memory
keyed either way can ask "is this still live?" with one lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

from eyes.core.entity_liveness import harvest_entity_ids


@dataclass(frozen=True)
class _Break:
    bos_id: str
    target_swing_id: str
    break_bar_id: str | None
    strength: float


@dataclass(frozen=True)
class _Frame:
    timeframe: str
    structure_breaks: tuple[_Break, ...]
    source_candle_ids: tuple[str, ...]
    metrics: dict[str, float]


def test_harvest_collects_every_id_field_through_nesting() -> None:
    frame = _Frame(
        timeframe="5m",
        structure_breaks=(
            _Break("b" * 24, "s" * 24, "c" * 64, 0.5),
            _Break("d" * 24, "e" * 24, None, 0.25),
        ),
        source_candle_ids=("f" * 64,),
        metrics={"atr": 1.5},
    )
    live = harvest_entity_ids(frame)
    assert {"b" * 24, "s" * 24, "c" * 64, "d" * 24, "e" * 24, "f" * 64} <= live
    # Non-identifier strings and numbers are not entities.
    assert "5m" not in live
    assert 0.5 not in live


def test_harvest_keeps_namespaced_and_bare_forms() -> None:
    live = harvest_entity_ids(
        {"candidates": [{"candidate_id": "swing:" + "a" * 24}]},
        {"items": ({"item_id": "pool:" + "b" * 24},)},
        {"level_id": "reference:session:2022-02-03:high:NQH2:3541"},
    )
    assert "swing:" + "a" * 24 in live and "a" * 24 in live
    assert "pool:" + "b" * 24 in live and "b" * 24 in live
    assert "reference:session:2022-02-03:high:NQH2:3541" in live


def test_harvest_ignores_none_sources_and_events() -> None:
    assert harvest_entity_ids(None, ()) == frozenset()
