"""One row per new Group-5 step, with the geometry the Setup gate labels from,
read at the bar the step is published."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.path_log import (
    LEVEL_COLUMNS,
    PATH_COLUMNS,
    STRUCTURE_COLUMNS,
    direction_sign,
    empty_path_log,
    minutes_since_open,
    path_rows,
    realized_volatility,
)
from contract.market import Direction, Timeframe

AT = pd.Timestamp("2022-01-04T15:30", tz="UTC")


def _step(step_id: str, kind: str, reason: str, strength: float) -> SimpleNamespace:
    return SimpleNamespace(step_id=step_id, kind=kind, reason=reason, strength=strength, observed_at=AT)


def _path(sequence_id: str, context_kind: str, context_id: str, direction, steps) -> SimpleNamespace:
    return SimpleNamespace(
        sequence_id=sequence_id, context_kind=context_kind, context_id=context_id,
        direction=direction, formed_at=AT - pd.Timedelta(minutes=30), lifecycle="active", steps=tuple(steps),
    )


def _timeframe_state(bsl, ssl, ext=Direction.LONG):
    return SimpleNamespace(
        liquidity=SimpleNamespace(unswept_bsl=bsl, unswept_ssl=ssl),
        structure=SimpleNamespace(external_direction=ext, internal_direction=Direction.SHORT, last_bos_direction=None),
    )


def _observation(update, manipulations=()):
    snapshot = SimpleNamespace(
        asof=AT,
        timeframe_states={
            Timeframe.M5: _timeframe_state([101.0, 103.0], [98.0, 99.5]),
            Timeframe.M15: _timeframe_state([104.0], [97.0]),
            Timeframe.H1: _timeframe_state([], [90.0], ext=Direction.SHORT),
        },
    )
    return SimpleNamespace(market_snapshot=snapshot, asof=AT, interaction_update=update, manipulations=tuple(manipulations))


def _rows(update, manipulations=()):
    return path_rows(
        _observation(update, manipulations), close=100.0, high=100.5, low=99.5, atr=2.0,
        history=[100.0 + 0.1 * i for i in range(90)], features=tuple(float(i) for i in range(len(FEATURE_NAMES))),
    )


def test_zone_return_row_carries_zone_geometry_levels_structure_and_state() -> None:
    first = _step("s:0", "zone_visible", "typed_entry_zone_registered", 0.6)
    second = _step("s:1", "reacceptance_held", "held", 0.8)
    path = _path("s", "zone_return", "loc-1", Direction.LONG, [first, second])
    location = SimpleNamespace(
        location_id="loc-1", lower_bound=99.0, upper_bound=99.8, near_edge=99.8, far_edge=99.0,
        failure_boundary=98.7, source_zone_kind="fvg", entry_mode="touch", first_penetration_fraction=0.25,
        nearest_visible_draw_distance_points=3.0,
    )
    # the zone's own reacceptance (context_kind "entry_zone", context_id = location_id)
    reacceptance = SimpleNamespace(
        context_id="loc-1", reference_price=99.8, failure_boundary=98.7, reclaim_margin_atr=0.3, hold_margin_atr=0.2,
    )
    update = SimpleNamespace(
        milestone_transitions=(("s", second),), interaction_paths=(path,), interaction_path_transitions=(),
        zone_interactions=(location,), reacceptance_interactions=(reacceptance,),
    )
    (row,) = _rows(update)
    assert (row["reference_price"], row["reclaim_margin_atr"], row["hold_margin_atr"]) == (99.8, 0.3, 0.2)
    assert set(row) == set(PATH_COLUMNS)
    assert row["known_at"] == AT and row["sequence_id"] == "s" and row["context_kind"] == "zone_return"
    assert row["direction"] == 1.0 and row["context_found"] is True
    assert row["step_kind"] == "reacceptance_held" and row["step_ordinal"] == 1
    assert json.loads(row["steps_so_far"]) == [
        ["zone_visible", "typed_entry_zone_registered", 0.6], ["reacceptance_held", "held", 0.8]
    ]
    assert row["failure_boundary"] == 98.7 and row["source_zone_kind"] == "fvg" and row["eye_draw_distance_points"] == 3.0
    assert math.isnan(row["sweep_extreme"])
    assert row["close"] == 100.0 and row["atr_1m"] == 2.0
    assert (row["bsl_5m"], row["ssl_5m"], row["bsl_15m"], row["ssl_15m"]) == (101.0, 99.5, 104.0, 97.0)
    assert math.isnan(row["bsl_1h"]) and row["ssl_1h"] == 90.0
    assert (row["ext_dir_5m"], row["int_dir_5m"], row["last_bos_dir_5m"], row["ext_dir_1h"]) == (1.0, -1.0, 0.0, -1.0)
    assert row[FEATURE_NAMES[0]] == 0.0 and row[FEATURE_NAMES[-1]] == float(len(FEATURE_NAMES) - 1)


def test_pool_reversal_row_reads_the_group4_manipulation_state() -> None:
    # A pool path's context is a Group-4 ManipulationState: its ``timeframe`` is
    # the 1m clock it is maintained on, the swept pool's scale is
    # ``source_timeframe``; there is no Group-5 ReacceptanceState for pools, so
    # the reference is the swept boundary and the reclaim margin comes from
    # ``reentry_price``.
    step = _step("p:0", "pool_swept", "typed_pool_manipulation_swept", 0.4)
    path = _path("p", "pool_reversal", "man-1", Direction.SHORT, [step])
    manipulation = SimpleNamespace(
        manipulation_id="man-1", side="above", source_lower_bound=104.0, source_upper_bound=104.5,
        sweep_extreme=105.2, penetration_atr=0.35, timeframe=Timeframe.M1, source_timeframe=Timeframe.M5,
        reentry_price=None,
    )
    update = SimpleNamespace(
        milestone_transitions=(("p", step),), interaction_paths=(path,), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update, [manipulation])
    assert row["direction"] == -1.0 and row["failure_boundary"] == 105.2 and row["source_timeframe"] == "5m"
    assert row["reference_price"] == 104.5 and math.isnan(row["reclaim_margin_atr"]) and math.isnan(row["hold_margin_atr"])
    assert math.isnan(row["lower_bound"])
    manipulation.reentry_price = 104.1
    (row,) = _rows(update, [manipulation])
    assert row["reclaim_margin_atr"] == (104.5 - 104.1) / 2.0  # |reentry − reference| in ATR₁ₘ
    manipulation.side = "below"
    (row,) = _rows(update, [manipulation])
    assert row["reference_price"] == 104.0


def test_a_pool_context_outlives_its_compacted_manipulation_state() -> None:
    # Group 4 keeps a resolved manipulation in the observation for one
    # completed bar, then compacts it; the path's later steps (a micro-break,
    # an opposite displacement) come after that. The recorder remembers the
    # last state it saw for every live pool path and reads the Setup from it,
    # so those rows carry the sweep extreme the path is still measured
    # against; a context the recorder never saw is still logged as missing.
    swept = _step("p:0", "pool_swept", "typed_pool_manipulation_swept", 0.4)
    broke = _step("p:1", "micro_break_observed", "typed_1m_bos_confirmed", 0.7)
    manipulation = SimpleNamespace(
        manipulation_id="man-1", side="above", source_lower_bound=104.0, source_upper_bound=104.5,
        sweep_extreme=105.2, penetration_atr=0.35, timeframe=Timeframe.M1, source_timeframe=Timeframe.M5,
        reentry_price=None,
    )
    memory: dict = {}
    first = SimpleNamespace(
        milestone_transitions=(("p", swept),), interaction_paths=(_path("p", "pool_reversal", "man-1", Direction.SHORT, [swept]),),
        interaction_path_transitions=(), zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = path_rows(_observation(first, [manipulation]), close=100.0, high=100.5, low=99.5, atr=2.0,
                       history=[100.0] * 90, features=tuple(float(i) for i in range(len(FEATURE_NAMES))),
                       pool_memory=memory)
    assert row["context_found"] and row["failure_boundary"] == 105.2
    # the resolution bar exposes the re-entry; the state is gone the bar after
    manipulation.reentry_price = 104.1
    resolved = SimpleNamespace(
        milestone_transitions=(), interaction_paths=first.interaction_paths, interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    assert path_rows(_observation(resolved, [manipulation]), close=100.0, high=100.5, low=99.5, atr=2.0,
                     history=[100.0] * 90, features=tuple(float(i) for i in range(len(FEATURE_NAMES))),
                     pool_memory=memory) == []
    later = SimpleNamespace(
        milestone_transitions=(("p", broke),),
        interaction_paths=(_path("p", "pool_reversal", "man-1", Direction.SHORT, [swept, broke]),),
        interaction_path_transitions=(), zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = path_rows(_observation(later, []), close=100.0, high=100.5, low=99.5, atr=2.0,
                       history=[100.0] * 90, features=tuple(float(i) for i in range(len(FEATURE_NAMES))),
                       pool_memory=memory)
    assert row["context_found"] and row["failure_boundary"] == 105.2 and row["source_timeframe"] == "5m"
    assert row["reclaim_margin_atr"] == (104.5 - 104.1) / 2.0 and row["step_ordinal"] == 1
    # without a memory the row is logged as missing, as before
    (bare,) = _rows(later, [])
    assert not bare["context_found"] and math.isnan(bare["failure_boundary"])
    # the memory follows the live paths: a path that left takes its state with it
    gone = SimpleNamespace(
        milestone_transitions=(), interaction_paths=(), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    path_rows(_observation(gone, []), close=100.0, high=100.5, low=99.5, atr=2.0,
              history=[100.0] * 90, features=tuple(float(i) for i in range(len(FEATURE_NAMES))), pool_memory=memory)
    assert memory == {}


def test_the_live_path_wins_over_a_same_bar_transition_copy() -> None:
    # The reducer publishes a registered path in the transitions before it
    # appends the same bar's departure step to the live copy.
    first = _step("s:0", "zone_visible", "typed_entry_zone_registered", 0.6)
    second = _step("s:1", "departure_confirmed", "formation_close_on_delivery_side", 0.5)
    stale = _path("s", "zone_return", "loc-1", Direction.LONG, [first])
    live = _path("s", "zone_return", "loc-1", Direction.LONG, [first, second])
    update = SimpleNamespace(
        milestone_transitions=(("s", second),), interaction_paths=(live,), interaction_path_transitions=(stale,),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert row["step_ordinal"] == 1 and len(json.loads(row["steps_so_far"])) == 2
    # a step the published path does not show is marked, not guessed
    update = SimpleNamespace(
        milestone_transitions=(("s", second),), interaction_paths=(stale,), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert row["step_ordinal"] == -1 and json.loads(row["steps_so_far"]) == [["zone_visible", "typed_entry_zone_registered", 0.6]]


def test_missing_context_is_logged_with_nan_geometry_and_flagged() -> None:
    step = _step("s:0", "zone_visible", "typed_entry_zone_registered", 0.6)
    path = _path("s", "zone_return", "loc-missing", Direction.LONG, [step])
    update = SimpleNamespace(
        milestone_transitions=(("s", step),), interaction_paths=(path,), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert row["context_found"] is False and math.isnan(row["failure_boundary"])
    # a step whose path is not on the update at all cannot be typed
    update = SimpleNamespace(
        milestone_transitions=(("ghost", step),), interaction_paths=(), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    (row,) = _rows(update)
    assert row["context_kind"] is None and row["context_found"] is False


def test_no_update_or_no_transitions_yields_nothing() -> None:
    assert _rows(None) == []
    empty = SimpleNamespace(
        milestone_transitions=(), interaction_paths=(), interaction_path_transitions=(),
        zone_interactions=(), reacceptance_interactions=(),
    )
    assert _rows(empty) == []


def test_helpers() -> None:
    assert direction_sign(Direction.LONG) == 1.0 and direction_sign("short") == -1.0 and direction_sign(None) == 0.0
    # 15:30 UTC on 2022-01-04 is 10:30 New York, 16.5 h after the 18:00 open
    assert minutes_since_open(AT) == 990.0
    assert minutes_since_open(pd.Timestamp("2022-01-04T23:00", tz="UTC")) == 0.0  # 18:00 New York
    closes = [100.0, 101.0, 100.0, 102.0]
    assert realized_volatility(closes, 3, 2.0) == np.sqrt(1 + 1 + 4) / 2.0
    assert math.isnan(realized_volatility([100.0], 3, 2.0))
    assert list(empty_path_log().columns) == list(PATH_COLUMNS)
    assert len(LEVEL_COLUMNS) == 6 and len(STRUCTURE_COLUMNS) == 9


def test_path_columns_are_unique_and_the_eye_state_does_not_shadow_them() -> None:
    assert len(PATH_COLUMNS) == len(set(PATH_COLUMNS))
    assert not (set(FEATURE_NAMES) & (set(PATH_COLUMNS) - set(FEATURE_NAMES)))
