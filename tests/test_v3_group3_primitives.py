from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd
import pytest

from smc_trader.displacement import (
    CausalDisplacementTracker,
    DisplacementLifecycle,
    DisplacementProtocol,
    DisplacementUpdate,
)
from smc_trader.group3 import (
    CausalGroup3Tracker,
    FVG_BOUNDARY_REASONS,
    Group3BOSSource,
    Group3Protocol,
    WINDOW_RESET_REASONS,
)
from smc_trader.model import (
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    FairValueGapLifecycle,
    FVGQualification,
    ORDER_BLOCK_FUNNEL_STAGES,
    OrderBlockAttemptOutcome,
    OrderBlockLifecycle,
    Timeframe,
)
from smc_trader.structure import StructureConfig, StructureTracker


ROOT = Path(__file__).resolve().parents[1]
GROUP3_PROTOCOL_PATH = ROOT / "configs/primitives_zones.json"
STRUCTURE_PROTOCOL_PATH = (
    ROOT / "configs/primitives_structure_liquidity.json"
)
DISPLACEMENT_PROTOCOL_PATH = (
    ROOT
    / "configs/primitives_displacement.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


GROUP3_PROTOCOL_SHA = _sha256(GROUP3_PROTOCOL_PATH)
STRUCTURE_PROTOCOL_SHA = _sha256(STRUCTURE_PROTOCOL_PATH)
BASE = pd.Timestamp("2025-01-06T09:30:00-05:00")

_FVG_FROZEN_FIELDS = (
    "fvg_id",
    "protocol_hash",
    "symbol",
    "instrument_id",
    "timeframe",
    "direction",
    "qualification",
    "source_displacement_id",
    "source_active_transition_id",
    "source_displacement_protocol_hash",
    "source_displacement_started_at",
    "source_displacement_active_at",
    "source_displacement_prefix_commitment",
    "source_candle_ids",
    "source_candle_starts",
    "lower_bound",
    "upper_bound",
    "midpoint",
    "invalidation_price",
    "width_points",
    "width_ticks",
    "formation_atr",
    "width_atr",
    "strength",
    "formed_at",
    "confirmed_at",
)
_ORDER_BLOCK_FROZEN_FIELDS = (
    "order_block_id",
    "protocol_hash",
    "symbol",
    "instrument_id",
    "timeframe",
    "direction",
    "source_displacement_id",
    "source_active_transition_id",
    "source_displacement_protocol_hash",
    "source_displacement_seed_candle_id",
    "source_displacement_started_at",
    "source_displacement_active_at",
    "source_displacement_prefix_commitment",
    "source_bos_id",
    "source_bos_protocol_hash",
    "source_bos_target_swing_id",
    "source_bos_structure_id",
    "source_bos_scope",
    "source_bos_pending_at",
    "source_bos_resolved_at",
    "source_bos_break_bar_id",
    "source_bos_mss_qualified",
    "anchor_candle_id",
    "anchor_candle_ids",
    "anchor_start",
    "anchor_end",
    "anchor_open",
    "anchor_close",
    "lower_bound",
    "upper_bound",
    "body_lower_bound",
    "body_upper_bound",
    "midpoint",
    "invalidation_price",
    "width_points",
    "width_ticks",
    "width_atr",
    "strength",
    "formed_at",
    "confirmed_at",
)


def _group3_protocol() -> Group3Protocol:
    return Group3Protocol.from_file(GROUP3_PROTOCOL_PATH)


def _displacement_protocol() -> DisplacementProtocol:
    return DisplacementProtocol.from_file(DISPLACEMENT_PROTOCOL_PATH)


def _candle(
    index: int,
    values: tuple[float, float, float, float],
    *,
    symbol: str = "NQH5",
    instrument_id: int = 1,
    real_minutes: int = 5,
    synthetic_minutes: int = 0,
) -> Candle:
    start = BASE + pd.Timedelta(minutes=5 * index)
    return Candle(
        timeframe=Timeframe.M5,
        start=start,
        end=start + pd.Timedelta(minutes=5),
        open=values[0],
        high=values[1],
        low=values[2],
        close=values[3],
        volume=100.0,
        symbol=symbol,
        instrument_id=instrument_id,
        observed_minutes=5,
        expected_minutes=5,
        complete=True,
        real_minutes=real_minutes,
        synthetic_minutes=synthetic_minutes,
    )


class _Harness:
    def __init__(self) -> None:
        displacement_protocol = _displacement_protocol()
        self.group3 = CausalGroup3Tracker(
            _group3_protocol(),
            displacement_protocol_hash=(
                displacement_protocol.protocol_hash
            ),
            structure_protocol_hash=STRUCTURE_PROTOCOL_SHA,
        )
        self.displacement = CausalDisplacementTracker(
            displacement_protocol
        )
        self.index = 0

    def send(
        self,
        values: tuple[float, float, float, float],
        *,
        bos_factory=None,
        real_minutes: int = 5,
        synthetic_minutes: int = 0,
    ):
        candle = _candle(
            self.index,
            values,
            real_minutes=real_minutes,
            synthetic_minutes=synthetic_minutes,
        )
        displacement = self.displacement.on_completed_5m(candle)
        bos_states = (
            ()
            if bos_factory is None
            else tuple(bos_factory(candle, displacement))
        )
        bos_sources = tuple(
            _bos_source(bos, candle=candle)
            for bos in bos_states
        )
        output = self.group3.on_completed_5m(
            candle,
            displacement,
            bos_sources,
        )
        self.index += 1
        return candle, displacement, bos_states, output


def _warm(
    harness: _Harness,
    *,
    count: int = 15,
    atr: float = 1.0,
    overrides: dict[
        int,
        tuple[float, float, float, float],
    ] | None = None,
) -> tuple[Candle, ...]:
    output = []
    replacements = overrides or {}
    for offset in range(count):
        values = replacements.get(
            offset,
            (100.0, 100.0 + atr, 100.0, 100.0),
        )
        candle, displacement, _, group3 = harness.send(values)
        assert displacement.state is None
        assert displacement.transitions == ()
        assert group3.fair_value_gaps == ()
        assert group3.order_blocks == ()
        output.append(candle)
    return tuple(output)


def _frozen(state, fields: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(getattr(state, name) for name in fields)


def _fvg_by_id(output, entity_id: str):
    return next(
        state
        for state in output.fair_value_gaps
        if state.fvg_id == entity_id
    )


def _order_block_by_id(output, entity_id: str):
    return next(
        state
        for state in output.order_blocks
        if state.order_block_id == entity_id
    )


def _order_block_attempt(output):
    assert len(output.order_block_funnel) == 1
    return output.order_block_funnel[0]


def _confirmed_bos(
    *,
    clock: pd.Timestamp,
    direction: Direction,
    suffix: str = "one",
    timeframe: Timeframe = Timeframe.M5,
    lifecycle: BOSLifecycle = BOSLifecycle.CONFIRMED,
    resolved_at: pd.Timestamp | None = None,
    scope: BOSScope = BOSScope.CONTINUATION,
    pending_at: pd.Timestamp | None = None,
    source_displacement_id: str | None = None,
    mss_qualified: bool = False,
) -> BreakOfStructureState:
    resolution = (
        clock
        if lifecycle is not BOSLifecycle.PENDING
        and resolved_at is None
        else resolved_at
    )
    pending_reference = resolution or clock
    target = 102.0 if direction is Direction.LONG else 99.0
    return BreakOfStructureState(
        bos_id=f"bos-{suffix}",
        timeframe=timeframe,
        direction=direction,
        lifecycle=lifecycle,
        scope=scope,
        target_swing_id=f"swing-{suffix}",
        source_structure_id=(
            None if scope is BOSScope.LOCAL else f"structure-{suffix}"
        ),
        target_price=target,
        target_ticks=round(target / 0.25),
        pending_at=(
            pending_reference - pd.Timedelta(minutes=5)
            if pending_at is None
            else pending_at
        ),
        resolved_at=resolution,
        age_bars=1,
        failure_reason=(
            "superseded"
            if lifecycle is BOSLifecycle.FAILED
            else None
        ),
        strength=(
            0.8
            if lifecycle is BOSLifecycle.CONFIRMED
            else 0.0
        ),
        break_bar_id=(
            "current-break-bar"
            if lifecycle is BOSLifecycle.CONFIRMED
            else None
        ),
        break_distance_atr=(
            0.8 if lifecycle is BOSLifecycle.CONFIRMED else None
        ),
        source_displacement_id=source_displacement_id,
        mss_qualified=mss_qualified,
        post_break_state=(
            BOSPostBreakState.PENDING
            if lifecycle is BOSLifecycle.CONFIRMED
            else None
        ),
    )


def _bos_source(
    bos: BreakOfStructureState,
    *,
    candle: Candle,
    symbol: str | None = None,
    instrument_id: int | None = None,
    protocol_hash: str = STRUCTURE_PROTOCOL_SHA,
    tick_size: float = 0.25,
) -> Group3BOSSource:
    if (
        bos.lifecycle is BOSLifecycle.CONFIRMED
        and bos.break_bar_id == "current-break-bar"
    ):
        break_id = CausalGroup3Tracker(
            _group3_protocol()
        )._candle_id(candle)
        bos = replace(bos, break_bar_id=break_id)
    return Group3BOSSource(
        state=bos,
        symbol=candle.symbol if symbol is None else symbol,
        instrument_id=(
            candle.instrument_id
            if instrument_id is None
            else instrument_id
        ),
        protocol_hash=protocol_hash,
        tick_size=tick_size,
    )


def _form_fvg(direction: Direction):
    harness = _Harness()
    _warm(harness)
    if direction is Direction.LONG:
        pattern = (
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 101.5, 100.0, 101.5),
            (101.5, 102.25, 101.5, 102.25),
        )
    else:
        pattern = (
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.0, 98.5, 98.5),
            (98.5, 98.5, 97.75, 97.75),
        )
    candles = []
    updates = []
    output = None
    for values in pattern:
        candle, displacement, _, output = harness.send(values)
        candles.append(candle)
        updates.append(displacement)
    assert output is not None
    assert len(output.fvg_transitions) == 1
    state = output.fvg_transitions[0]
    assert state.lifecycle is FairValueGapLifecycle.OPEN
    return (
        harness,
        state,
        tuple(candles),
        tuple(updates),
        output,
    )


def _opposite_anchor(direction: Direction):
    return (
        (101.0, 102.0, 99.0, 100.75)
        if direction is Direction.LONG
        else (99.75, 102.0, 99.0, 100.0)
    )


def _older_opposite_anchor(direction: Direction):
    return (
        (100.5, 101.0, 100.0, 100.25)
        if direction is Direction.LONG
        else (100.25, 101.0, 100.0, 100.5)
    )


def _displacement_pair(direction: Direction):
    return (
        (
            (100.75, 102.0, 100.75, 102.0),
            (102.0, 102.75, 101.75, 102.75),
        )
        if direction is Direction.LONG
        else (
            (101.0, 101.0, 99.75, 99.75),
            (99.75, 100.0, 99.0, 99.0),
        )
    )


def _prepare_order_block_evidence(
    direction: Direction = Direction.LONG,
):
    harness = _Harness()
    history = _warm(
        harness,
        count=64,
        overrides={
            10: _older_opposite_anchor(direction),
            63: _opposite_anchor(direction),
        },
    )
    seed_values, active_values = _displacement_pair(direction)
    harness.send(seed_values)
    candle = _candle(harness.index, active_values)
    displacement = harness.displacement.on_completed_5m(candle)
    assert displacement.state is not None
    assert displacement.state.lifecycle is DisplacementLifecycle.ACTIVE
    harness.index += 1
    return harness, history[-1], candle, displacement


def _form_order_block(direction: Direction = Direction.LONG):
    harness, anchor, candle, displacement = (
        _prepare_order_block_evidence(direction)
    )
    bos = _confirmed_bos(
        clock=candle.end,
        direction=direction,
    )
    output = harness.group3.on_completed_5m(
        candle,
        displacement,
        (_bos_source(bos, candle=candle),),
    )
    assert len(output.order_block_transitions) == 1
    state = output.order_block_transitions[0]
    assert state.lifecycle is OrderBlockLifecycle.CREATED
    return harness, state, anchor, candle, displacement, bos, output


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("protocol_hash", "0" * 63),
        ("protocol_version", ""),
        ("tick_size", 0.0),
        ("timeframe", ""),
        ("fvg_source_bars", 0),
        ("fvg_formation_atr_period", 0),
        ("ob_anchor_history_bars", 0),
        ("maximum_fvg_states", 0),
        ("maximum_order_block_states", 0),
    ),
)
def test_group3_protocol_tracks_current_config_and_rejects_invalid_ranges(
    field: str,
    value: object,
) -> None:
    protocol = _group3_protocol()
    payload = json.loads(GROUP3_PROTOCOL_PATH.read_bytes())

    assert protocol.protocol_hash == GROUP3_PROTOCOL_SHA
    assert (
        protocol.protocol_version,
        protocol.tick_size,
        protocol.timeframe,
        protocol.fvg_source_bars,
        protocol.fvg_formation_atr_period,
        protocol.ob_anchor_history_bars,
        protocol.maximum_fvg_states,
        protocol.maximum_order_block_states,
    ) == (
        payload["protocol_version"],
        payload["tick_size"],
        payload["timeframe"],
        payload["fvg_source_bars"],
        payload["fvg_formation_atr_period"],
        payload["ob_anchor_history_bars"],
        payload["maximum_fvg_states"],
        payload["maximum_order_block_states"],
    )
    assert isinstance(protocol.protocol_version, str)
    assert protocol.protocol_version
    assert isinstance(protocol.tick_size, (int, float))
    assert not isinstance(protocol.tick_size, bool)
    assert protocol.tick_size > 0.0
    assert protocol.timeframe
    assert all(
        type(item) is int and item > 0
        for item in (
            protocol.fvg_source_bars,
            protocol.fvg_formation_atr_period,
            protocol.ob_anchor_history_bars,
            protocol.maximum_fvg_states,
            protocol.maximum_order_block_states,
        )
    )
    with pytest.raises(ValueError):
        replace(protocol, **{field: value})


def test_group3_requires_frozen_protocol_and_completed_tick_grid() -> None:
    with pytest.raises(TypeError, match="frozen Group 3 protocol"):
        CausalGroup3Tracker(object())

    base = _candle(0, (100.0, 101.0, 100.0, 100.0))
    invalid = (
        replace(base, timeframe=Timeframe.M1),
        replace(
            base,
            complete=False,
            observed_minutes=4,
            real_minutes=4,
        ),
        _candle(0, (100.1, 101.0, 100.0, 100.5)),
    )
    for candle in invalid:
        tracker = CausalGroup3Tracker(_group3_protocol())
        with pytest.raises(ValueError):
            tracker.on_completed_5m(
                candle,
                DisplacementUpdate(None),
            )


@pytest.mark.parametrize(
    ("direction", "bounds"),
    (
        (Direction.LONG, (100.5, 101.5)),
        (Direction.SHORT, (98.5, 99.5)),
    ),
)
def test_fvg_strict_three_bar_and_activation_on_c3(
    direction: Direction,
    bounds: tuple[float, float],
) -> None:
    harness, state, candles, updates, output = _form_fvg(direction)
    c1, c2, c3 = candles
    active = updates[-1]
    assert active.state is not None
    assert (
        active.state.lifecycle,
        active.state.active_at,
    ) == (DisplacementLifecycle.ACTIVE, c3.end)
    assert (
        state.direction,
        state.lower_bound,
        state.upper_bound,
        state.midpoint,
        state.formed_at,
        state.confirmed_at,
        state.age_bars,
        state.max_fill_fraction,
    ) == (
        direction,
        bounds[0],
        bounds[1],
        sum(bounds) / 2.0,
        c3.end,
        c3.end,
        0,
        0.0,
    )
    assert state.source_candle_starts == (
        c1.start,
        c2.start,
        c3.start,
    )
    assert state.qualification is FVGQualification.DISPLACEMENT_LINKED
    assert state.formation_atr > 0.0
    assert state.width_atr == pytest.approx(
        state.width_points / state.formation_atr
    )
    assert state.source_displacement_id == active.state.entity_id
    assert (
        state.source_active_transition_id
        == active.transitions[-1].transition_id
    )
    assert output.fvg_transitions == (state,)
    assert len(harness.group3.snapshot()[0]) == 1


@pytest.mark.parametrize(
    "pattern",
    (
        (
            (100.0, 101.0, 99.5, 100.0),
            (100.0, 101.25, 100.0, 101.25),
            (101.25, 102.0, 101.0, 102.0),
        ),
        (
            (100.0, 100.5, 99.0, 100.0),
            (100.0, 100.0, 98.75, 98.75),
            (98.75, 99.0, 98.0, 98.0),
        ),
    ),
    ids=("bullish-equality", "bearish-equality"),
)
def test_fvg_outer_bar_equality_does_not_form(pattern) -> None:
    harness = _Harness()
    _warm(harness)
    output = None
    for values in pattern:
        _, _, _, output = harness.send(values)
    assert output is not None
    assert output.fair_value_gaps == ()
    assert output.fvg_transitions == ()


def test_fvg_geometry_without_qualified_displacement_forms_raw() -> None:
    harness = _Harness()
    updates = []
    output = None
    for values in (
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 100.0, 101.5),
        (101.5, 102.25, 101.5, 102.25),
    ):
        _, displacement, _, output = harness.send(values)
        updates.append(displacement)
    assert all(update.state is None for update in updates)
    assert output is not None
    assert len(output.fair_value_gaps) == 1
    state = output.fair_value_gaps[0]
    assert state.qualification is FVGQualification.RAW
    assert state.source_displacement_id is None
    assert state.source_active_transition_id is None


def test_fvg_requires_exact_central_bar_episode_membership() -> None:
    harness = _Harness()
    _warm(harness)
    harness.send((100.0, 100.5, 99.5, 100.0))
    harness.send((100.0, 101.5, 100.0, 101.5))
    c3 = _candle(
        harness.index,
        (101.5, 102.25, 101.5, 102.25),
    )
    real = harness.displacement.on_completed_5m(c3)
    assert real.state is not None
    assert real.state.lifecycle is DisplacementLifecycle.ACTIVE
    foreign_state = replace(
        real.state,
        entity_id=f"foreign-{real.state.entity_id}",
    )
    foreign = DisplacementUpdate(
        state=foreign_state,
        transitions=tuple(
            replace(
                transition,
                transition_id=f"foreign-{transition.transition_id}",
                state=foreign_state,
            )
            for transition in real.transitions
        ),
    )
    output = harness.group3.on_completed_5m(c3, foreign)
    assert len(output.fair_value_gaps) == 1
    assert output.fair_value_gaps[0].qualification is FVGQualification.RAW
    assert output.fair_value_gaps[0].source_displacement_id is None


@pytest.mark.parametrize("case", ("cross-contract", "future-clock"))
def test_group3_rejects_displacement_provenance_mismatch(
    case: str,
) -> None:
    harness = _Harness()
    _warm(harness)
    harness.send((100.0, 100.5, 99.5, 100.0))
    harness.send((100.0, 101.5, 100.0, 101.5))
    candle = _candle(
        harness.index,
        (101.5, 102.25, 101.5, 102.25),
    )
    real = harness.displacement.on_completed_5m(candle)
    assert real.state is not None
    changes = (
        {"symbol": "ESH5"}
        if case == "cross-contract"
        else {
            "observed_at": candle.end + pd.Timedelta(minutes=5),
            "prefix_last_admitted_at": (
                candle.end + pd.Timedelta(minutes=5)
            ),
        }
    )
    invalid_state = replace(real.state, **changes)
    invalid = DisplacementUpdate(
        state=invalid_state,
        transitions=tuple(
            replace(transition, state=invalid_state)
            for transition in real.transitions
        ),
    )
    with pytest.raises(ValueError, match="provenance"):
        harness.group3.on_completed_5m(candle, invalid)
    assert harness.group3._failed is True


def test_fvg_late_activation_never_backfills_the_prior_gap() -> None:
    harness = _Harness()
    _warm(harness, atr=2.5)
    patterns = (
        (100.0, 101.0, 99.5, 100.0),
        (100.0, 101.5, 99.5, 101.25),
        (101.25, 101.5, 101.25, 101.5),
        (101.5, 101.75, 101.5, 101.75),
        (101.75, 103.0, 101.25, 103.0),
    )
    outputs = [harness.send(values)[-1] for values in patterns]
    first_raw = next(
        state
        for output in outputs
        for state in output.fair_value_gaps
        if state.qualification is FVGQualification.RAW
    )
    assert harness.displacement.snapshot() is not None
    assert (
        harness.displacement.snapshot().lifecycle
        is DisplacementLifecycle.ACTIVE
    )
    assert (
        harness.displacement.snapshot().active_at
        == BASE + pd.Timedelta(minutes=5 * harness.index)
    )
    retained = next(
        state
        for state in outputs[-1].fair_value_gaps
        if state.fvg_id == first_raw.fvg_id
    )
    assert retained.qualification is FVGQualification.RAW
    assert retained.source_displacement_id is None


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_fvg_open_partial_mitigated_and_frozen_geometry(
    direction: Direction,
) -> None:
    harness, created, _, _, _ = _form_fvg(direction)
    frozen = _frozen(created, _FVG_FROZEN_FIELDS)
    partial_values, mitigated_values = (
        (
            (102.25, 102.5, 101.0, 101.75),
            (101.75, 102.0, 100.5, 101.0),
        )
        if direction is Direction.LONG
        else (
            (97.75, 99.0, 97.5, 98.5),
            (98.5, 99.5, 98.25, 99.25),
        )
    )
    partial_output = harness.send(partial_values)[-1]
    partial = _fvg_by_id(partial_output, created.fvg_id)
    assert partial.lifecycle is FairValueGapLifecycle.PARTIAL
    assert partial.partial_at == partial.last_updated_at
    assert 0.0 < partial.max_fill_fraction < 1.0
    assert _frozen(partial, _FVG_FROZEN_FIELDS) == frozen

    mitigated_output = harness.send(mitigated_values)[-1]
    mitigated = _fvg_by_id(mitigated_output, created.fvg_id)
    assert mitigated.lifecycle is FairValueGapLifecycle.MITIGATED
    assert mitigated.mitigated_at == mitigated.last_updated_at
    assert mitigated.max_fill_fraction == 1.0
    assert _frozen(mitigated, _FVG_FROZEN_FIELDS) == frozen


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_fvg_emits_a_later_midpoint_crossing_without_rewriting_partial(
    direction: Direction,
) -> None:
    harness, created, _, _, _ = _form_fvg(direction)
    shallow, midpoint = (
        (
            (102.0, 102.25, 101.25, 101.75),
            (101.75, 102.0, 101.0, 101.25),
        )
        if direction is Direction.LONG
        else (
            (98.0, 98.75, 97.75, 98.25),
            (98.25, 99.0, 98.0, 98.75),
        )
    )

    first_output = harness.send(shallow)[-1]
    first = _fvg_by_id(first_output, created.fvg_id)
    assert first.lifecycle is FairValueGapLifecycle.PARTIAL
    assert first.max_fill_fraction == pytest.approx(0.25)
    assert first.midpoint_touched_at is None

    midpoint_output = harness.send(midpoint)[-1]
    advanced = _fvg_by_id(midpoint_output, created.fvg_id)
    transition = next(
        item
        for item in midpoint_output.fvg_transitions
        if item.fvg_id == created.fvg_id
    )
    assert advanced.lifecycle is FairValueGapLifecycle.PARTIAL
    assert advanced.partial_at == first.partial_at
    assert advanced.midpoint_touched_at == transition.last_updated_at
    assert transition.transition_reason == "midpoint_touched"
    assert transition.max_fill_fraction >= 0.5


@pytest.mark.parametrize(
    ("direction", "values", "expected"),
    (
        (
            Direction.LONG,
            (102.25, 102.5, 100.25, 100.25),
            FairValueGapLifecycle.INVALIDATED,
        ),
        (
            Direction.LONG,
            (102.25, 102.5, 100.5, 100.5),
            FairValueGapLifecycle.MITIGATED,
        ),
        (
            Direction.SHORT,
            (97.75, 99.75, 97.5, 99.75),
            FairValueGapLifecycle.INVALIDATED,
        ),
        (
            Direction.SHORT,
            (97.75, 99.5, 97.5, 99.5),
            FairValueGapLifecycle.MITIGATED,
        ),
    ),
    ids=(
        "long-close-through-priority",
        "long-far-edge-equality",
        "short-close-through-priority",
        "short-far-edge-equality",
    ),
)
def test_fvg_same_bar_terminal_priority(
    direction: Direction,
    values: tuple[float, float, float, float],
    expected: FairValueGapLifecycle,
) -> None:
    harness, created, _, _, _ = _form_fvg(direction)
    output = harness.send(values)[-1]
    terminal = _fvg_by_id(output, created.fvg_id)
    assert terminal.lifecycle is expected
    assert (
        terminal.invalidated_at is not None
        if expected is FairValueGapLifecycle.INVALIDATED
        else terminal.mitigated_at is not None
    )
    assert _frozen(terminal, _FVG_FROZEN_FIELDS) == _frozen(
        created,
        _FVG_FROZEN_FIELDS,
    )


@pytest.mark.parametrize("reason", tuple(sorted(FVG_BOUNDARY_REASONS)))
def test_fvg_hard_boundary_terminalizes_live_zone(reason: str) -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    output = harness.group3.on_boundary(
        reason,
        created.confirmed_at + pd.Timedelta(minutes=5),
    )
    assert len(output.fvg_transitions) == 1
    terminal = _fvg_by_id(output, created.fvg_id)
    assert (
        next(
            state
            for state in harness.group3.snapshot()[0]
            if state.fvg_id == created.fvg_id
        )
        == terminal
    )
    assert (
        terminal.lifecycle,
        terminal.transition_reason,
    ) == (FairValueGapLifecycle.INVALIDATED, reason)
    assert _frozen(terminal, _FVG_FROZEN_FIELDS) == _frozen(
        created,
        _FVG_FROZEN_FIELDS,
    )


@pytest.mark.parametrize("reason", tuple(sorted(WINDOW_RESET_REASONS)))
def test_fvg_window_boundary_preserves_live_zone_without_aging(
    reason: str,
) -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    output = harness.group3.on_boundary(
        reason,
        created.confirmed_at + pd.Timedelta(minutes=5),
    )
    assert _fvg_by_id(output, created.fvg_id) == created
    assert output.fvg_transitions == ()
    assert tuple(harness.group3._history) == ()
    assert harness.group3._episode_membership == {}


def test_soft_boundary_retains_contract_identity() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    boundary_clock = created.confirmed_at + pd.Timedelta(minutes=5)
    harness.group3.on_boundary(
        "registered_session_reset",
        boundary_clock,
    )
    before = harness.group3.snapshot()
    foreign = _candle(
        harness.index + 1,
        (102.25, 102.5, 102.0, 102.25),
        symbol="ESH5",
        instrument_id=2,
    )
    with pytest.raises(ValueError, match="contract changed"):
        harness.group3.on_completed_5m(
            foreign,
            DisplacementUpdate(None),
        )
    assert harness.group3.snapshot() == before


def test_boundary_exact_retry_and_same_clock_conflict() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    clock = created.confirmed_at + pd.Timedelta(minutes=5)
    first = harness.group3.on_boundary(
        "registered_session_reset",
        clock,
    )
    assert (
        harness.group3.on_boundary(
            "registered_session_reset",
            clock,
        )
        is first
    )
    before = harness.group3.snapshot()
    with pytest.raises(ValueError, match="out of order"):
        harness.group3.on_boundary(
            "synthetic_interruption",
            clock,
        )
    assert harness.group3.snapshot() == before


@pytest.mark.parametrize(
    "synthetic",
    (False, True),
    ids=("real-candle", "synthetic-boundary"),
)
def test_group3_rejects_off_grid_before_any_state_change(
    synthetic: bool,
) -> None:
    harness = _Harness()
    harness.send((100.0, 100.5, 99.5, 100.0))
    malformed = _candle(
        harness.index,
        (100.0, 100.5, 99.75, 100.1),
        real_minutes=0 if synthetic else 5,
        synthetic_minutes=5 if synthetic else 0,
    )
    before = pickle.dumps(harness.group3)

    with pytest.raises(ValueError, match="off-grid"):
        harness.group3.on_completed_5m(
            malformed,
            DisplacementUpdate(None),
        )

    assert pickle.dumps(harness.group3) == before
    assert harness.group3._failed is False


def test_group3_rejects_exact_retry_from_different_stored_grid() -> None:
    harness = _Harness()
    candle, displacement, _, output = harness.send(
        (100.0, 100.5, 99.5, 100.0)
    )
    retry = replace(
        candle,
        price_tick_size=0.5,
        normalized_ohlc_ticks=None,
    )
    assert retry == candle
    before = pickle.dumps(harness.group3)

    with pytest.raises(ValueError, match="grid disagrees"):
        harness.group3.on_completed_5m(retry, displacement)

    assert pickle.dumps(harness.group3) == before
    assert harness.group3._last_output is output
    assert harness.group3._failed is False


def test_fvg_synthetic_boundary_cannot_touch_or_age_live_zone() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    _, displacement, _, output = harness.send(
        (102.25, 102.5, 100.25, 100.25),
        real_minutes=0,
        synthetic_minutes=5,
    )
    assert displacement.transitions
    assert (
        displacement.transitions[-1].state.lifecycle,
        displacement.transitions[-1].state.terminal_reason,
    ) == (
        DisplacementLifecycle.CENSORED,
        "synthetic_interruption",
    )
    assert _fvg_by_id(output, created.fvg_id) == created
    assert output.fvg_transitions == ()


def test_contract_change_censored_boundary_requires_new_identity() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    candle = _candle(
        harness.index,
        (102.25, 102.5, 102.0, 102.25),
        symbol="ESH5",
        instrument_id=2,
    )
    displacement = harness.displacement.on_completed_5m(candle)
    assert (
        displacement.transitions[-1].state.terminal_reason
        == "contract_change_history_reset"
    )

    output = harness.group3.on_completed_5m(
        candle,
        displacement,
    )
    terminal = _fvg_by_id(output, created.fvg_id)
    assert terminal.lifecycle is FairValueGapLifecycle.INVALIDATED
    assert terminal.transition_reason == "contract_change_reset"


def test_foreign_synthetic_boundary_without_transition_is_rejected() -> None:
    harness = _Harness()
    harness.send((100.0, 100.25, 99.75, 100.0))
    candle = _candle(
        harness.index,
        (100.0, 100.25, 99.75, 100.0),
        symbol="ESH5",
        instrument_id=2,
        real_minutes=0,
        synthetic_minutes=5,
    )
    displacement = harness.displacement.on_completed_5m(candle)
    assert displacement == DisplacementUpdate(None)
    before = harness.group3.snapshot()

    with pytest.raises(ValueError, match="boundary candle identity"):
        harness.group3.on_completed_5m(candle, displacement)
    assert harness.group3.snapshot() == before


@pytest.mark.parametrize("case", ("protocol", "future-clock"))
def test_censored_boundary_requires_exact_displacement_provenance(
    case: str,
) -> None:
    harness, _, _, _, _ = _form_fvg(Direction.LONG)
    candle = _candle(
        harness.index,
        (102.25, 102.5, 100.25, 100.25),
        real_minutes=0,
        synthetic_minutes=5,
    )
    valid = harness.displacement.on_completed_5m(candle)
    terminal = valid.transitions[-1]
    if case == "protocol":
        invalid_state = replace(
            terminal.state,
            protocol_hash="f" * 64,
        )
    else:
        future = candle.end + pd.Timedelta(minutes=5)
        invalid_state = replace(
            terminal.state,
            state_started_at=future,
            last_updated_at=future,
            observed_at=future,
            terminal_at=future,
        )
    invalid = DisplacementUpdate(
        None,
        (replace(terminal, state=invalid_state),),
    )
    before = harness.group3.snapshot()
    with pytest.raises(ValueError, match="boundary provenance|protocol"):
        harness.group3.on_completed_5m(
            candle,
            invalid,
        )
    assert harness.group3.snapshot() == before


def test_censored_soft_boundary_rejects_foreign_candle_identity() -> None:
    harness, _, _, _, _ = _form_fvg(Direction.LONG)
    candle = _candle(
        harness.index,
        (102.25, 102.5, 100.25, 100.25),
        symbol="ESH5",
        instrument_id=2,
        real_minutes=0,
        synthetic_minutes=5,
    )
    displacement = harness.displacement.on_completed_5m(candle)
    assert displacement.transitions
    before = harness.group3.snapshot()

    with pytest.raises(ValueError, match="boundary candle identity"):
        harness.group3.on_completed_5m(
            candle,
            displacement,
        )
    assert harness.group3.snapshot() == before


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_order_block_uses_latest_reverse_bar_and_new_confirmed_bos(
    direction: Direction,
) -> None:
    (
        harness,
        state,
        anchor,
        creation_candle,
        displacement,
        bos,
        output,
    ) = _form_order_block(direction)
    assert displacement.state is not None
    assert (
        state.lifecycle,
        state.direction,
        state.anchor_start,
        state.anchor_end,
        state.lower_bound,
        state.upper_bound,
        state.midpoint,
        state.formed_at,
        state.age_bars,
        state.first_test_at,
    ) == (
        OrderBlockLifecycle.CREATED,
        direction,
        anchor.start,
        anchor.end,
        anchor.low,
        anchor.high,
        (anchor.low + anchor.high) / 2.0,
        creation_candle.end,
        0,
        None,
    )
    assert state.source_displacement_id == displacement.state.entity_id
    assert (
        state.source_active_transition_id
        == displacement.transitions[-1].transition_id
    )
    assert (
        state.source_bos_id,
        state.source_bos_protocol_hash,
        state.source_bos_resolved_at,
    ) == (bos.bos_id, STRUCTURE_PROTOCOL_SHA, creation_candle.end)
    assert (
        creation_candle.low <= state.upper_bound
        and creation_candle.high >= state.lower_bound
    )
    assert output.order_block_transitions == (state,)
    attempt = _order_block_attempt(output)
    assert attempt.observed_at == creation_candle.end
    assert attempt.outcome is OrderBlockAttemptOutcome.CREATED
    assert attempt.stages == tuple(
        (name, 1) for name in ORDER_BLOCK_FUNNEL_STAGES
    )

    retried = harness.group3.on_completed_5m(
        creation_candle,
        displacement,
        (_bos_source(bos, candle=creation_candle),),
    )
    assert retried is output
    assert _order_block_by_id(retried, state.order_block_id) == state


def test_real_structure_displacement_ids_join_without_fixture_rewrite() -> None:
    structure_config = StructureConfig.from_file(
        STRUCTURE_PROTOCOL_PATH
    )
    structure = StructureTracker(Timeframe.M5, structure_config)
    displacement_protocol = _displacement_protocol()
    displacement_tracker = CausalDisplacementTracker(
        displacement_protocol
    )
    group3 = CausalGroup3Tracker(
        _group3_protocol(),
        displacement_protocol_hash=displacement_protocol.protocol_hash,
        structure_protocol_hash=structure_config.protocol_hash,
    )
    highs = (
        101, 102, 104, 103, 102, 102, 102, 102, 102, 103,
        106, 104, 103, 103, 103, 103, 103, 103, 104, 107,
    )
    lows = (
        99, 99, 99, 99, 99, 98, 96, 97, 98, 98,
        98, 98, 98, 99, 97, 98, 99, 100, 101, 104,
    )
    opens = (*([100.0] * 17), 102.0, 101.0, 104.0)
    closes = (*([100.0] * 17), 101.0, 104.0, 107.0)
    final_displacement = None
    output = None
    candles = []
    for index, values in enumerate(zip(opens, highs, lows, closes)):
        candle = _candle(index, tuple(float(value) for value in values))
        candles.append(candle)
        structure.on_candle(candle)
        final_displacement = displacement_tracker.on_completed_5m(candle)
        confirmed_now = tuple(
            state
            for state in structure.snapshot()[2]
            if state.lifecycle is BOSLifecycle.CONFIRMED
            and state.resolved_at == candle.end
        )
        sources = tuple(
            Group3BOSSource(
                state=state,
                symbol=candle.symbol,
                instrument_id=candle.instrument_id,
                protocol_hash=structure_config.protocol_hash,
                tick_size=structure_config.tick_size,
            )
            for state in confirmed_now
        )
        output = group3.on_completed_5m(
            candle,
            final_displacement,
            sources,
        )

    assert output is not None
    assert final_displacement is not None
    assert final_displacement.state is not None
    assert len(output.order_block_transitions) == 1
    order_block = output.order_block_transitions[0]
    assert (
        order_block.source_bos_break_bar_id
        == final_displacement.state.last_valid_candle_id
    )
    assert order_block.source_bos_break_bar_id in (
        final_displacement.state.admitted_candle_ids
    )
    assert order_block.anchor_end == candles[17].end
    assert (
        order_block.source_bos_pending_at
        <= order_block.source_displacement_started_at
    )


def test_order_block_does_not_require_complete_64_bar_window() -> None:
    harness = _Harness()
    _warm(
        harness,
        count=63,
        overrides={62: _opposite_anchor(Direction.LONG)},
    )
    seed, active = _displacement_pair(Direction.LONG)
    harness.send(seed)

    def bos_factory(candle, _):
        return (
            _confirmed_bos(
                clock=candle.end,
                direction=Direction.LONG,
            ),
        )

    output = harness.send(active, bos_factory=bos_factory)[-1]
    assert len(output.order_block_transitions) == 1
    assert (
        output.order_block_transitions[0].anchor_end
        == output.order_block_transitions[0].source_displacement_started_at
        - pd.Timedelta(minutes=5)
    )


def test_order_block_never_expands_seed_window_to_bar_65() -> None:
    harness = _Harness()
    _warm(
        harness,
        count=65,
        overrides={0: _opposite_anchor(Direction.LONG)},
    )
    harness.send((100.0, 101.25, 100.0, 101.25))

    def bos_factory(candle, _):
        return (
            _confirmed_bos(
                clock=candle.end,
                direction=Direction.LONG,
            ),
        )

    output = harness.send(
        (101.25, 102.0, 101.0, 102.0),
        bos_factory=bos_factory,
    )[-1]
    assert output.order_blocks == ()
    assert output.order_block_transitions == ()
    attempt = _order_block_attempt(output)
    assert (
        attempt.outcome
        is OrderBlockAttemptOutcome.REVERSE_ANCHOR_CLUSTER_MISSING
    )
    assert dict(attempt.stages)[
        "break_bar_belongs_to_displacement"
    ] == 1
    assert dict(attempt.stages)["reverse_anchor_cluster_found"] == 0


def test_order_block_started_displacement_is_not_qualified() -> None:
    harness = _Harness()
    _warm(
        harness,
        count=64,
        overrides={63: _opposite_anchor(Direction.LONG)},
    )

    def bos_factory(candle, _):
        return (
            _confirmed_bos(
                clock=candle.end,
                direction=Direction.LONG,
            ),
        )

    output = harness.send(
        _displacement_pair(Direction.LONG)[0],
        bos_factory=bos_factory,
    )[-1]
    assert harness.displacement.snapshot() is not None
    assert (
        harness.displacement.snapshot().lifecycle
        is DisplacementLifecycle.STARTED
    )
    assert output.order_blocks == ()
    attempt = _order_block_attempt(output)
    assert (
        attempt.outcome
        is OrderBlockAttemptOutcome.NO_ACTIVE_DISPLACEMENT
    )
    assert dict(attempt.stages) == dict.fromkeys(
        ORDER_BLOCK_FUNNEL_STAGES,
        0,
    )


@pytest.mark.parametrize(
    "case",
    (
        "pending",
        "failed",
        "wrong-direction",
        "stale-clock",
    ),
)
def test_order_block_bos_direction_and_clock_gates_fail_closed(
    case: str,
) -> None:
    harness, _, candle, displacement = _prepare_order_block_evidence()
    if case == "pending":
        bos = _confirmed_bos(
            clock=candle.end,
            direction=Direction.LONG,
            lifecycle=BOSLifecycle.PENDING,
        )
    elif case == "failed":
        bos = _confirmed_bos(
            clock=candle.end,
            direction=Direction.LONG,
            lifecycle=BOSLifecycle.FAILED,
        )
    elif case == "wrong-direction":
        bos = _confirmed_bos(
            clock=candle.end,
            direction=Direction.SHORT,
        )
    elif case == "stale-clock":
        bos = _confirmed_bos(
            clock=candle.end,
            direction=Direction.LONG,
            resolved_at=candle.start,
        )
    output = harness.group3.on_completed_5m(
        candle,
        displacement,
        (_bos_source(bos, candle=candle),),
    )
    assert output.order_blocks == ()
    assert output.order_block_transitions == ()
    attempt = _order_block_attempt(output)
    assert attempt.outcome is OrderBlockAttemptOutcome.NO_COMPATIBLE_BOS
    assert dict(attempt.stages)["active_displacement"] == 1
    assert dict(attempt.stages)["compatible_bos"] == 0


def test_order_block_rejects_future_or_wrong_timeframe_bos_source() -> None:
    harness, _, candle, displacement = _prepare_order_block_evidence()
    future = _confirmed_bos(
        clock=candle.end,
        direction=Direction.LONG,
        resolved_at=candle.end + pd.Timedelta(minutes=5),
    )
    before = harness.group3.snapshot()
    with pytest.raises(ValueError, match="BOS provenance"):
        harness.group3.on_completed_5m(
            candle,
            displacement,
            (_bos_source(future, candle=candle),),
        )
    assert harness.group3.snapshot() == before

    wrong_timeframe = _confirmed_bos(
        clock=candle.end,
        direction=Direction.LONG,
        timeframe=Timeframe.M1,
    )
    with pytest.raises(ValueError, match="contract-bound BOS"):
        _bos_source(wrong_timeframe, candle=candle)


def test_order_block_funnel_separates_compatible_bos_from_membership() -> None:
    harness, _, candle, displacement = _prepare_order_block_evidence()
    source = _bos_source(
        _confirmed_bos(
            clock=candle.end,
            direction=Direction.LONG,
        ),
        candle=candle,
    )
    source = replace(
        source,
        state=replace(
            source.state,
            break_bar_id="foreign-completed-break-bar",
        ),
    )

    output = harness.group3.on_completed_5m(
        candle,
        displacement,
        (source,),
    )

    attempt = _order_block_attempt(output)
    assert (
        attempt.outcome
        is OrderBlockAttemptOutcome.BREAK_BAR_NOT_IN_DISPLACEMENT
    )
    assert dict(attempt.stages) == {
        "active_displacement": 1,
        "compatible_bos": 1,
        "break_bar_belongs_to_displacement": 0,
        "reverse_anchor_cluster_found": 0,
        "unique_eligible_bos": 0,
        "ob_created": 0,
    }


@pytest.mark.parametrize(
    ("case", "changes", "match"),
    (
        (
            "contract",
            {"symbol": "ESH5"},
            "BOS provenance",
        ),
        (
            "protocol",
            {"protocol_hash": "f" * 64},
            "structure protocol provenance",
        ),
        (
            "tick-grid",
            {"tick_size": 0.5},
            "BOS provenance",
        ),
    ),
)
def test_order_block_rejects_cross_source_bos_provenance(
    case: str,
    changes: dict[str, object],
    match: str,
) -> None:
    del case
    harness, _, candle, displacement = _prepare_order_block_evidence()
    bos = _confirmed_bos(
        clock=candle.end,
        direction=Direction.LONG,
    )
    before = harness.group3.snapshot()
    source = _bos_source(bos, candle=candle, **changes)
    with pytest.raises(ValueError, match=match):
        harness.group3.on_completed_5m(
            candle,
            displacement,
            (source,),
        )
    assert harness.group3.snapshot() == before


def test_order_block_ambiguous_bos_does_not_consume_displacement() -> None:
    harness, _, candle, displacement = _prepare_order_block_evidence()
    ambiguous = (
        _confirmed_bos(
            clock=candle.end,
            direction=Direction.LONG,
            suffix="left",
        ),
        _confirmed_bos(
            clock=candle.end,
            direction=Direction.LONG,
            suffix="right",
        ),
    )
    first = harness.group3.on_completed_5m(
        candle,
        displacement,
        tuple(_bos_source(bos, candle=candle) for bos in ambiguous),
    )
    assert first.order_blocks == ()
    attempt = _order_block_attempt(first)
    assert (
        attempt.outcome
        is OrderBlockAttemptOutcome.DUPLICATE_ELIGIBLE_BOS
    )
    assert dict(attempt.stages) == {
        "active_displacement": 1,
        "compatible_bos": 2,
        "break_bar_belongs_to_displacement": 2,
        "reverse_anchor_cluster_found": 1,
        "unique_eligible_bos": 0,
        "ob_created": 0,
    }

    next_candle = _candle(
        harness.index,
        (102.75, 103.5, 102.5, 103.5),
    )
    next_displacement = harness.displacement.on_completed_5m(
        next_candle
    )
    later = harness.group3.on_completed_5m(
        next_candle,
        next_displacement,
        (
            _bos_source(
                    _confirmed_bos(
                        clock=next_candle.end,
                        direction=Direction.LONG,
                        suffix="later",
                        pending_at=displacement.state.started_at,
                    ),
                candle=next_candle,
            ),
        ),
    )
    assert len(later.order_block_transitions) == 1
    assert (
        later.order_block_transitions[0].lifecycle
        is OrderBlockLifecycle.CREATED
    )


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_order_block_created_untested_mitigated_and_frozen_geometry(
    direction: Direction,
) -> None:
    harness, created, _, _, _, _, _ = _form_order_block(direction)
    frozen = _frozen(created, _ORDER_BLOCK_FROZEN_FIELDS)
    no_touch, touch = (
        (
            (102.75, 103.5, 102.25, 103.5),
            (103.5, 103.75, 101.75, 103.0),
        )
        if direction is Direction.LONG
        else (
            (98.75, 98.75, 98.0, 98.0),
            (98.0, 99.25, 97.75, 98.25),
        )
    )
    untested_output = harness.send(no_touch)[-1]
    untested = _order_block_by_id(
        untested_output,
        created.order_block_id,
    )
    assert (
        untested.lifecycle,
        untested.age_bars,
        untested.first_test_at,
    ) == (OrderBlockLifecycle.UNTESTED, 1, None)
    assert _frozen(untested, _ORDER_BLOCK_FROZEN_FIELDS) == frozen

    mitigated_output = harness.send(touch)[-1]
    mitigated = _order_block_by_id(
        mitigated_output,
        created.order_block_id,
    )
    assert mitigated.lifecycle is OrderBlockLifecycle.MITIGATED
    assert (
        mitigated.first_test_at,
        mitigated.mitigated_at,
    ) == (mitigated.last_updated_at,) * 2
    assert _frozen(mitigated, _ORDER_BLOCK_FROZEN_FIELDS) == frozen


@pytest.mark.parametrize(
    ("direction", "values", "expected"),
    (
        (
            Direction.LONG,
            (102.75, 103.0, 98.75, 98.75),
            OrderBlockLifecycle.FAILED,
        ),
        (
            Direction.LONG,
            (102.75, 103.0, 99.0, 99.0),
            OrderBlockLifecycle.MITIGATED,
        ),
        (
            Direction.SHORT,
            (99.0, 102.25, 98.75, 102.25),
            OrderBlockLifecycle.FAILED,
        ),
        (
            Direction.SHORT,
            (99.0, 102.0, 98.75, 102.0),
            OrderBlockLifecycle.MITIGATED,
        ),
    ),
    ids=(
        "long-failure-priority",
        "long-distal-equality",
        "short-failure-priority",
        "short-distal-equality",
    ),
)
def test_order_block_same_bar_failure_priority(
    direction: Direction,
    values: tuple[float, float, float, float],
    expected: OrderBlockLifecycle,
) -> None:
    harness, created, _, _, _, _, _ = _form_order_block(direction)
    output = harness.send(values)[-1]
    terminal = _order_block_by_id(
        output,
        created.order_block_id,
    )
    assert terminal.lifecycle is expected
    assert (
        terminal.failed_at is not None
        if expected is OrderBlockLifecycle.FAILED
        else terminal.mitigated_at is not None
    )
    assert _frozen(terminal, _ORDER_BLOCK_FROZEN_FIELDS) == _frozen(
        created,
        _ORDER_BLOCK_FROZEN_FIELDS,
    )


def test_order_block_boundaries_preserve_or_fail_frozen_zone() -> None:
    live_harness, live, _, _, _, _, _ = _form_order_block()
    reset = live_harness.group3.on_boundary(
        "registered_session_reset",
        live.confirmed_at + pd.Timedelta(minutes=5),
    )
    assert _order_block_by_id(reset, live.order_block_id) == live
    assert reset.order_block_transitions == ()

    failed_harness, created, _, _, _, _, _ = _form_order_block()
    failed_output = failed_harness.group3.on_boundary(
        "data_gap_reset",
        created.confirmed_at + pd.Timedelta(minutes=5),
    )
    assert len(failed_output.order_block_transitions) == 1
    failed = _order_block_by_id(
        failed_output,
        created.order_block_id,
    )
    assert (
        next(
            state
            for state in failed_harness.group3.snapshot()[1]
            if state.order_block_id == created.order_block_id
        )
        == failed
    )
    assert (
        failed.lifecycle,
        failed.transition_reason,
    ) == (OrderBlockLifecycle.FAILED, "data_gap_reset")
    assert _frozen(failed, _ORDER_BLOCK_FROZEN_FIELDS) == _frozen(
        created,
        _ORDER_BLOCK_FROZEN_FIELDS,
    )


def test_exact_retry_requires_identical_upstream_provenance() -> None:
    harness, _, candles, updates, output = _form_fvg(Direction.LONG)
    c3 = candles[-1]
    exact = harness.group3.on_completed_5m(c3, updates[-1])
    assert exact is output
    before = harness.group3.snapshot()
    with pytest.raises(ValueError, match="duplicate or out-of-order"):
        harness.group3.on_completed_5m(
            c3,
            updates[-1],
            (
                _bos_source(
                    _confirmed_bos(
                        clock=c3.end,
                        direction=Direction.LONG,
                    ),
                    candle=c3,
                ),
            ),
        )
    assert harness.group3.snapshot() == before


def test_pickle_resume_matches_uninterrupted_lifecycle() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    harness.send((102.25, 102.5, 101.0, 101.75))
    resumed = pickle.loads(pickle.dumps(harness))
    assert resumed.group3.snapshot() == harness.group3.snapshot()
    assert resumed.displacement.snapshot() == harness.displacement.snapshot()

    values = (101.75, 102.0, 100.5, 101.0)
    left = harness.send(values)
    right = resumed.send(values)
    assert left == right
    assert harness.group3.snapshot() == resumed.group3.snapshot()
    terminal = _fvg_by_id(left[-1], created.fvg_id)
    assert terminal.lifecycle is FairValueGapLifecycle.MITIGATED


def test_capacity_evicts_only_previously_exposed_terminal_state() -> None:
    terminal_harness, created, _, _, _ = _form_fvg(Direction.LONG)
    terminal_output = terminal_harness.send(
        (102.25, 102.5, 100.5, 100.5)
    )[-1]
    terminal = _fvg_by_id(terminal_output, created.fvg_id)
    assert terminal.lifecycle is FairValueGapLifecycle.MITIGATED

    protocol = _group3_protocol()
    compactable = CausalGroup3Tracker(protocol)
    for index in range(protocol.maximum_fvg_states):
        entity_id = f"terminal-fvg-{index:03d}"
        terminal_clock = terminal.mitigated_at + pd.Timedelta(
            minutes=0 if index == 200 else 5
        )
        compactable._fair_value_gaps[entity_id] = replace(
            terminal,
            fvg_id=entity_id,
            state_started_at=terminal_clock,
            last_updated_at=terminal_clock,
            mitigated_at=terminal_clock,
        )
        compactable._fvg_order.append(entity_id)
        compactable._exposed_terminal_ids.add(entity_id)
    compactable._admit_capacity(
        states=compactable._fair_value_gaps,
        order=compactable._fvg_order,
        maximum=protocol.maximum_fvg_states,
        terminal=compactable._is_fvg_terminal,
    )
    assert len(compactable._fair_value_gaps) == (
        protocol.maximum_fvg_states - 1
    )
    assert "terminal-fvg-200" not in compactable._fair_value_gaps

    blocked = CausalGroup3Tracker(protocol)
    for index in range(protocol.maximum_fvg_states):
        entity_id = f"live-fvg-{index:03d}"
        blocked._fair_value_gaps[entity_id] = replace(
            created,
            fvg_id=entity_id,
        )
        blocked._fvg_order.append(entity_id)
    with pytest.raises(RuntimeError, match="cannot admit"):
        blocked._admit_capacity(
            states=blocked._fair_value_gaps,
            order=blocked._fvg_order,
            maximum=protocol.maximum_fvg_states,
            terminal=blocked._is_fvg_terminal,
        )
    assert blocked._failed is True
    assert len(blocked._fair_value_gaps) == (
        protocol.maximum_fvg_states
    )


def test_failed_completed_bar_update_is_transactional() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    protocol = _group3_protocol()
    for index in range(1, protocol.maximum_fvg_states):
        entity_id = f"live-capacity-fvg-{index:03d}"
        harness.group3._fair_value_gaps[entity_id] = replace(
            created,
            fvg_id=entity_id,
        )
        harness.group3._fvg_order.append(entity_id)
    before = harness.group3.snapshot()
    last_clock = harness.group3._last_clock
    with pytest.raises(RuntimeError, match="cannot admit"):
        harness.send((102.25, 103.0, 102.25, 103.0))
    assert harness.group3.snapshot() == before
    assert harness.group3._last_clock == last_clock
    assert harness.group3._failed is True
