from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import pickle
from typing import Any

import pandas as pd
import pytest

from smc_trader.foundation_registry import FOUNDATION_VERSION
from smc_trader.zone import ZoneRawOnlyStructureDisposition
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    Timeframe,
)
from smc_trader.observation import CausalObserver, ObserverConfig
from smc_trader.semantic_zones import (
    BaseOriginCore,
    CompatibleStructureKind,
    CompletedZoneBar,
    FVGAvailability,
    FVGStructuralLifecycle,
    FVGTerminationCause,
    ZoneEntrySide,
    ZoneFirstReinteractionTracker,
    ZoneFirstRetestSpec,
    ZoneObjectKind,
    bind_fvg_structural_context,
    qualify_order_block,
    reduce_fvg_termination,
)
from smc_trader.structural_outcome import (
    OutcomeBar,
    OutcomeTerminal,
    StructuralOutcomeEngine,
    StructuralOutcomeSpec,
    project_conservative_execution,
)
from tests.test_v3_displacement_replay import (
    CORE_TEST_SCALE_SPECS,
    GROUP12_PROTOCOL_PATH,
    GROUP3_PROTOCOL_PATH,
    PROTOCOL_PATH,
    _m5 as _observer_m5,
    _update as _observer_update,
)
from tests.test_zone_primitives import (
    _form_fvg,
    _form_order_block,
)


_START = pd.Timestamp("2024-06-03T14:30:00Z")


def _at(*, minutes: int = 0, seconds: int = 0) -> pd.Timestamp:
    return _START + pd.Timedelta(minutes * 60 + seconds, unit="s")


def _base_origin_core() -> BaseOriginCore:
    return BaseOriginCore(
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        source_displacement_id="disp-1",
        source_displacement_event_id="event-disp-1",
        anchor_bar_event_ids=("bar-anchor-1", "bar-anchor-2"),
        anchor_candle_ids=("candle-anchor-1", "candle-anchor-2"),
        anchor_completed_at=(_at(minutes=-10), _at(minutes=-5)),
        lower_bound=99.0,
        upper_bound=101.0,
        body_lower_bound=99.25,
        body_upper_bound=100.75,
        tick_size=0.25,
        formed_at=_at(minutes=-5),
        known_at=_START,
    )


def test_base_origin_core_rejects_a_gapped_anchor_cluster() -> None:
    values = _base_origin_core().__dict__.copy()
    values.pop("core_id")
    values["anchor_completed_at"] = (
        _at(minutes=-15),
        _at(minutes=-5),
    )

    with pytest.raises(ValueError, match="not contiguous"):
        BaseOriginCore(**values)


def _retest_spec() -> ZoneFirstRetestSpec:
    return ZoneFirstRetestSpec(
        object_kind=ZoneObjectKind.FVG,
        object_id="fvg-1",
        creation_event_id="event-fvg-1",
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        lower_bound=100.0,
        upper_bound=102.0,
        tick_size=0.25,
        object_created_at=_at(minutes=-5),
        object_known_at=_START,
        departure_confirmed_at=_START,
        departure_source_event_id="event-departure-1",
        creation_declared_departed=True,
    )


def _zone_bar(
    minutes: int,
    *,
    bar_event_id: str | None = None,
    open_: float = 103.0,
    high: float = 104.0,
    low: float = 102.25,
    close: float = 103.5,
    instrument_id: int = 123,
    timeframe: Timeframe = Timeframe.M5,
) -> CompletedZoneBar:
    return CompletedZoneBar(
        bar_event_id=bar_event_id or f"bar-{minutes}",
        symbol="ES",
        instrument_id=instrument_id,
        timeframe=timeframe,
        known_at=_at(minutes=minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
        session="RTH",
        context_event_ids=("context-parent-1",),
    )


def _fvg_lifecycle() -> FVGStructuralLifecycle:
    return FVGStructuralLifecycle(
        fvg_id="fvg-1",
        source_creation_event_id="event-fvg-1",
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        created_at=_at(minutes=-5),
        known_at=_START,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        context_source_event_ids=(
            "event-structure-generation-1",
            "event-structural-range-1",
        ),
    )


def _outcome_spec(
    *,
    horizon_bars: int = 3,
    horizon_seconds: int = 900,
    window_end_minutes: int = 60,
) -> StructuralOutcomeSpec:
    return StructuralOutcomeSpec(
        source_event_id="sweep-1",
        symbol="ES",
        instrument_id=123,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        observation_start_known_at=_START,
        observation_window_end_exclusive=_at(minutes=window_end_minutes),
        reference_price=100.0,
        target_price=102.0,
        invalidation_price=98.0,
        target_definition="plus_one_atr",
        invalidation_definition="minus_one_atr",
        atr_at_start=2.0,
        tick_size=0.25,
        horizon_bars=horizon_bars,
        horizon_seconds=horizon_seconds,
    )


def _outcome_bar(
    minutes: int,
    *,
    bar_event_id: str | None = None,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    instrument_id: int = 123,
    timeframe: Timeframe = Timeframe.M5,
) -> OutcomeBar:
    return OutcomeBar(
        bar_event_id=bar_event_id or f"outcome-bar-{minutes}",
        symbol="ES",
        instrument_id=instrument_id,
        timeframe=timeframe,
        known_at=_at(minutes=minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def _pending_foundation_bindings(tracker) -> dict[str, Any]:
    """Supply explicit stand-ins for the production canonical event maps."""

    bars: dict[str, str] = {}
    displacements: dict[str, str] = {}
    displacement_clocks: dict[str, pd.Timestamp] = {}
    structures: dict[str, str] = {}
    fvg_creations: dict[str, str] = {}
    fvg_terminals: dict[str, str] = {}
    order_block_creations: dict[str, str] = {}
    sessions: dict[pd.Timestamp, str] = {}

    def bind_bar(candle_id: str) -> None:
        bars[candle_id] = f"canonical-bar:{candle_id}"

    for completed in tracker._pending_foundation_completed:
        bind_bar(completed.candle_id)
        sessions[completed.candle.end] = "RTH"
        for candidate in completed.new_base_origin_candidates:
            for candle_id in candidate.cluster_ids:
                bind_bar(candle_id)
            identity = candidate.source_displacement_transition_identity
            displacements[identity] = f"canonical-displacement:{identity}"
        for seed in completed.new_qualified_order_blocks:
            candidate = seed.candidate
            for candle_id in candidate.cluster_ids:
                bind_bar(candle_id)
            displacement_identity = (
                candidate.source_displacement_transition_identity
            )
            displacements[displacement_identity] = (
                f"canonical-displacement:{displacement_identity}"
            )
            displacement_clocks[displacement_identity] = completed.candle.end
            active_identity = (
                seed.legacy_state.source_active_transition_id
            )
            displacements[active_identity] = (
                f"canonical-displacement:{active_identity}"
            )
            displacement_clocks[active_identity] = completed.candle.end
            structure_identity = seed.compatible_structure_entity_id
            structures[structure_identity] = (
                f"canonical-structure:{structure_identity}"
            )
            entity_id = seed.legacy_state.order_block_id
            order_block_creations[entity_id] = (
                f"canonical-origin-zone-created:{entity_id}"
            )
        for state in completed.new_fvgs:
            for candle_id in state.source_candle_ids:
                bind_bar(candle_id)
            fvg_creations[state.fvg_id] = (
                f"canonical-fvg-created:{state.fvg_id}"
            )
            if state.source_active_transition_id is not None:
                identity = state.source_active_transition_id
                displacements[identity] = (
                    f"canonical-displacement:{identity}"
                )
                displacement_clocks[identity] = completed.candle.end
        for fvg_id in completed.price_invalidated_fvg_ids:
            fvg_terminals[fvg_id] = (
                f"canonical-fvg-invalidated:{fvg_id}"
            )
    return {
        "bar_event_ids_by_candle_id": bars,
        "displacement_event_ids_by_identity": displacements,
        "displacement_event_known_at_by_identity": displacement_clocks,
        "structure_event_ids_by_entity": structures,
        "fvg_creation_event_ids_by_entity": fvg_creations,
        "fvg_terminal_event_ids_by_entity": fvg_terminals,
        "order_block_creation_event_ids_by_entity": (
            order_block_creations
        ),
        "sessions_by_clock": sessions,
    }


def _pickle_round_trip(value):
    return pickle.loads(pickle.dumps(value))


def _raw_only_disposition(
    legacy,
    *,
    bos_id: str | None = None,
) -> ZoneRawOnlyStructureDisposition:
    return ZoneRawOnlyStructureDisposition(
        bos_id=bos_id or legacy.source_bos_id,
        raw_break_event_id="canonical-raw-break",
        protected_assignment_event_id="canonical-protected-assignment",
        timeframe=legacy.timeframe,
        direction=legacy.direction,
        resolved_at=legacy.source_bos_resolved_at,
        source_structure_id=legacy.source_bos_structure_id,
        target_swing_id=legacy.source_bos_target_swing_id,
        break_bar_id=legacy.source_bos_break_bar_id,
        bos_source_displacement_id=legacy.source_displacement_id,
    )


def _production_observer() -> CausalObserver:
    return CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            structure_protocol=str(GROUP12_PROTOCOL_PATH),
            liquidity_protocol=str(GROUP12_PROTOCOL_PATH),
            displacement_protocol=str(PROTOCOL_PATH),
            zone_protocol=str(GROUP3_PROTOCOL_PATH),
        )
    )


def test_base_origin_core_freezes_existing_full_cluster_geometry() -> None:
    core = _base_origin_core()

    assert core.semantic_version == FOUNDATION_VERSION
    assert core.lower_bound == 99.0
    assert core.upper_bound == 101.0
    assert core.width_ticks == 8
    assert core.source_event_ids == (
        "event-disp-1",
        "bar-anchor-1",
        "bar-anchor-2",
    )
    assert core.core_id == _base_origin_core().core_id
    with pytest.raises(FrozenInstanceError):
        core.lower_bound = 98.0  # type: ignore[misc]


def test_base_origin_core_rejects_unordered_or_off_grid_ancestry() -> None:
    values = _base_origin_core().__dict__.copy()
    values.pop("core_id")
    values["anchor_completed_at"] = (_at(minutes=-5), _at(minutes=-10))
    with pytest.raises(ValueError, match="ordered one-to-one"):
        BaseOriginCore(**values)

    values["anchor_completed_at"] = (_at(minutes=-10), _at(minutes=-5))
    values["lower_bound"] = 99.1
    with pytest.raises(ValueError, match="off-grid"):
        BaseOriginCore(**values)


@pytest.mark.parametrize(
    "kind",
    (
        CompatibleStructureKind.QUALIFIED_BOS,
        CompatibleStructureKind.MSS_CORE_CONFIRMED,
    ),
)
def test_qualified_order_block_binds_exact_core_and_structure_fact(
    kind: CompatibleStructureKind,
) -> None:
    core = _base_origin_core()
    qualified = qualify_order_block(
        core,
        source_displacement_id="disp-1",
        source_displacement_event_id="event-disp-1",
        compatible_structure_event_id=f"event-{kind.value}",
        compatible_structure_kind=kind,
        qualified_at=_at(minutes=5),
        known_at=_at(minutes=5),
    )

    assert qualified.base_origin_core_id == core.core_id
    assert qualified.source_displacement_id == core.source_displacement_id
    assert qualified.compatible_structure_kind is kind
    assert (
        qualified.lower_bound,
        qualified.upper_bound,
        qualified.body_lower_bound,
        qualified.body_upper_bound,
    ) == (
        core.lower_bound,
        core.upper_bound,
        core.body_lower_bound,
        core.body_upper_bound,
    )


@pytest.mark.parametrize(
    ("displacement_id", "displacement_event_id"),
    (("sibling-disp", "event-disp-1"), ("disp-1", "sibling-event")),
)
def test_qualified_order_block_rejects_sibling_displacement_borrowing(
    displacement_id: str,
    displacement_event_id: str,
) -> None:
    with pytest.raises(ValueError, match="exact core"):
        qualify_order_block(
            _base_origin_core(),
            source_displacement_id=displacement_id,
            source_displacement_event_id=displacement_event_id,
            compatible_structure_event_id="event-qualified-bos",
            compatible_structure_kind=CompatibleStructureKind.QUALIFIED_BOS,
            qualified_at=_at(minutes=5),
            known_at=_at(minutes=5),
        )


def test_qualified_order_block_cannot_predate_its_core() -> None:
    with pytest.raises(ValueError, match="predates"):
        qualify_order_block(
            _base_origin_core(),
            source_displacement_id="disp-1",
            source_displacement_event_id="event-disp-1",
            compatible_structure_event_id="event-qualified-bos",
            compatible_structure_kind=CompatibleStructureKind.QUALIFIED_BOS,
            qualified_at=_at(minutes=-1),
            known_at=_at(minutes=5),
        )

    with pytest.raises(ValueError):
        qualify_order_block(
            _base_origin_core(),
            source_displacement_id="disp-1",
            source_displacement_event_id="event-disp-1",
            compatible_structure_event_id="event-raw-break",
            compatible_structure_kind="raw_boundary_break",  # type: ignore[arg-type]
            qualified_at=_at(minutes=5),
            known_at=_at(minutes=5),
        )


def test_first_retest_registration_supports_later_observed_departure() -> None:
    values = _retest_spec().__dict__.copy()
    values["creation_declared_departed"] = False
    values["departure_confirmed_at"] = None
    values["departure_source_event_id"] = None
    waiting = ZoneFirstRetestSpec(**values)
    assert waiting.creation_declared_departed is False

    values["creation_declared_departed"] = True
    values["departure_confirmed_at"] = _at(minutes=1)
    with pytest.raises(ValueError, match="departure"):
        ZoneFirstRetestSpec(**values)

    values["departure_confirmed_at"] = _at(minutes=-10)
    with pytest.raises(ValueError, match="departure"):
        ZoneFirstRetestSpec(**values)


def test_first_retest_waits_for_future_departure_then_first_reentry() -> None:
    values = _retest_spec().__dict__.copy()
    values.update(
        creation_declared_departed=False,
        departure_confirmed_at=None,
        departure_source_event_id=None,
    )
    tracker = ZoneFirstReinteractionTracker(ZoneFirstRetestSpec(**values))

    still_inside = tracker.on_completed_bar(
        _zone_bar(
            5,
            open_=101.0,
            high=101.75,
            low=100.25,
            close=101.5,
        )
    )
    assert still_inside.departure_confirmed_at is None
    assert still_inside.first_retest is None

    departed = still_inside.on_completed_bar(
        _zone_bar(
            10,
            open_=101.5,
            high=103.0,
            low=101.25,
            close=102.5,
        )
    )
    assert departed.departure_confirmed_at == _at(minutes=10)
    assert departed.departure_source_event_id == "bar-10"
    assert departed.first_retest is None

    reentered = departed.on_completed_bar(
        _zone_bar(
            15,
            open_=103.0,
            high=103.25,
            low=102.0,
            close=102.5,
        )
    )
    assert reentered.first_retest is not None
    assert reentered.first_retest.departure_source_event_id == "bar-10"
    assert reentered.first_retest.source_bar_event_id == "bar-15"
    assert reentered.first_retest.age_bars == 3


def test_first_retest_is_strictly_later_and_boundary_equality_counts() -> None:
    tracker = ZoneFirstReinteractionTracker(_retest_spec())
    with pytest.raises(ValueError, match="strictly later"):
        tracker.on_completed_bar(
            _zone_bar(
                0,
                open_=101.0,
                high=102.0,
                low=100.0,
                close=101.0,
            )
        )

    untouched = tracker.on_completed_bar(_zone_bar(5))
    terminal = untouched.on_completed_bar(
        _zone_bar(
            10,
            open_=103.0,
            high=104.0,
            low=102.0,
            close=102.5,
        )
    )
    event = terminal.first_retest

    assert event is not None
    assert event.entry_side is ZoneEntrySide.FROM_ABOVE
    assert event.fill_fraction == 0.0
    assert event.age_bars == 2
    assert event.age_seconds == 600
    assert event.source_bar_event_id == "bar-10"
    assert event.source_event_ids == (
        "event-fvg-1",
        "event-departure-1",
        "bar-10",
    )
    assert event.session == "RTH"
    assert event.context_event_ids == ("context-parent-1",)


def test_first_retest_gap_open_inside_and_future_path_invariance() -> None:
    initial = ZoneFirstReinteractionTracker(_retest_spec())
    retest_bar = _zone_bar(
        5,
        open_=101.0,
        high=101.5,
        low=100.5,
        close=101.25,
    )
    prefix_result = initial.on_completed_bar(retest_bar)
    event = prefix_result.first_retest
    assert event is not None
    assert event.entry_side is ZoneEntrySide.GAP_OPENED_INSIDE
    assert event.fill_fraction == 0.75

    future_extreme = _zone_bar(
        10,
        open_=103.0,
        high=110.0,
        low=90.0,
        close=100.0,
    )
    full_path_result = prefix_result.on_completed_bar(future_extreme)
    assert full_path_result is prefix_result
    assert full_path_result.first_retest is event
    assert set(event.__dict__).isdisjoint(
        {
            "eventual_midpoint_touch",
            "eventual_full_fill",
            "eventual_invalidation",
            "eventual_continuation",
        }
    )


def test_first_retest_rejects_non_native_timeframe() -> None:
    with pytest.raises(ValueError, match="scope differs"):
        ZoneFirstReinteractionTracker(_retest_spec()).on_completed_bar(
            _zone_bar(5, timeframe=Timeframe.M1)
        )


def test_first_retest_rejects_a_missing_native_bar_before_later_contact() -> None:
    tracker = ZoneFirstReinteractionTracker(_retest_spec()).on_completed_bar(
        _zone_bar(5)
    )

    with pytest.raises(ValueError, match="not contiguous"):
        tracker.on_completed_bar(
            _zone_bar(
                15,
                open_=103.0,
                high=104.0,
                low=102.0,
                close=102.5,
            )
        )


def test_first_retest_advances_across_registered_memorial_closure() -> None:
    memorial_close = pd.Timestamp("2024-05-27 13:00", tz="America/New_York")
    prior_completion = memorial_close - pd.Timedelta(5, unit="min")
    spec = replace(
        _retest_spec(),
        object_created_at=prior_completion - pd.Timedelta(5, unit="min"),
        object_known_at=prior_completion,
        departure_confirmed_at=prior_completion,
    )
    at_close = replace(
        _zone_bar(5),
        bar_event_id="memorial-close-m5",
        known_at=memorial_close,
    )
    first_reopen = replace(
        _zone_bar(10),
        bar_event_id="memorial-reopen-m5",
        known_at=pd.Timestamp(
            "2024-05-27 18:05",
            tz="America/New_York",
        ),
    )

    tracker = ZoneFirstReinteractionTracker(spec).on_completed_bar(at_close)
    advanced = tracker.on_completed_bar(first_reopen)

    assert advanced.observed_native_bars == 2
    assert advanced.last_observed_at == first_reopen.known_at
    assert advanced.first_retest is None


def test_first_retest_rejects_skipping_first_memorial_reopen_bucket() -> None:
    memorial_close = pd.Timestamp("2024-05-27 13:00", tz="America/New_York")
    prior_completion = memorial_close - pd.Timedelta(5, unit="min")
    spec = replace(
        _retest_spec(),
        object_created_at=prior_completion - pd.Timedelta(5, unit="min"),
        object_known_at=prior_completion,
        departure_confirmed_at=prior_completion,
    )
    tracker = ZoneFirstReinteractionTracker(spec).on_completed_bar(
        replace(
            _zone_bar(5),
            bar_event_id="memorial-close-m5",
            known_at=memorial_close,
        )
    )

    with pytest.raises(ValueError, match="not contiguous"):
        tracker.on_completed_bar(
            replace(
                _zone_bar(15),
                bar_event_id="memorial-skipped-reopen-m5",
                known_at=pd.Timestamp(
                    "2024-05-27 18:10",
                    tz="America/New_York",
                ),
            )
        )


@pytest.mark.parametrize(
    ("cause", "availability"),
    (
        (
            FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE,
            FVGAvailability.INVALIDATED,
        ),
        (FVGTerminationCause.CONTRACT_ROLLOVER, FVGAvailability.EXPIRED),
        (FVGTerminationCause.SEMANTIC_RESET, FVGAvailability.EXPIRED),
        (FVGTerminationCause.DATA_GAP, FVGAvailability.CENSORED),
    ),
)
def test_fvg_terminal_causes_have_frozen_dispositions(
    cause: FVGTerminationCause,
    availability: FVGAvailability,
) -> None:
    state = _fvg_lifecycle()
    terminal = reduce_fvg_termination(
        state,
        cause=cause,
        known_at=_at(minutes=5),
        cause_event_ids=(f"event-{cause.value}",),
    )

    assert terminal.availability is availability
    assert terminal.terminal_reason is cause
    assert reduce_fvg_termination(
        terminal,
        cause=cause,
        known_at=_at(minutes=5),
        cause_event_ids=(f"event-{cause.value}",),
    ) is terminal


@pytest.mark.parametrize(
    ("cause", "related_entity_id"),
    (
        (
            FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
            "structure-generation-1",
        ),
        (
            FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
            "structural-range-1",
        ),
    ),
)
def test_fvg_structural_expiry_binds_exact_parent_or_range(
    cause: FVGTerminationCause,
    related_entity_id: str,
) -> None:
    terminal = reduce_fvg_termination(
        _fvg_lifecycle(),
        cause=cause,
        known_at=_at(minutes=5),
        cause_event_ids=(f"event-{cause.value}",),
        related_entity_id=related_entity_id,
    )
    assert terminal.availability is FVGAvailability.EXPIRED
    assert terminal.terminal_source_event_ids[:3] == (
        "event-fvg-1",
        "event-structure-generation-1",
        "event-structural-range-1",
    )

    with pytest.raises(ValueError, match="exact"):
        reduce_fvg_termination(
            _fvg_lifecycle(),
            cause=cause,
            known_at=_at(minutes=5),
            cause_event_ids=(f"event-{cause.value}",),
            related_entity_id="sibling-entity",
        )


def test_fvg_creation_context_cannot_be_bound_after_age_or_rewritten() -> None:
    unbound = replace(
        _fvg_lifecycle(),
        parent_structure_generation_id=None,
        structural_range_id=None,
        context_source_event_ids=(),
    )
    bound = bind_fvg_structural_context(
        unbound,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        source_event_ids=(
            "event-structure-generation-1",
            "event-structural-range-1",
        ),
    )
    assert bound.parent_structure_generation_id == "structure-generation-1"
    assert bind_fvg_structural_context(
        bound,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        source_event_ids=bound.context_source_event_ids,
    ) is bound

    with pytest.raises(ValueError, match="immutable"):
        bind_fvg_structural_context(
            bound,
            parent_structure_generation_id="structure-generation-2",
            structural_range_id="structural-range-2",
            source_event_ids=("event-structure-generation-2",),
        )
    with pytest.raises(ValueError, match="creation time"):
        bind_fvg_structural_context(
            replace(
                unbound,
                age_bars=1,
                age_seconds=300,
                last_updated_at=_at(minutes=5),
            ),
            parent_structure_generation_id="structure-generation-1",
            structural_range_id=None,
            source_event_ids=("event-structure-generation-1",),
        )


def test_fvg_has_no_ttl_and_terminal_generation_is_immutable() -> None:
    with pytest.raises(ValueError):
        FVGTerminationCause("age_ttl")

    expired = reduce_fvg_termination(
        _fvg_lifecycle(),
        cause=FVGTerminationCause.SEMANTIC_RESET,
        known_at=_at(minutes=5),
        cause_event_ids=("event-semantic-reset",),
    )
    with pytest.raises(ValueError, match="immutable"):
        reduce_fvg_termination(
            expired,
            cause=FVGTerminationCause.DATA_GAP,
            known_at=_at(minutes=10),
            cause_event_ids=("event-data-gap",),
        )


def test_outcome_scans_strictly_after_start_and_freezes_full_spec() -> None:
    spec = _outcome_spec()
    outcome = StructuralOutcomeEngine.evaluate(
        spec,
        (
            _outcome_bar(
                0,
                bar_event_id="source-bar",
                high=103.0,
                low=97.0,
            ),
            _outcome_bar(5, high=102.0, low=99.0, close=101.0),
            _outcome_bar(10, high=101.0, low=97.75, close=99.0),
            _outcome_bar(15, high=100.5, low=99.5, close=100.0),
        ),
    )

    assert outcome.terminal is OutcomeTerminal.TARGET_FIRST
    assert outcome.target_hit_at == _at(minutes=5)
    assert outcome.invalidation_hit_at == _at(minutes=10)
    assert outcome.source_bar_event_ids == (
        "outcome-bar-5",
        "outcome-bar-10",
        "outcome-bar-15",
    )
    assert outcome.mfe_atr == 1.0
    assert outcome.mae_atr == 1.125
    assert outcome.atr_at_start == spec.atr_at_start
    assert outcome.observation_window_end_exclusive == (
        spec.observation_window_end_exclusive
    )
    assert (outcome.symbol, outcome.instrument_id) == (
        spec.symbol,
        spec.instrument_id,
    )


def test_outcome_same_bar_is_factually_ambiguous_and_projection_is_separate() -> None:
    outcome = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=1),
        (_outcome_bar(5, high=102.0, low=98.0),),
    )
    factual_id = outcome.outcome_id

    assert outcome.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    projection = project_conservative_execution(outcome)
    assert projection.factual_outcome_id == factual_id
    assert projection.factual_terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    assert projection.execution_terminal is OutcomeTerminal.INVALIDATION_FIRST
    assert projection.conservative_assumption_applied
    assert outcome.terminal is OutcomeTerminal.AMBIGUOUS_SAME_BAR
    assert outcome.outcome_id == factual_id


def test_outcome_contract_change_and_incomplete_paths_are_censored() -> None:
    contract_censored = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=2),
        (_outcome_bar(5, instrument_id=999),),
    )
    assert contract_censored.terminal is OutcomeTerminal.CENSORED
    assert contract_censored.observed_bars == 0
    assert contract_censored.path_censor_reason == "contract_change"
    assert contract_censored.resolved_at == _at(minutes=5)

    incomplete = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=3),
        (_outcome_bar(5),),
    )
    assert incomplete.terminal is OutcomeTerminal.CENSORED
    assert incomplete.observed_bars == 1
    assert incomplete.path_censor_reason == "incomplete_completed_bar_census"
    assert incomplete.mfe_atr is None
    assert incomplete.mae_atr is None


def test_outcome_window_boundary_is_explicit_censoring() -> None:
    outcome = StructuralOutcomeEngine.evaluate(
        _outcome_spec(
            horizon_bars=3,
            horizon_seconds=1_800,
            window_end_minutes=10,
        ),
        (_outcome_bar(5), _outcome_bar(10)),
    )
    assert outcome.terminal is OutcomeTerminal.CENSORED
    assert outcome.path_censor_reason == "observation_window_end"
    assert outcome.resolved_at == _at(minutes=10)
    assert outcome.source_bar_event_ids == ("outcome-bar-5",)


def test_outcome_seconds_horizon_and_native_bar_clocks_are_frozen() -> None:
    elapsed = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=4, horizon_seconds=600),
        (_outcome_bar(5), _outcome_bar(10)),
    )
    assert elapsed.terminal is OutcomeTerminal.CENSORED
    assert elapsed.path_censor_reason == "horizon_seconds_elapsed"
    assert elapsed.resolved_at == _at(minutes=10)

    with pytest.raises(ValueError, match="duplicated or out of order"):
        StructuralOutcomeEngine.evaluate(
            _outcome_spec(horizon_bars=2),
            (
                _outcome_bar(5, bar_event_id="bar-a"),
                _outcome_bar(5, bar_event_id="bar-b"),
            ),
        )


def test_outcome_censors_at_first_missing_native_completed_bar() -> None:
    outcome = StructuralOutcomeEngine.evaluate(
        _outcome_spec(horizon_bars=2, horizon_seconds=3_600),
        (
            _outcome_bar(5),
            _outcome_bar(50, high=110.0, low=90.0),
        ),
    )

    assert outcome.terminal is OutcomeTerminal.CENSORED
    assert outcome.observed_bars == 1
    assert outcome.source_bar_event_ids == ("outcome-bar-5",)
    assert outcome.path_censor_reason == "missing_native_completed_bar"
    assert outcome.resolved_at == _at(minutes=10)
    assert outcome.target_hit_at is None
    assert outcome.invalidation_hit_at is None


def test_outcome_is_deterministic_and_invariant_to_bars_after_horizon() -> None:
    spec = _outcome_spec(horizon_bars=2)
    prefix = (_outcome_bar(5), _outcome_bar(10))
    first = StructuralOutcomeEngine.evaluate(spec, prefix)
    repeated = StructuralOutcomeEngine.evaluate(spec, prefix)
    with_future = StructuralOutcomeEngine.evaluate(
        spec,
        (
            *prefix,
            _outcome_bar(15, high=110.0, low=90.0),
        ),
    )

    assert first.terminal is OutcomeTerminal.CENSORED
    assert first.full_horizon_observed
    assert first.path_censor_reason is None
    assert first.outcome_id == repeated.outcome_id == with_future.outcome_id


def test_outcome_rejects_off_grid_input_and_wrong_timeframe() -> None:
    spec = _outcome_spec(horizon_bars=1)
    with pytest.raises(ValueError, match="off-grid"):
        StructuralOutcomeEngine.evaluate(
            spec,
            (_outcome_bar(5, high=101.1),),
        )
    with pytest.raises(ValueError, match="timeframe differs"):
        StructuralOutcomeEngine.evaluate(
            spec,
            (_outcome_bar(5, timeframe=Timeframe.M1),),
        )


def test_group3_exactly_binds_base_core_and_qualified_ob() -> None:
    tracker, legacy, _, _, _, _, update = _form_order_block()
    bindings = _pending_foundation_bindings(tracker.group3)

    finalized = tracker.group3.finalize_foundation(update, **bindings)

    assert len(finalized.base_origin_cores) == 1
    assert len(finalized.qualified_order_blocks) == 1
    core = finalized.base_origin_cores[0]
    qualified = finalized.qualified_order_blocks[0]
    assert qualified.base_origin_core_id == core.core_id
    assert qualified.source_displacement_id == (
        legacy.source_displacement_id
    )
    assert qualified.source_displacement_event_id == (
        core.source_displacement_event_id
    )
    assert qualified.compatible_structure_event_id == (
        bindings["structure_event_ids_by_entity"][legacy.source_bos_id]
    )
    assert core.source_displacement_event_id in (
        bindings["displacement_event_ids_by_identity"].values()
    )
    assert set(core.anchor_bar_event_ids).issubset(
        bindings["bar_event_ids_by_candle_id"].values()
    )
    provisional = {
        seed.compatible_structure_entity_id
        for completed in tracker.group3._pending_foundation_completed
        for seed in completed.new_qualified_order_blocks
    }
    assert qualified.compatible_structure_event_id not in provisional


def test_group3_foundation_finalization_is_failure_atomic() -> None:
    tracker, _, _, _, _, _, update = _form_order_block()
    bindings = _pending_foundation_bindings(tracker.group3)
    bindings["structure_event_ids_by_entity"] = {}
    before = (
        dict(tracker.group3._base_origin_cores),
        dict(tracker.group3._qualified_order_blocks),
        dict(tracker.group3._fvg_structural_lifecycles),
        dict(tracker.group3._zone_reinteraction_trackers),
        tuple(tracker.group3._pending_foundation_completed),
        tracker.group3.snapshot(),
    )

    with pytest.raises(ValueError, match="not canonically bound"):
        tracker.group3.finalize_foundation(update, **bindings)

    after = (
        dict(tracker.group3._base_origin_cores),
        dict(tracker.group3._qualified_order_blocks),
        dict(tracker.group3._fvg_structural_lifecycles),
        dict(tracker.group3._zone_reinteraction_trackers),
        tuple(tracker.group3._pending_foundation_completed),
        tracker.group3.snapshot(),
    )
    assert after == before
    assert tracker.group3._failed is True


def test_group3_raw_only_bos_retains_legacy_ob_without_qob_companion() -> None:
    tracker, legacy, _, _, _, _, update = _form_order_block()
    restored = _pickle_round_trip(tracker.group3)
    bindings = _pending_foundation_bindings(tracker.group3)
    bindings["structure_event_ids_by_entity"] = {}
    raw_only = (_raw_only_disposition(legacy),)

    finalized = tracker.group3.finalize_foundation(
        update,
        raw_only_structure_dispositions=raw_only,
        **bindings,
    )
    replayed = restored.finalize_foundation(
        update,
        raw_only_structure_dispositions=raw_only,
        **bindings,
    )
    assert finalized == replayed
    assert legacy in finalized.order_blocks
    assert len(finalized.base_origin_cores) == 1
    assert finalized.qualified_order_blocks == ()
    assert finalized.first_retests == ()
    assert tracker.group3._qualified_order_blocks == {}
    assert tracker.group3._zone_reinteraction_trackers == {}


def test_group3_raw_only_bos_conflict_and_arbitrary_identity_fail_closed() -> None:
    tracker, legacy, _, _, _, _, update = _form_order_block()
    bindings = _pending_foundation_bindings(tracker.group3)

    with pytest.raises(ValueError, match="conflicts"):
        tracker.group3.finalize_foundation(
            update,
            raw_only_structure_dispositions=(
                _raw_only_disposition(legacy),
            ),
            **bindings,
        )

    tracker, _, _, _, _, _, update = _form_order_block()
    bindings = _pending_foundation_bindings(tracker.group3)
    bindings["structure_event_ids_by_entity"] = {}
    with pytest.raises(ValueError, match="no exact provisional QOB seed"):
        tracker.group3.finalize_foundation(
            update,
            raw_only_structure_dispositions=(
                _raw_only_disposition(legacy, bos_id="unrelated-bos"),
            ),
            **bindings,
        )


def test_group3_raw_only_bos_rejects_nonqualified_seed_disposition() -> None:
    tracker, legacy, _, _, _, _, update = _form_order_block()
    completed = tracker.group3._pending_foundation_completed[-1]
    seed = completed.new_qualified_order_blocks[0]
    tracker.group3._pending_foundation_completed[-1] = replace(
        completed,
        new_qualified_order_blocks=(
            replace(
                seed,
                compatible_structure_kind=(
                    CompatibleStructureKind.MSS_CORE_CONFIRMED
                ),
            ),
        ),
    )
    bindings = _pending_foundation_bindings(tracker.group3)
    bindings["structure_event_ids_by_entity"] = {}

    with pytest.raises(ValueError, match="does not match"):
        tracker.group3.finalize_foundation(
            update,
            raw_only_structure_dispositions=(
                _raw_only_disposition(legacy),
            ),
            **bindings,
        )


def _install_raw_only_qob_provenance(
    observer: CausalObserver,
    tracker,
    *,
    include_bos_displacement: bool = False,
    mutate_raw=None,
) -> tuple[str, MarketEvent]:
    observer._zone_tracker = tracker
    completed = next(
        completed
        for completed in reversed(tracker._pending_foundation_completed)
        if completed.new_qualified_order_blocks
    )
    seed = completed.new_qualified_order_blocks[0]
    state = seed.legacy_state
    bos_id = state.source_bos_id
    target_event_id = "canonical-target-swing"
    break_bar_event_id = "canonical-break-bar"
    structure_event_id = "canonical-structure-direction"
    displacement_event_id = "canonical-displacement"
    bos_state_event_id = "typed-confirmed-bos"
    protected_assignment_event_id = "canonical-protected-assignment"
    protected_swing_id = "protected-opposite-swing"
    bos_source_displacement_id = (
        state.source_displacement_id if include_bos_displacement else None
    )

    structure = MarketEvent(
        event_id=structure_event_id,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        observed_at=state.source_bos_pending_at,
        timeframe=state.timeframe,
        side=(
            "above" if state.direction is Direction.LONG else "below"
        ),
        price=state.invalidation_price,
        strength=state.strength,
        direction=state.direction,
        evidence={"structure_id": state.source_bos_structure_id},
        source_entity_ids=(
            state.source_bos_structure_id,
            "source-high",
            "source-low",
        ),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    displacement = MarketEvent(
        event_id=displacement_event_id,
        kind=EventKind.DISPLACEMENT_OBSERVED,
        observed_at=state.source_displacement_active_at,
        timeframe=state.timeframe,
        side=None,
        price=None,
        strength=state.strength,
        direction=state.direction,
        event_time=state.source_displacement_started_at,
        evidence={
            "transition_id": state.source_active_transition_id,
            "displacement_id": state.source_displacement_id,
            "lifecycle": "active",
        },
        source_entity_ids=(state.source_displacement_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    bos_state = MarketEvent(
        event_id=bos_state_event_id,
        kind=EventKind.STRUCTURE_BREAK,
        observed_at=state.source_bos_resolved_at,
        timeframe=state.timeframe,
        side=(
            "above" if state.direction is Direction.LONG else "below"
        ),
        price=state.invalidation_price,
        strength=state.strength,
        source_ids=(
            state.source_bos_target_swing_id,
            state.source_bos_structure_id,
            *(
                ()
                if bos_source_displacement_id is None
                else (bos_source_displacement_id,)
            ),
            state.source_bos_break_bar_id,
        ),
        entity_id=bos_id,
        lifecycle=BOSLifecycle.CONFIRMED.value,
        formed_at=state.source_bos_pending_at,
        confirmed_at=state.source_bos_resolved_at,
        direction=state.direction,
        evidence={
            "bos_id": bos_id,
            "scope": BOSScope.CONTINUATION.value,
            "source_structure_id": state.source_bos_structure_id,
            "break_bar_id": state.source_bos_break_bar_id,
            "source_displacement_id": bos_source_displacement_id,
        },
    )
    protected_assignment = MarketEvent(
        event_id=protected_assignment_event_id,
        kind=EventKind.PROTECTED_SWING_ASSIGNED,
        observed_at=state.source_bos_pending_at,
        timeframe=state.timeframe,
        side=(
            "below" if state.direction is Direction.LONG else "above"
        ),
        price=state.invalidation_price,
        strength=state.strength,
        direction=(
            Direction.SHORT
            if state.direction is Direction.LONG
            else Direction.LONG
        ),
        evidence={"protected_swing_id": protected_swing_id},
        source_entity_ids=(protected_swing_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    raw = MarketEvent(
        event_id="canonical-raw-break",
        kind=EventKind.RAW_BOUNDARY_BREAK,
        observed_at=state.source_bos_resolved_at,
        timeframe=state.timeframe,
        side=(
            "above" if state.direction is Direction.LONG else "below"
        ),
        price=state.invalidation_price,
        strength=state.strength,
        direction=state.direction,
        evidence={
            "bos_id": bos_id,
            "scope": BOSScope.CONTINUATION.value,
            "target_swing_id": state.source_bos_target_swing_id,
            "break_bar_id": state.source_bos_break_bar_id,
            "source_displacement_id": bos_source_displacement_id,
        },
        source_event_ids=(target_event_id, break_bar_event_id),
        source_data_ids=(state.source_bos_break_bar_id,),
        source_entity_ids=(
            bos_id,
            state.source_bos_target_swing_id,
            state.source_bos_structure_id,
            *(
                ()
                if bos_source_displacement_id is None
                else (bos_source_displacement_id,)
            ),
        ),
        context_event_ids=(
            bos_state_event_id,
            *(
                ()
                if bos_source_displacement_id is None
                else (displacement_event_id,)
            ),
        ),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    if mutate_raw is not None:
        raw = mutate_raw(raw, state)
    for event in (
        structure,
        displacement,
        bos_state,
        protected_assignment,
        raw,
    ):
        observer.memory.append(event)
    observer._confirmed_swing_event_ids[
        state.source_bos_target_swing_id
    ] = target_event_id
    observer._bar_event_ids_by_candle_id[
        state.source_bos_break_bar_id
    ] = break_bar_event_id
    observer._structure_direction_event_ids[
        state.source_bos_structure_id
    ] = structure_event_id
    observer._displacement_event_ids[
        state.source_active_transition_id
    ] = displacement_event_id
    observer._displacement_event_ids[state.source_displacement_id] = (
        displacement_event_id
    )
    observer._raw_break_event_ids[bos_id] = raw.event_id
    observer._protected_swing_event_ids[protected_swing_id] = (
        protected_assignment_event_id
    )
    observer._raw_only_structure_dispositions[bos_id] = (
        ZoneRawOnlyStructureDisposition(
            bos_id=bos_id,
            raw_break_event_id=raw.event_id,
            protected_assignment_event_id=(
                protected_assignment_event_id
            ),
            timeframe=state.timeframe,
            direction=state.direction,
            resolved_at=state.source_bos_resolved_at,
            source_structure_id=state.source_bos_structure_id,
            target_swing_id=state.source_bos_target_swing_id,
            break_bar_id=state.source_bos_break_bar_id,
            bos_source_displacement_id=bos_source_displacement_id,
        )
    )
    return bos_id, raw


def _tamper_raw_only_qob(
    raw: MarketEvent,
    state,
    tamper: str,
) -> MarketEvent:
    if tamper == "bos":
        return replace(raw, evidence={**raw.evidence, "bos_id": "other-bos"})
    if tamper == "scope":
        return replace(raw, evidence={**raw.evidence, "scope": "opposed"})
    if tamper == "timeframe":
        return replace(raw, timeframe=Timeframe.M15)
    if tamper == "direction":
        direction = (
            Direction.SHORT
            if state.direction is Direction.LONG
            else Direction.LONG
        )
        return replace(raw, direction=direction)
    if tamper == "clock":
        observed_at = raw.known_at + pd.Timedelta(minutes=5)
        return replace(raw, observed_at=observed_at, known_at=observed_at)
    if tamper == "structure":
        return replace(
            raw,
            source_entity_ids=(
                raw.source_entity_ids[0],
                raw.source_entity_ids[1],
                "other-structure",
                *raw.source_entity_ids[3:],
            ),
        )
    if tamper == "target":
        return replace(
            raw,
            evidence={**raw.evidence, "target_swing_id": "other-target"},
        )
    if tamper == "break-bar":
        return replace(raw, source_data_ids=("other-break-bar",))
    if tamper == "forged-displacement-context":
        return replace(
            raw,
            context_event_ids=(raw.context_event_ids[0], "canonical-displacement"),
        )
    raise AssertionError(f"unknown raw-only QOB tamper: {tamper}")


def _raw_only_qob_case(
    *,
    include_bos_displacement: bool = False,
    raw_tamper: str | None = None,
) -> tuple[CausalObserver, Any, Any, str, MarketEvent]:
    harness, _, _, _, _, _, _ = _form_order_block()
    observer = _production_observer()
    mutate_raw = (
        None
        if raw_tamper is None
        else lambda raw, state: _tamper_raw_only_qob(raw, state, raw_tamper)
    )
    bos_id, raw = _install_raw_only_qob_provenance(
        observer,
        harness.group3,
        include_bos_displacement=include_bos_displacement,
        mutate_raw=mutate_raw,
    )
    completed = next(
        completed
        for completed in reversed(harness.group3._pending_foundation_completed)
        if completed.new_qualified_order_blocks
    )
    state = completed.new_qualified_order_blocks[0].legacy_state
    return observer, harness.group3, state, bos_id, raw


def test_observer_proves_exact_raw_only_qob_disposition() -> None:
    observer, _, _, bos_id, _ = _raw_only_qob_case()

    dispositions = observer._group3_raw_only_structure_dispositions()
    assert tuple(item.bos_id for item in dispositions) == (bos_id,)
    assert dispositions[0].protected_assignment_event_id == (
        "canonical-protected-assignment"
    )

    observer._qualified_structure_event_ids[bos_id] = (
        "canonical-qualified-bos"
    )
    with pytest.raises(ValueError, match="conflicts"):
        observer._group3_raw_only_structure_dispositions()

    observer, _, _, bos_id, _ = _raw_only_qob_case(
        include_bos_displacement=True,
    )
    dispositions = observer._group3_raw_only_structure_dispositions()
    assert dispositions[0].bos_source_displacement_id is not None


@pytest.mark.parametrize(
    ("stage", "error_type", "error_text"),
    (
        ("success", None, None),
        ("tracker", ValueError, "tracker finalize failed"),
        ("projection", ValueError, "projection validation failed"),
        ("changed", RuntimeError, "changed during finalization"),
    ),
)
def test_observer_retires_only_matched_raw_only_receipt_after_full_finalize(
    monkeypatch,
    stage: str,
    error_type,
    error_text: str | None,
) -> None:
    tracker, _, _, _, _, _, update = _form_order_block()
    observer = _production_observer()
    bos_id, _ = _install_raw_only_qob_provenance(
        observer,
        tracker.group3,
    )
    receipt = observer._raw_only_structure_dispositions[bos_id]
    unrelated = replace(
        receipt,
        bos_id="unrelated-raw-only-bos",
        raw_break_event_id="unrelated-raw-only-break",
    )
    observer._raw_only_structure_dispositions[unrelated.bos_id] = unrelated
    finalized = object()

    def finalize(candidate, **bindings):
        assert candidate is update
        assert bindings["raw_only_structure_dispositions"] == (receipt,)
        if stage == "tracker":
            raise ValueError("tracker finalize failed")
        if stage == "changed":
            observer._raw_only_structure_dispositions[bos_id] = replace(
                receipt,
                protected_assignment_event_id="changed-assignment",
            )
        return finalized

    monkeypatch.setattr(tracker.group3, "finalize_foundation", finalize)

    def validate(candidate) -> None:
        assert candidate is finalized
        if stage == "projection":
            raise ValueError("projection validation failed")

    monkeypatch.setattr(
        observer,
        "_validate_group3_foundation_projection",
        validate,
    )
    before = dict(observer._raw_only_structure_dispositions)
    if error_type is None:
        assert observer._finalize_group3_foundation(
            update,
            _observer_update(_observer_m5(0).end),
        ) is finalized
        assert observer._raw_only_structure_dispositions == {
            unrelated.bos_id: unrelated,
        }
    else:
        with pytest.raises(error_type, match=error_text):
            observer._finalize_group3_foundation(
                update,
                _observer_update(_observer_m5(0).end),
            )
        if stage != "changed":
            assert observer._raw_only_structure_dispositions == before
        else:
            assert bos_id in observer._raw_only_structure_dispositions
            assert unrelated.bos_id in observer._raw_only_structure_dispositions


def test_observer_raw_only_qob_rejects_foreign_bos_displacement() -> None:
    observer, _, _, bos_id, _ = _raw_only_qob_case()
    observer._raw_only_structure_dispositions[bos_id] = replace(
        observer._raw_only_structure_dispositions[bos_id],
        bos_source_displacement_id="foreign-displacement",
    )

    with pytest.raises(ValueError, match="optional BOS displacement"):
        observer._group3_raw_only_structure_dispositions()


def test_observer_raw_only_qob_uses_capture_time_witness_and_no_relation_fact() -> None:
    observer, _, _, bos_id, _ = _raw_only_qob_case()
    observer._protected_swing_event_ids.clear()
    assert tuple(
        item.bos_id
        for item in observer._group3_raw_only_structure_dispositions()
    ) == (bos_id,)

    observer, _, _, bos_id, raw = _raw_only_qob_case()
    observer.memory.append(
        MarketEvent(
            event_id="unmapped-qualified-bos",
            kind=EventKind.QUALIFIED_BOS,
            observed_at=raw.known_at,
            timeframe=raw.timeframe,
            side=raw.side,
            price=raw.price,
            strength=raw.strength,
            direction=raw.direction,
            evidence={"bos_id": bos_id},
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
    )
    with pytest.raises(ValueError, match="audit fact"):
        observer._group3_raw_only_structure_dispositions()


@pytest.mark.parametrize("semantic_drift", (False, True))
def test_historical_protected_custody_uses_committed_digest_for_cold_retry(
    monkeypatch,
    semantic_drift: bool,
) -> None:
    observer, _, _, _, raw = _raw_only_qob_case()
    committed_events = tuple(observer.memory._audit_pending)
    committed_by_id = {
        event.event_id: event for event in committed_events
    }
    committed_digests = {
        event_id: observer.audit_store.recompute_event_digest(event)
        for event_id, event in committed_by_id.items()
    }
    monkeypatch.setattr(
        observer.audit_store,
        "events",
        lambda: committed_events,
    )
    monkeypatch.setattr(
        observer.audit_store,
        "get",
        committed_by_id.get,
    )
    monkeypatch.setattr(
        observer.audit_store,
        "event_digest",
        committed_digests.__getitem__,
    )
    assignment = committed_by_id["canonical-protected-assignment"]
    committed_raw = committed_by_id[raw.event_id]
    observer.memory = type(observer.memory)(
        1,
        audit_store=observer.audit_store,
    )
    retry = observer.memory.append(
        replace(
            assignment,
            sequence_no=0,
            price=(
                assignment.price + 0.25
                if semantic_drift
                else assignment.price
            ),
        ),
        sequence_floor=100,
    )
    assert retry.sequence_no != assignment.sequence_no
    before = (
        tuple(observer.memory._audit_pending),
        observer.memory.recent(),
        dict(observer.memory._entity_timelines),
        dict(observer.memory._sequence_counts),
    )

    if semantic_drift:
        with pytest.raises(
            ValueError,
            match="pending protected-swing history conflicts with audit",
        ):
            observer._historical_live_protected_assignment(
                assignment.timeframe,
                at_event=committed_raw,
            )
    else:
        historical = observer._historical_live_protected_assignment(
            assignment.timeframe,
            at_event=committed_raw,
        )
        assert historical is assignment
        assert historical.sequence_no == assignment.sequence_no
    assert (
        tuple(observer.memory._audit_pending),
        observer.memory.recent(),
        dict(observer.memory._entity_timelines),
        dict(observer.memory._sequence_counts),
    ) == before


def test_observer_raw_only_qob_capture_witness_survives_later_replacement() -> None:
    observer, _, _, bos_id, raw = _raw_only_qob_case()
    original = observer.memory.audit_event_including_pending(
        "canonical-protected-assignment"
    )
    assert original is not None
    later_clock = raw.known_at + pd.Timedelta(minutes=5)
    replacement = replace(
        original,
        event_id="later-protected-assignment",
        observed_at=later_clock,
        known_at=later_clock,
        event_time=later_clock,
        evidence={"protected_swing_id": "later-protected-swing"},
        source_entity_ids=("later-protected-swing",),
    )
    replacement = observer.memory.append(
        replacement,
        include_in_recent=False,
    )
    observer._protected_swing_event_ids.clear()
    observer._protected_swing_event_ids["later-protected-swing"] = (
        replacement.event_id
    )

    assert tuple(
        item.bos_id
        for item in observer._group3_raw_only_structure_dispositions()
    ) == (bos_id,)


def test_observer_raw_only_qob_rejects_witness_replaced_before_raw() -> None:
    observer, _, _, _, raw = _raw_only_qob_case()
    original = observer.memory.audit_event_including_pending(
        "canonical-protected-assignment"
    )
    assert original is not None
    replacement_clock = raw.known_at - pd.Timedelta(nanoseconds=1)
    replacement = replace(
        original,
        event_id="prior-protected-assignment",
        observed_at=replacement_clock,
        known_at=replacement_clock,
        event_time=replacement_clock,
        evidence={"protected_swing_id": "prior-protected-swing"},
        source_entity_ids=("prior-protected-swing",),
    )
    observer.memory.append(replacement, include_in_recent=False)

    with pytest.raises(ValueError, match="historical protected-assignment"):
        observer._group3_raw_only_structure_dispositions()


@pytest.mark.parametrize("terminal_after_raw", (False, True))
def test_observer_raw_only_qob_replays_protected_acceptance_interval(
    terminal_after_raw: bool,
) -> None:
    observer, _, _, bos_id, raw = _raw_only_qob_case()
    assignment = observer.memory.audit_event_including_pending(
        "canonical-protected-assignment"
    )
    assert assignment is not None
    terminal_clock = raw.known_at + pd.Timedelta(
        nanoseconds=1 if terminal_after_raw else -1
    )
    state_timeframe = raw.timeframe
    acceptance = MarketEvent(
        event_id=(
            "later-protected-acceptance"
            if terminal_after_raw
            else "prior-protected-acceptance"
        ),
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        observed_at=terminal_clock,
        timeframe=state_timeframe,
        side="below",
        price=assignment.price,
        strength=assignment.strength,
        direction=(
            Direction.SHORT
            if assignment.direction is Direction.LONG
            else Direction.LONG
        ),
        event_time=terminal_clock,
        evidence={
            "protected_swing_id": "protected-opposite-swing",
            "protected_swing_event_id": assignment.event_id,
            "source_timeframe": state_timeframe.value,
        },
        context_event_ids=(assignment.event_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    observer.memory.append(acceptance, include_in_recent=False)
    observer._protected_swing_event_ids.clear()

    if terminal_after_raw:
        assert tuple(
            item.bos_id
            for item in observer._group3_raw_only_structure_dispositions()
        ) == (bos_id,)
    else:
        with pytest.raises(
            ValueError,
            match="historical protected-assignment",
        ):
            observer._group3_raw_only_structure_dispositions()


def test_observer_raw_only_qob_uses_active_transition_not_entity_alias() -> None:
    observer, _, state, bos_id, _ = _raw_only_qob_case(
        include_bos_displacement=True,
    )
    active_id = observer._displacement_event_ids[
        state.source_active_transition_id
    ]
    active = observer.memory.audit_event_including_pending(active_id)
    assert active is not None
    terminal_clock = state.confirmed_at + pd.Timedelta(minutes=5)
    terminal = replace(
        active,
        event_id="later-terminal-displacement",
        observed_at=terminal_clock,
        known_at=terminal_clock,
        evidence={
            **active.evidence,
            "transition_id": "later-terminal-transition",
            "lifecycle": "exhausted",
        },
    )
    observer.memory.append(terminal, include_in_recent=False)
    observer._displacement_event_ids[state.source_displacement_id] = (
        terminal.event_id
    )

    assert tuple(
        item.bos_id
        for item in observer._group3_raw_only_structure_dispositions()
    ) == (bos_id,)


@pytest.mark.parametrize(
    "tamper",
    ("transition_id", "lifecycle", "active_clock"),
)
def test_observer_raw_only_qob_rejects_tampered_active_transition(
    tamper: str,
) -> None:
    observer, _, state, _, _ = _raw_only_qob_case()
    active_event_id = observer._displacement_event_ids[
        state.source_active_transition_id
    ]
    active = observer.memory.audit_event_including_pending(active_event_id)
    assert active is not None
    evidence = dict(active.evidence)
    observed_at = active.known_at
    if tamper == "transition_id":
        evidence["transition_id"] = "foreign-active-transition"
    elif tamper == "lifecycle":
        evidence["lifecycle"] = "exhausted"
    else:
        observed_at += pd.Timedelta(minutes=5)
    forged = replace(
        active,
        event_id=f"tampered-active-displacement:{tamper}",
        observed_at=observed_at,
        known_at=observed_at,
        evidence=evidence,
    )
    observer.memory.append(forged, include_in_recent=False)
    observer._displacement_event_ids[state.source_active_transition_id] = (
        forged.event_id
    )

    with pytest.raises(ValueError, match="BOS, structure, displacement"):
        observer._group3_raw_only_structure_dispositions()


@pytest.mark.parametrize(
    "tamper",
    (
        "bos",
        "scope",
        "timeframe",
        "direction",
        "clock",
        "structure",
        "target",
        "break-bar",
        "forged-displacement-context",
    ),
)
def test_observer_rejects_tampered_raw_only_qob_disposition(
    tamper: str,
) -> None:
    observer, _, _, _, _ = _raw_only_qob_case(raw_tamper=tamper)

    with pytest.raises(ValueError, match="raw-only QOB disposition"):
        observer._group3_raw_only_structure_dispositions()


def test_group3_fvg_first_retest_is_first_only_and_future_invariant() -> None:
    tracker, state, _, _, update = _form_fvg(Direction.LONG)
    finalized = tracker.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(tracker.group3),
    )
    assert finalized.first_retests == ()
    lifecycle = finalized.fvg_structural_lifecycles[0]
    assert lifecycle.fvg_id == state.fvg_id

    _, _, _, equality = tracker.send(
        (102.25, 103.0, state.upper_bound, 102.5)
    )
    equality = tracker.group3.finalize_foundation(
        equality,
        **_pending_foundation_bindings(tracker.group3),
    )
    assert len(equality.first_retest_transitions) == 1
    first = equality.first_retest_transitions[0]
    assert first.fill_fraction == 0.0
    assert first.entry_side is ZoneEntrySide.FROM_ABOVE
    aged_once = next(
        item
        for item in equality.fvg_structural_lifecycles
        if item.fvg_id == state.fvg_id
    )
    assert aged_once.age_bars == 1
    assert aged_once.age_seconds == 300

    _, _, _, future = tracker.send(
        (102.5, 110.0, 90.0, 102.5)
    )
    future = tracker.group3.finalize_foundation(
        future,
        **_pending_foundation_bindings(tracker.group3),
    )
    assert future.first_retest_transitions == ()
    assert future.first_retests == (first,)
    aged_twice = next(
        item
        for item in future.fvg_structural_lifecycles
        if item.fvg_id == state.fvg_id
    )
    assert aged_twice.age_bars == 2
    assert aged_twice.age_seconds == 600
    assert set(first.__dict__).isdisjoint(
        {
            "eventual_midpoint_touch",
            "eventual_full_fill",
            "eventual_invalidation",
            "eventual_continuation",
        }
    )


@pytest.mark.parametrize(
    ("cause", "related_entity_id"),
    (
        (
            FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
            "structure-generation-1",
        ),
        (
            FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
            "structural-range-1",
        ),
    ),
)
def test_group3_structural_fvg_expiry_is_reachable_from_bound_context(
    cause: FVGTerminationCause,
    related_entity_id: str,
) -> None:
    tracker, state, _, _, update = _form_fvg(Direction.LONG)
    tracker.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(tracker.group3),
    )
    bound = tracker.group3.bind_fvg_foundation_context(
        fvg_id=state.fvg_id,
        parent_structure_generation_id="structure-generation-1",
        structural_range_id="structural-range-1",
        source_event_ids=(
            "event-structure-generation-1",
            "event-structural-range-1",
        ),
    )
    assert bound.fvg_structural_lifecycles[0].structural_range_id == (
        "structural-range-1"
    )

    expired = tracker.group3.expire_fvg_foundation_context(
        cause=cause,
        related_entity_id=related_entity_id,
        known_at=state.confirmed_at + pd.Timedelta(5, unit="min"),
        cause_event_id=f"event-{cause.value}",
    )
    assert len(expired.fvg_structural_transitions) == 1
    terminal = expired.fvg_structural_transitions[0]
    assert terminal.availability is FVGAvailability.EXPIRED
    assert terminal.terminal_event_id == f"event-{cause.value}"


@pytest.mark.parametrize("cause", tuple(FVGTerminationCause))
def test_group3_fvg_terminal_without_retest_removes_only_its_tracker(
    cause: FVGTerminationCause,
) -> None:
    harness, state, _, _, update = _form_fvg(Direction.LONG)
    harness.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(harness.group3),
    )
    related_entity_id = None
    if cause in {
        FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
        FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
    }:
        harness.group3.bind_fvg_foundation_context(
            fvg_id=state.fvg_id,
            parent_structure_generation_id="structure-generation-1",
            structural_range_id="structural-range-1",
            source_event_ids=(
                "event-structure-generation-1",
                "event-structural-range-1",
            ),
        )
        related_entity_id = (
            "structure-generation-1"
            if cause is FVGTerminationCause.PARENT_STRUCTURE_TERMINATED
            else "structural-range-1"
        )
    fvg_key = harness.group3._zone_tracker_key(
        ZoneObjectKind.FVG,
        state.fvg_id,
    )
    original_tracker = harness.group3._zone_reinteraction_trackers[fvg_key]
    sibling_id = "sibling-fvg"
    sibling_key = harness.group3._zone_tracker_key(
        ZoneObjectKind.FVG,
        sibling_id,
    )
    qob_key = harness.group3._zone_tracker_key(
        ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
        "sibling-qob",
    )
    harness.group3._fvg_structural_lifecycles[sibling_id] = replace(
        harness.group3._fvg_structural_lifecycles[state.fvg_id],
        fvg_id=sibling_id,
        source_creation_event_id="event-sibling-fvg",
    )
    harness.group3._zone_reinteraction_trackers[sibling_key] = (
        ZoneFirstReinteractionTracker(
            replace(
                original_tracker.spec,
                object_id=sibling_id,
                creation_event_id="event-sibling-fvg",
            )
        )
    )
    harness.group3._zone_reinteraction_trackers[qob_key] = (
        ZoneFirstReinteractionTracker(
            replace(
                original_tracker.spec,
                object_kind=ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
                object_id="sibling-qob",
                creation_event_id="event-sibling-qob",
            )
        )
    )
    restored = _pickle_round_trip(harness.group3)
    kwargs = {
        "entity_id": state.fvg_id,
        "cause": cause,
        "known_at": state.confirmed_at + pd.Timedelta(minutes=5),
        "cause_event_id": f"event-{cause.value}",
        "terminal_event_id": f"event-{cause.value}",
        "related_entity_id": related_entity_id,
    }

    terminal = harness.group3._terminate_fvg_foundation(**kwargs)
    restored_terminal = restored._terminate_fvg_foundation(**kwargs)

    assert terminal == restored_terminal
    assert terminal is not None
    assert fvg_key not in harness.group3._zone_reinteraction_trackers
    assert sibling_key in harness.group3._zone_reinteraction_trackers
    assert qob_key in harness.group3._zone_reinteraction_trackers
    assert restored.current_update() == harness.group3.current_update()


@pytest.mark.parametrize(
    ("reason", "drops_unresolved"),
    (
        ("data_gap_reset", True),
        ("synthetic_interruption", True),
        ("registered_session_reset", False),
    ),
)
def test_group3_boundary_cleans_only_censored_unresolved_qob_tracker(
    reason: str,
    drops_unresolved: bool,
) -> None:
    harness, legacy, _, _, _, _, update = _form_order_block()
    finalized = harness.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(harness.group3),
    )
    key = harness.group3._zone_tracker_key(
        ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
        legacy.order_block_id,
    )
    assert key in harness.group3._zone_reinteraction_trackers
    restored = _pickle_round_trip(harness.group3)
    boundary_event_id = f"canonical-boundary:{reason}"

    boundary = harness.group3.on_boundary(
        reason,
        legacy.confirmed_at + pd.Timedelta(minutes=5),
        foundation_boundary_event_id=boundary_event_id,
    )
    restored_boundary = restored.on_boundary(
        reason,
        legacy.confirmed_at + pd.Timedelta(minutes=5),
        foundation_boundary_event_id=boundary_event_id,
    )

    assert restored_boundary == boundary
    assert finalized.qualified_order_blocks[0] in (
        boundary.qualified_order_blocks
    )
    assert (
        key not in harness.group3._zone_reinteraction_trackers
    ) is drops_unresolved
    assert restored.current_update() == harness.group3.current_update()


def test_group3_qob_terminal_bar_keeps_retest_but_drops_unresolved_path() -> None:
    harness, legacy, _, _, _, _, update = _form_order_block()
    finalized = harness.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(harness.group3),
    )
    key = harness.group3._zone_tracker_key(
        ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
        legacy.order_block_id,
    )
    retained_qob = finalized.qualified_order_blocks[0]
    failure_values = (
        legacy.lower_bound - 0.25,
        legacy.lower_bound - 0.25,
        legacy.lower_bound - 1.0,
        legacy.lower_bound - 0.5,
    )
    _, _, _, failed = harness.send(failure_values)
    failed = harness.group3.finalize_foundation(
        failed,
        **_pending_foundation_bindings(harness.group3),
    )
    assert key not in harness.group3._zone_reinteraction_trackers
    assert failed.qualified_order_blocks == (retained_qob,)

    replayed, replayed_legacy, _, _, _, _, replayed_update = (
        _form_order_block()
    )
    replayed.group3.finalize_foundation(
        replayed_update,
        **_pending_foundation_bindings(replayed.group3),
    )
    no_touch = (102.75, 103.5, 102.25, 103.5)
    _, _, _, no_touch_update = replayed.send(no_touch)
    replayed.group3.finalize_foundation(
        no_touch_update,
        **_pending_foundation_bindings(replayed.group3),
    )
    touch = (103.5, 103.75, 101.75, 103.0)
    _, _, _, touched = replayed.send(touch)
    touched = replayed.group3.finalize_foundation(
        touched,
        **_pending_foundation_bindings(replayed.group3),
    )
    replayed_key = replayed.group3._zone_tracker_key(
        ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
        replayed_legacy.order_block_id,
    )
    assert touched.first_retest_transitions
    historical = replayed.group3._zone_reinteraction_trackers[
        replayed_key
    ].first_retest
    assert historical is not None

    boundary = replayed.group3.on_boundary(
        "synthetic_interruption",
        historical.known_at + pd.Timedelta(minutes=5),
        foundation_boundary_event_id="canonical-boundary:after-qob-retest",
    )
    assert replayed_key in replayed.group3._zone_reinteraction_trackers
    assert tuple(
        first_retest
        for first_retest in boundary.first_retests
        if first_retest.object_kind is ZoneObjectKind.QUALIFIED_ORDER_BLOCK
    ) == (historical,)


def test_group3_fvg_gap_open_inside_is_geometric_first_retest() -> None:
    tracker, state, _, _, update = _form_fvg(Direction.LONG)
    tracker.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(tracker.group3),
    )

    _, _, _, reentry = tracker.send(
        (101.0, 101.25, 100.75, 101.0)
    )
    reentry = tracker.group3.finalize_foundation(
        reentry,
        **_pending_foundation_bindings(tracker.group3),
    )
    event = reentry.first_retest_transitions[0]
    assert event.entry_side is ZoneEntrySide.GAP_OPENED_INSIDE
    assert event.fill_fraction == pytest.approx(0.75)
    assert event.object_id == state.fvg_id


@pytest.mark.parametrize(
    ("reason", "availability", "cause"),
    (
        (
            "data_gap_reset",
            FVGAvailability.CENSORED,
            FVGTerminationCause.DATA_GAP,
        ),
        (
            "contract_change_reset",
            FVGAvailability.EXPIRED,
            FVGTerminationCause.CONTRACT_ROLLOVER,
        ),
        (
            "semantic_reset",
            FVGAvailability.EXPIRED,
            FVGTerminationCause.SEMANTIC_RESET,
        ),
    ),
)
def test_group3_fvg_reset_classification_has_no_ttl(
    reason: str,
    availability: FVGAvailability,
    cause: FVGTerminationCause,
) -> None:
    tracker, state, _, _, update = _form_fvg(Direction.LONG)
    tracker.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(tracker.group3),
    )
    reset_event_id = f"canonical-reset:{reason}"

    boundary = tracker.group3.on_boundary(
        reason,
        state.confirmed_at + pd.Timedelta(minutes=5),
        foundation_boundary_event_id=reset_event_id,
    )

    terminal = boundary.fvg_structural_transitions[0]
    assert terminal.availability is availability
    assert terminal.terminal_reason is cause
    assert terminal.terminal_event_id == reset_event_id
    assert terminal.terminal_source_event_ids[-1] == reset_event_id


def _finalize_fvg_damage(harness, state):
    damage_values = (102.25, 102.5, 99.5, state.lower_bound - 0.25)
    _, _, _, damaged = harness.send(damage_values)
    bindings = _pending_foundation_bindings(harness.group3)
    return (
        harness.group3.finalize_foundation(damaged, **bindings),
        bindings,
    )


def test_group3_price_damage_is_invalidated_with_exact_terminal_event() -> None:
    tracker, state, _, _, update = _form_fvg(Direction.LONG)
    tracker.group3.finalize_foundation(
        update,
        **_pending_foundation_bindings(tracker.group3),
    )
    restored = _pickle_round_trip(tracker)
    damaged, bindings = _finalize_fvg_damage(tracker, state)
    restored_damaged, _ = _finalize_fvg_damage(restored, state)
    assert restored_damaged == damaged

    terminal = damaged.fvg_structural_transitions[0]
    assert terminal.availability is FVGAvailability.INVALIDATED
    assert terminal.terminal_reason is (
        FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE
    )
    assert terminal.terminal_event_id == (
        bindings["fvg_terminal_event_ids_by_entity"][state.fvg_id]
    )
    assert terminal.terminal_event_id in terminal.terminal_source_event_ids
    assert any(
        event_id in bindings["bar_event_ids_by_candle_id"].values()
        for event_id in terminal.terminal_source_event_ids
    )
    tracker_key = tracker.group3._zone_tracker_key(
        ZoneObjectKind.FVG,
        state.fvg_id,
    )
    retained_tracker = tracker.group3._zone_reinteraction_trackers[
        tracker_key
    ]
    assert retained_tracker.first_retest is not None
    assert retained_tracker.first_retest.known_at == terminal.last_updated_at

    _, _, _, future = tracker.send(
        (102.5, 110.0, 90.0, 102.5)
    )
    future = tracker.group3.finalize_foundation(
        future,
        **_pending_foundation_bindings(tracker.group3),
    )
    assert future.first_retest_transitions == ()
    assert future.first_retests == (retained_tracker.first_retest,)

    replayed, replayed_state, _, _, replayed_update = _form_fvg(
        Direction.LONG
    )
    replayed.group3.finalize_foundation(
        replayed_update,
        **_pending_foundation_bindings(replayed.group3),
    )
    replayed_damage, _ = _finalize_fvg_damage(replayed, replayed_state)
    assert replayed_damage == damaged


def test_observer_publishes_base_core_at_start_and_keeps_history() -> None:
    observer = _production_observer()
    for index in range(64):
        values = (
            (101.0, 102.0, 99.0, 100.75)
            if index == 63
            else (100.0, 101.0, 100.0, 100.0)
        )
        candle = _observer_m5(index, values)
        observer.observe(
            _observer_update(candle.end, m5=(candle,))
        )
    seed = _observer_m5(64, (100.75, 102.0, 100.75, 102.0))
    observation = observer.observe(
        _observer_update(seed.end, m5=(seed,))
    )
    projection = observer.last_zone_foundation_projection
    assert projection is not None
    assert observation.displacement is not None
    assert observation.displacement.lifecycle == "started"
    assert len(projection.base_origin_cores) == 1
    assert projection.qualified_order_blocks == ()
    core = projection.base_origin_cores[0]
    displacement = observer.memory.audit_event_including_pending(
        core.source_displacement_event_id
    )
    assert displacement is not None
    assert displacement.kind is EventKind.DISPLACEMENT_OBSERVED
    assert displacement.evidence["displacement_id"] == (
        core.source_displacement_id
    )
    for candle_id, event_id in zip(
        core.anchor_candle_ids,
        core.anchor_bar_event_ids,
    ):
        event = observer.memory.audit_event_including_pending(event_id)
        assert event is not None
        assert event.kind is EventKind.BAR_COMPLETED
        assert event.evidence["detector_candle_id"] == candle_id

    for index in (65, 66):
        pause = _observer_m5(
            index,
            (102.0, 102.25, 101.75, 102.0),
        )
        observation = observer.observe(
            _observer_update(pause.end, m5=(pause,))
        )
    assert observation.displacement is not None
    assert observation.displacement.lifecycle == "idle"
    retained = observer.last_zone_foundation_projection
    assert retained is not None
    assert retained.base_origin_cores == (core,)
    assert retained.qualified_order_blocks == ()


def test_observer_fvg_and_first_retest_sources_are_canonical_events() -> None:
    observer = _production_observer()
    index = 0
    for _ in range(15):
        candle = _observer_m5(
            index,
            (100.0, 101.0, 100.0, 100.0),
        )
        observer.observe(
            _observer_update(candle.end, m5=(candle,))
        )
        index += 1
    for values in (
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 100.0, 101.5),
        (101.5, 102.25, 101.5, 102.25),
    ):
        candle = _observer_m5(index, values)
        observer.observe(
            _observer_update(candle.end, m5=(candle,))
        )
        index += 1
    projection = observer.last_zone_foundation_projection
    assert projection is not None
    lifecycle = projection.fvg_structural_lifecycles[0]
    created = observer.memory.audit_event_including_pending(
        lifecycle.source_creation_event_id
    )
    assert created is not None
    assert created.kind is EventKind.FVG_CREATED
    assert created.evidence["fvg_id"] == lifecycle.fvg_id

    equality = _observer_m5(
        index,
        (102.25, 103.0, 101.5, 102.5),
    )
    observer.observe(
        _observer_update(equality.end, m5=(equality,))
    )
    projection = observer.last_zone_foundation_projection
    assert projection is not None
    assert len(projection.first_retest_transitions) == 1
    retest = projection.first_retest_transitions[0]
    assert retest.fill_fraction == 0.0
    expected_kinds = (
        EventKind.FVG_CREATED,
        EventKind.DISPLACEMENT_OBSERVED,
        EventKind.BAR_COMPLETED,
    )
    for event_id, expected_kind in zip(
        retest.source_event_ids,
        expected_kinds,
    ):
        event = observer.memory.audit_event_including_pending(event_id)
        assert event is not None
        assert event.kind is expected_kind

    future = _observer_m5(
        index + 1,
        (102.5, 103.0, 102.0, 102.75),
    )
    observer.observe(_observer_update(future.end, m5=(future,)))
    future_projection = observer.last_zone_foundation_projection
    assert future_projection is not None
    assert future_projection.first_retest_transitions == ()
    assert future_projection.first_retests == (retest,)


def test_observer_deferred_synthetic_boundary_is_censored_exactly() -> None:
    observer = _production_observer()
    index = 0
    last_real = None
    for _ in range(15):
        candle = _observer_m5(
            index,
            (100.0, 101.0, 100.0, 100.0),
        )
        observer.observe(
            _observer_update(candle.end, m5=(candle,))
        )
        last_real = candle
        index += 1
    for values in (
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 100.0, 101.5),
        (101.5, 102.25, 101.5, 102.25),
    ):
        candle = _observer_m5(index, values)
        observer.observe(
            _observer_update(candle.end, m5=(candle,))
        )
        last_real = candle
        index += 1
    assert last_real is not None
    synthetic_m5 = replace(
        _observer_m5(
            index,
            (102.25, 102.5, 102.0, 102.25),
        ),
        real_minutes=4,
        synthetic_minutes=1,
    )
    for minute_offset in range(4, 0, -1):
        update = _observer_update(
            synthetic_m5.end
            - pd.Timedelta(minutes=minute_offset)
        )
        observer.observe(
            replace(
                update,
                histories={
                    **update.histories,
                    Timeframe.M5: (last_real,),
                },
            )
        )
    update = _observer_update(
        synthetic_m5.end,
        m5=(synthetic_m5,),
    )
    synthetic_m1 = replace(
        update.completed_1m,
        open=102.25,
        high=102.25,
        low=102.25,
        close=102.25,
        volume=0.0,
        real_minutes=0,
        synthetic_minutes=1,
    )
    observation = observer.observe(
        replace(
            update,
            completed_1m=synthetic_m1,
            newly_completed={
                **update.newly_completed,
                Timeframe.M1: (synthetic_m1,),
            },
            histories={
                **update.histories,
                Timeframe.M1: (synthetic_m1,),
            },
        )
    )

    projection = observer.last_zone_foundation_projection
    assert projection is not None
    assert projection.boundary_reason == "synthetic_interruption"
    terminal = projection.fvg_structural_transitions[0]
    assert terminal.availability is FVGAvailability.CENSORED
    assert terminal.terminal_reason is FVGTerminationCause.DATA_GAP
    boundary = observer.memory.audit_event_including_pending(
        terminal.terminal_event_id
    )
    assert boundary is not None
    assert boundary.kind is EventKind.DISPLACEMENT_OBSERVED
    assert boundary.evidence["lifecycle"] == "censored"
    assert boundary.evidence["terminal_reason"] == (
        "synthetic_interruption"
    )
    assert observation.displacement is not None
    assert observation.displacement.lifecycle == "idle"
