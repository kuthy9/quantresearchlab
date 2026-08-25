from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_semantic_signal_research as signal_research_runner
from smc_trader.model import (
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    Timeframe,
    to_primitive,
)
from smc_trader.semantics import SemanticRegistry
from smc_trader.signal_research import (
    REQUIRED_IDENTITY_BINDINGS,
    REQUIRED_RUNTIME_CODE_BINDINGS,
    ResearchContractError,
    canonical_result_identity,
    completed_bar_distance,
    direct_lineage_tokens,
    find_prior_source_link,
    load_frozen_research_contract,
    resolve_lineage_tokens,
)
from scripts.run_semantic_signal_research import (
    ATOMIC_KINDS,
    DIAGNOSTIC_DATA_GAP_POLICY,
    FVG_LIFECYCLE_KINDS,
    PARENT_RELATION_PRIORITY,
    PRIMARY_PATH_SCAN_CONTRACT,
    RANGE_INVALIDATION_POOLING,
    RANGE_INVALIDATION_VARIANT_CONTRACT,
    REGISTERED_ATOMIC_POPULATION,
    RESEARCH_SELECTION,
    REQUIRED_RESEARCH_LIMITATIONS,
    SECONDARY_OUTCOME_DEFINITIONS,
    SIGNAL_DIRECTION_ASSIGNMENT,
    SNAPSHOT_AUTHORITY,
    STATE_PROJECTION_PERSISTENCE,
    _build_eye,
    _artifact_status,
    _assert_full_input_census,
    _enrich_next_structural_context,
    _event_record,
    ROOT,
    _load_contract_and_registry,
    _matched_controls,
    _outcome,
    _parent_bucket,
    _passes_research_selection,
    _quartile,
    _quartile_cutoffs,
    _range_invalidation_variant,
    _source_linked_stage,
    _summary,
    _validate_directional_entry_rows,
    _validate_registry_atomic_population,
    _validated_link_contracts,
    _validated_research_design,
    _write_jsonl,
)


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-02 09:30", tz="America/New_York") + pd.Timedelta(
        int(minutes), unit="min"
    )


def _fixture_rows(
    rows: list[dict[str, object]],
    *,
    signal: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Bind legacy unit rows to explicit test-only BAR identities."""

    signal = {} if signal is None else signal
    bound: list[dict[str, object]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        row.setdefault("symbol", signal.get("symbol", "TEST"))
        row.setdefault("instrument_id", signal.get("instrument_id", 0))
        row.setdefault("open", row["close"])
        row.setdefault("timeframe", Timeframe.M1.value)
        row.setdefault("tick_size", 0.01)
        if "bar_event_id" not in row:
            payload = json.dumps(
                to_primitive({"fixture_index": index, "row": row}),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            row["bar_event_id"] = (
                "research-row:" + hashlib.sha256(payload).hexdigest()
            )
        bound.append(row)
    return bound


def _fixture_outcome(
    signal: dict[str, object],
    rows: list[dict[str, object]],
    row_index: dict[pd.Timestamp, int],
    **parameters: object,
) -> dict[str, object] | None:
    return _outcome(
        signal,
        _fixture_rows(rows, signal=signal),
        row_index,
        _allow_test_compatibility_rows=True,
        **parameters,
    )


def _fixture_summary(
    signals: list[dict[str, object]],
    rows: list[dict[str, object]],
    row_index: dict[pd.Timestamp, int],
    *,
    outcome_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    signal = signals[0] if signals else None
    return _summary(
        signals,
        _fixture_rows(rows, signal=signal),
        row_index,
        outcome_parameters=outcome_parameters,
        _allow_test_compatibility_rows=True,
    )


def test_research_eye_does_not_persist_redundant_state_projections() -> None:
    _, observer = _build_eye()

    assert observer.config.persist_state_projections is False
    assert STATE_PROJECTION_PERSISTENCE == (
        "disabled_in_diagnostic_redundant_non_authoritative_transport"
    )


def test_structural_outcome_starts_after_the_semantic_known_at_bar() -> None:
    rows = [
        {
            "asof": _clock(0),
            "close": 100.0,
            "high": 102.0,
            "low": 98.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "close": 100.5,
            "high": 101.25,
            "low": 99.75,
            "atr": 1.0,
        },
    ]
    signal = {"known_at": _clock(0), "direction": "long"}

    outcome = _fixture_outcome(signal, rows, {_clock(0): 0, _clock(1): 1})

    assert outcome is not None
    assert outcome["success"] is True
    assert outcome["time_to_target_completed_bars"] == 1


def test_same_later_bar_target_and_invalidation_is_ambiguous() -> None:
    rows = [
        {
            "asof": _clock(0),
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "close": 100.0,
            "high": 101.25,
            "low": 98.75,
            "atr": 1.0,
        },
    ]
    signal = {"known_at": _clock(0), "direction": "long"}

    outcome = _fixture_outcome(signal, rows, {_clock(0): 0, _clock(1): 1})

    assert outcome is not None
    assert outcome["ambiguous"] is True
    assert outcome["resolved"] is False
    assert outcome["success"] is None


def test_path_metrics_scan_full_horizon_after_primary_first_hit() -> None:
    rows = [
        {
            "asof": _clock(0),
            "close": 100.0,
            "high": 100.2,
            "low": 99.8,
            "atr": 1.0,
            "instrument_id": 1,
        },
        {
            "asof": _clock(1),
            "close": 101.0,
            "high": 101.25,
            "low": 100.5,
            "atr": 1.0,
            "instrument_id": 1,
        },
        {
            "asof": _clock(2),
            "close": 98.5,
            "high": 101.5,
            "low": 98.0,
            "atr": 1.0,
            "instrument_id": 1,
        },
    ]
    signal = {
        "known_at": _clock(0),
        "direction": "long",
        "instrument_id": 1,
    }

    outcome = _fixture_outcome(
        signal,
        rows,
        {row["asof"]: index for index, row in enumerate(rows)},
        horizon=2,
    )

    assert outcome is not None
    assert outcome["success"] is True
    assert outcome["time_to_target_completed_bars"] == 1
    assert outcome["time_to_invalidation_completed_bars"] == 2
    assert outcome["mfe_atr"] == pytest.approx(1.5)
    assert outcome["mae_atr"] == pytest.approx(2.0)
    assert outcome["continuation_distance_atr"] == pytest.approx(1.0)
    assert outcome["retracement_depth_atr"] == pytest.approx(2.5)
    assert outcome["range_extension_atr"] == pytest.approx(1.3)
    assert outcome["observed_completed_bars"] == 2


def test_zone_retest_and_fvg_midpoint_use_later_completed_bars() -> None:
    rows = [
        {
            "asof": _clock(0),
            "close": 102.0,
            "high": 102.5,
            "low": 101.5,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "close": 102.0,
            "high": 102.3,
            "low": 101.2,
            "atr": 1.0,
        },
        {
            "asof": _clock(2),
            "close": 100.5,
            "high": 101.0,
            "low": 99.8,
            "atr": 1.0,
        },
    ]
    signal = {
        "known_at": _clock(0),
        "direction": "long",
        "kind": EventKind.FVG_CREATED.value,
        "zone": (99.0, 101.0),
    }

    outcome = _fixture_outcome(
        signal,
        rows,
        {row["asof"]: index for index, row in enumerate(rows)},
        horizon=2,
    )

    assert outcome is not None
    assert outcome["time_to_first_retest_completed_bars"] == 2
    assert outcome["time_to_fvg_midpoint_touch_completed_bars"] == 2


def test_structural_outcome_censors_at_contract_change() -> None:
    rows = [
        {
            "asof": _clock(0),
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
            "instrument_id": 1,
        },
        {
            "asof": _clock(1),
            "close": 101.0,
            "high": 102.0,
            "low": 100.5,
            "atr": 1.0,
            "instrument_id": 2,
        },
    ]
    signal = {
        "known_at": _clock(0),
        "direction": "long",
        "instrument_id": 1,
    }

    outcome = _fixture_outcome(signal, rows, {_clock(0): 0, _clock(1): 1})

    assert outcome is not None
    assert outcome["resolved"] is False
    assert outcome["time_to_target_completed_bars"] is None
    assert outcome["observed_completed_bars"] == 0
    assert outcome["censored_by_contract_change"] is True
    assert outcome["full_horizon_observed"] is False
    assert outcome["mfe_atr"] is None


def test_window_end_censor_excludes_full_horizon_path_metrics() -> None:
    rows = [
        {
            "asof": _clock(0),
            "symbol": "NQ",
            "instrument_id": 1,
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "symbol": "NQ",
            "instrument_id": 1,
            "close": 101.0,
            "high": 101.25,
            "low": 100.5,
            "atr": 1.0,
        },
    ]
    signal = {
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 1,
        "direction": "long",
    }

    outcome = _fixture_outcome(
        signal,
        rows,
        {row["asof"]: index for index, row in enumerate(rows)},
        horizon=2,
    )

    assert outcome is not None
    assert outcome["success"] is True
    assert outcome["censored_by_window_end"] is True
    assert outcome["full_horizon_observed"] is False
    assert outcome["mfe_atr"] is None
    assert outcome["continuation_distance_atr"] is None

    summary = _fixture_summary(
        [signal],
        rows,
        {row["asof"]: index for index, row in enumerate(rows)},
        outcome_parameters={"horizon": 2},
    )
    assert summary["resolved_n"] == 1
    assert summary["window_end_censored_n"] == 1
    assert summary["full_horizon_outcome_n"] == 0
    assert summary["mean_mfe_atr"] is None


def test_outcome_censors_on_symbol_change_even_when_instrument_id_repeats() -> None:
    rows = [
        {
            "asof": _clock(0),
            "symbol": "NQ",
            "instrument_id": 1,
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "symbol": "ES",
            "instrument_id": 1,
            "close": 101.0,
            "high": 102.0,
            "low": 100.5,
            "atr": 1.0,
        },
    ]
    signal = {
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 1,
        "direction": "long",
    }

    outcome = _fixture_outcome(
        signal,
        rows,
        {_clock(0): 0, _clock(1): 1},
        horizon=1,
    )

    assert outcome is not None
    assert outcome["censored_by_contract_change"] is True
    assert outcome["observed_completed_bars"] == 0


def test_outcome_rejects_signal_entry_row_identity_mismatch() -> None:
    rows = [
        {
            "asof": _clock(0),
            "symbol": "ES",
            "instrument_id": 1,
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "symbol": "NQ",
            "instrument_id": 1,
            "close": 101.0,
            "high": 102.0,
            "low": 100.5,
            "atr": 1.0,
        },
    ]
    signal = {
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 1,
        "direction": "long",
    }

    with pytest.raises(ResearchContractError, match="exact known_at entry row"):
        _fixture_outcome(
            signal,
            rows,
            {_clock(0): 0, _clock(1): 1},
            horizon=1,
        )


def test_formal_outcome_requires_normalized_bar_ids() -> None:
    signal = {
        "event_id": "semantic:signal:1",
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 1,
        "direction": "long",
    }
    rows = _fixture_rows(
        [
            {
                "asof": _clock(0),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "atr": 1.0,
            },
            {
                "asof": _clock(1),
                "open": 100.0,
                "high": 101.0,
                "low": 100.0,
                "close": 101.0,
                "atr": 1.0,
            },
        ],
        signal=signal,
    )

    with pytest.raises(
        ResearchContractError,
        match="normalized BAR event ID",
    ):
        _outcome(
            signal,
            rows,
            {row["asof"]: index for index, row in enumerate(rows)},
            horizon=1,
        )


def test_unified_outcome_spec_and_result_ids_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = {
        "event_id": "semantic:signal:deterministic",
        "known_at": _clock(0),
        "symbol": "NQ",
        "instrument_id": 1,
        "direction": "long",
    }
    rows = [
        {
            "asof": _clock(0),
            "bar_event_id": "normalized-bar:m1:0",
            "timeframe": Timeframe.M1.value,
            "tick_size": 0.25,
            "symbol": "NQ",
            "instrument_id": 1,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "bar_event_id": "normalized-bar:m1:1",
            "timeframe": Timeframe.M1.value,
            "tick_size": 0.25,
            "symbol": "NQ",
            "instrument_id": 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.75,
            "close": 100.75,
            "atr": 1.0,
        },
    ]
    row_index = {row["asof"]: index for index, row in enumerate(rows)}
    captured_ids: list[tuple[str, str]] = []
    original_evaluate = signal_research_runner.StructuralOutcomeEngine.evaluate

    def capture_ids(spec: object, bars: object) -> object:
        outcome = original_evaluate(spec, bars)  # type: ignore[arg-type]
        captured_ids.append((outcome.spec_id, outcome.outcome_id))
        return outcome

    monkeypatch.setattr(
        signal_research_runner.StructuralOutcomeEngine,
        "evaluate",
        staticmethod(capture_ids),
    )

    first = _outcome(signal, rows, row_index, horizon=1)
    second = _outcome(signal, rows, row_index, horizon=1)

    assert first == second
    assert len(captured_ids) == 2
    assert captured_ids[0] == captured_ids[1]


def test_unified_outcome_preserves_frozen_signal_research_schema() -> None:
    rows = [
        {
            "asof": _clock(0),
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
        },
        {
            "asof": _clock(1),
            "close": 101.0,
            "high": 101.25,
            "low": 99.75,
            "atr": 1.0,
        },
    ]
    outcome = _fixture_outcome(
        {"known_at": _clock(0), "direction": "long"},
        rows,
        {_clock(0): 0, _clock(1): 1},
        horizon=1,
    )

    assert outcome is not None
    assert set(outcome) == {
        "resolved",
        "ambiguous",
        "success",
        "mfe_atr",
        "mae_atr",
        "mfe_over_mae",
        "continuation_distance_atr",
        "retracement_depth_atr",
        "range_extension_atr",
        "time_to_target_completed_bars",
        "time_to_invalidation_completed_bars",
        "time_to_first_retest_completed_bars",
        "time_to_fvg_midpoint_touch_completed_bars",
        "observed_completed_bars",
        "censored_by_contract_change",
        "censored_by_window_end",
        "full_horizon_observed",
        "next_structural_event_id",
        "next_structural_event_kind",
        "next_structural_event_direction",
        "next_structural_event_time",
        "next_structural_event_known_at",
        "next_structural_direction_match",
        "next_qualified_bos_direction",
        "next_qualified_bos_direction_match",
    }
    assert "first_retest_event" not in outcome


def test_quartile_cutoffs_are_frozen_once_and_handle_missing_values() -> None:
    cutoffs = _quartile_cutoffs((1.0, 2.0, 3.0, 4.0))

    assert cutoffs == (2.0, 3.0, 4.0)
    assert _quartile(cutoffs, 1.5) == 0
    assert _quartile(cutoffs, 2.5) == 1
    assert _quartile(cutoffs, float("inf")) == -1
    assert _quartile_cutoffs((float("inf"),)) is None


def test_parent_invalidation_bucket_precedes_directionless_transition() -> None:
    relation = SimpleNamespace(
        parent_tf=Timeframe.H1,
        child_tf=Timeframe.M5,
        parent_direction=None,
        parent_state_invalidated=True,
        parent_protected_swing_intact=False,
        role=SimpleNamespace(value="parent_transition"),
    )
    snapshot = SimpleNamespace(relations={"1H__5m": relation})

    assert (
        _parent_bucket(snapshot, Timeframe.M5, Direction.LONG)
        == "after_parent_invalidation"
    )


def test_parent_bucket_uses_event_direction_not_relation_role() -> None:
    relation = SimpleNamespace(
        parent_tf=Timeframe.H1,
        child_tf=Timeframe.M5,
        parent_direction=Direction.LONG,
        parent_state_invalidated=False,
        parent_protected_swing_intact=True,
        role=SimpleNamespace(value="parent_retracement"),
    )
    snapshot = SimpleNamespace(relations={"1H__5m": relation})

    assert (
        _parent_bucket(snapshot, Timeframe.M5, Direction.LONG)
        == "aligned_with_parent"
    )
    assert (
        _parent_bucket(snapshot, Timeframe.M5, Direction.SHORT)
        == "against_parent_but_parent_intact"
    )


def test_parent_bucket_uses_frozen_parent_priority() -> None:
    h1 = SimpleNamespace(
        parent_tf=Timeframe.H1,
        child_tf=Timeframe.M5,
        parent_direction=Direction.LONG,
        parent_state_invalidated=False,
        parent_protected_swing_intact=True,
    )
    m15 = SimpleNamespace(
        parent_tf=Timeframe.M15,
        child_tf=Timeframe.M5,
        parent_direction=Direction.SHORT,
        parent_state_invalidated=False,
        parent_protected_swing_intact=True,
    )
    snapshot = SimpleNamespace(relations={"15m__5m": m15, "1h__5m": h1})

    assert PARENT_RELATION_PRIORITY["5m"] == ["1h", "15m"]
    assert (
        _parent_bucket(snapshot, Timeframe.M5, Direction.LONG)
        == "aligned_with_parent"
    )


def test_completed_bar_distance_does_not_use_wall_clock_minutes() -> None:
    earlier = _clock(0)
    later = _clock(90)
    completed_index = {earlier: 10, later: 11}

    assert completed_bar_distance(earlier, later, completed_index) == 1
    assert completed_bar_distance(earlier, _clock(1), completed_index) is None


def test_source_link_rejects_arbitrary_window_cooccurrence() -> None:
    earlier = _clock(0)
    later = _clock(90)
    completed_index = {earlier: 0, later: 1}
    linked = {
        "event_id": "sweep:linked",
        "kind": "sweep_confirmed",
        "known_at": earlier,
        "direction": "long",
        "timeframe": "1m",
        "lineage_tokens": ("event:sweep:linked", "entity:L1"),
    }
    unlinked = {
        **linked,
        "event_id": "sweep:unlinked",
        "lineage_tokens": ("event:sweep:unlinked", "entity:L1"),
    }
    current = {
        "event_id": "displacement:D1",
        "kind": "displacement_observed",
        "known_at": later,
        "direction": "long",
        "timeframe": "5m",
        "lineage_tokens": ("event:sweep:linked", "entity:L1"),
    }

    link = find_prior_source_link(
        [linked, unlinked],
        current,
        completed_index=completed_index,
        kinds=frozenset({"sweep_confirmed"}),
        maximum_completed_bars=1,
        timeframe="1m",
    )

    assert link is not None
    assert link.prior["event_id"] == "sweep:linked"
    assert link.completed_bars == 1
    assert link.shared_tokens == ("event:sweep:linked",)

    entity_only = {**current, "lineage_tokens": ("entity:L1",)}
    assert (
        find_prior_source_link(
            [linked, unlinked],
            entity_only,
            completed_index=completed_index,
            kinds=frozenset({"sweep_confirmed"}),
            maximum_completed_bars=1,
            timeframe="1m",
        )
        is None
    )


def test_lineage_resolution_follows_immutable_source_ids() -> None:
    values = {
        "touch:1": SimpleNamespace(
            event_id="touch:1",
            origin=EventOrigin.SEMANTIC_ATOMIC,
            evidence={"level_id": "must-not-be-inferred"},
            source_ids=("opaque-legacy-id",),
            source_event_ids=(),
            source_data_ids=("bar:1",),
            source_entity_ids=("level:L1",),
            context_event_ids=(),
        ),
        "projection:1": SimpleNamespace(
            event_id="projection:1",
            origin=EventOrigin.STATE_PROJECTION,
            source_ids=("opaque-projection-id",),
            source_event_ids=("unresolved-projection-parent",),
            source_data_ids=(),
            source_entity_ids=(),
            context_event_ids=(),
        ),
        "sweep:1": SimpleNamespace(
            event_id="sweep:1",
            origin=EventOrigin.SEMANTIC_ATOMIC,
            evidence={},
            source_ids=("must-not-be-used",),
            source_event_ids=("touch:1",),
            source_data_ids=("bar:2",),
            source_entity_ids=("level:L1",),
            context_event_ids=("projection:1",),
        ),
    }

    tokens = resolve_lineage_tokens("sweep:1", values.get)

    assert "event:touch:1" in tokens
    assert "event:projection:1" in tokens
    assert "data:bar:1" in tokens
    assert "data:bar:2" in tokens
    assert "entity:level:L1" in tokens
    assert not any(token.startswith("source:") for token in tokens)
    assert "event:opaque-legacy-id" not in tokens
    assert "event:unresolved-projection-parent" not in tokens
    assert "level:must-not-be-inferred" not in tokens


@pytest.mark.parametrize("field", ("source_event_ids", "context_event_ids"))
def test_unresolved_canonical_lineage_parent_fails_closed(field: str) -> None:
    root = SimpleNamespace(
        event_id="atomic:root",
        origin=EventOrigin.SEMANTIC_ATOMIC,
        source_event_ids=(),
        source_data_ids=(),
        source_entity_ids=(),
        context_event_ids=(),
    )
    setattr(root, field, ("missing-parent",))

    with pytest.raises(ResearchContractError, match="parent is unresolved"):
        resolve_lineage_tokens("atomic:root", {"atomic:root": root}.get)


def test_research_lineage_rejects_non_atomic_root() -> None:
    projection = SimpleNamespace(
        event_id="projection:root",
        origin=EventOrigin.STATE_PROJECTION,
        source_event_ids=(),
        source_data_ids=(),
        source_entity_ids=(),
        context_event_ids=(),
    )

    with pytest.raises(ResearchContractError, match="not semantic_atomic"):
        resolve_lineage_tokens(
            "projection:root",
            {"projection:root": projection}.get,
        )


def test_raw_data_and_entity_ids_never_become_event_lineage() -> None:
    prior = SimpleNamespace(
        event_id="prior",
        origin=EventOrigin.LEGACY_TRANSPORT,
        source_ids=("raw-id",),
        source_event_ids=("raw-id",),
        source_data_ids=("data-id",),
        source_entity_ids=("entity-id",),
        context_event_ids=(),
    )

    tokens = direct_lineage_tokens(prior)

    assert "event:prior" in tokens
    assert "data:data-id" in tokens
    assert "entity:entity-id" in tokens
    assert "event:raw-id" not in tokens
    assert not any(token.startswith("source:") for token in tokens)


def test_research_event_record_requires_and_persists_explicit_origin() -> None:
    snapshot = SimpleNamespace(
        asof=_clock(1),
        relations={},
        session=SimpleNamespace(phase="new_york_am"),
        symbol="NQ",
        instrument_id=123,
    )
    canonical = MarketEvent(
        event_id="fvg:partial",
        kind=EventKind.FVG_PARTIALLY_FILLED,
        observed_at=_clock(1),
        timeframe=Timeframe.M5,
        side="below",
        price=100.0,
        strength=0.5,
        direction=Direction.LONG,
        source_event_ids=("fvg:created",),
        source_data_ids=("bar:1",),
        source_entity_ids=("fvg:1",),
        context_event_ids=("fvg-state:1",),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )

    record = _event_record(canonical, snapshot)

    assert record["origin"] == "semantic_atomic"
    assert record["source_event_ids"] == ("fvg:created",)
    assert record["source_data_ids"] == ("bar:1",)
    assert record["source_entity_ids"] == ("fvg:1",)
    assert record["context_event_ids"] == ("fvg-state:1",)

    legacy = SimpleNamespace(**{**canonical.__dict__, "origin": EventOrigin.LEGACY_TRANSPORT})
    with pytest.raises(ResearchContractError, match="not semantic_atomic"):
        _event_record(legacy, snapshot)

    late_snapshot = SimpleNamespace(**{**snapshot.__dict__, "asof": _clock(2)})
    with pytest.raises(ResearchContractError, match="exact known_at"):
        _event_record(canonical, late_snapshot)


@pytest.mark.parametrize(
    ("variant", "source_kinds"),
    (
        (
            "forming_close_before_activation",
            (EventKind.DEALING_RANGE_CREATED, EventKind.BAR_COMPLETED),
        ),
        (
            "forming_source_invalidated",
            (EventKind.DEALING_RANGE_CREATED, EventKind.BAR_COMPLETED),
        ),
        (
            "forming_maturity_deadline_elapsed",
            (EventKind.DEALING_RANGE_CREATED, EventKind.BAR_COMPLETED),
        ),
        (
            "active_acceptance",
            (
                EventKind.DEALING_RANGE_CREATED,
                EventKind.DEALING_RANGE_ACTIVATED,
                EventKind.BAR_COMPLETED,
                EventKind.ACCEPTANCE_CONFIRMED,
            ),
        ),
    ),
)
def test_range_invalidation_record_freezes_variant_and_exact_ancestry(
    variant: str,
    source_kinds: tuple[EventKind, ...],
) -> None:
    definition = RANGE_INVALIDATION_VARIANT_CONTRACT[variant]
    source_ids = tuple(f"source:{index}" for index in range(len(source_kinds)))
    event = MarketEvent(
        event_id=f"range:{variant}",
        kind=EventKind.DEALING_RANGE_INVALIDATED,
        observed_at=_clock(1),
        timeframe=Timeframe.H1,
        side=None,
        price=100.0,
        strength=0.5,
        direction=Direction.LONG,
        evidence={"transition_reason": definition["transition_reason"]},
        source_event_ids=source_ids,
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    snapshot = SimpleNamespace(
        asof=_clock(1),
        relations={},
        session=SimpleNamespace(phase="new_york_am"),
        symbol="NQ",
        instrument_id=123,
    )
    source_event_kinds = {
        event_id: kind.value
        for event_id, kind in zip(source_ids, source_kinds, strict=True)
    }

    record = _event_record(
        event,
        snapshot,
        source_event_kinds=source_event_kinds,
    )

    assert record["event_variant"] == variant
    assert _range_invalidation_variant(event, source_event_kinds) == variant

    wrong = {
        **source_event_kinds,
        source_ids[0]: EventKind.FVG_CREATED.value,
    }
    with pytest.raises(ResearchContractError, match="ancestry disagrees"):
        _event_record(event, snapshot, source_event_kinds=wrong)


def test_fvg_atomic_population_uses_explicit_lifecycle_events() -> None:
    fvg_kinds = {
        kind for kind in ATOMIC_KINDS if kind.value.startswith("fvg_")
    }

    assert EventKind.FVG_TOUCHED not in ATOMIC_KINDS
    assert FVG_LIFECYCLE_KINDS == {kind.value for kind in fvg_kinds}
    assert fvg_kinds == {
        EventKind.FVG_CREATED,
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
    }


def test_displacement_research_selection_uses_the_frozen_active_predicate() -> None:
    active = SimpleNamespace(
        kind=EventKind.DISPLACEMENT_OBSERVED,
        evidence={"lifecycle": "active"},
    )
    started = SimpleNamespace(
        kind=EventKind.DISPLACEMENT_OBSERVED,
        evidence={"lifecycle": "started"},
    )
    swing = SimpleNamespace(kind=EventKind.SWING_CONFIRMED, evidence={})

    assert _passes_research_selection(active) is True
    assert _passes_research_selection(started) is False
    assert _passes_research_selection(swing) is True


def test_atomic_population_is_the_emitted_phase2_phase3_semantic_surface() -> None:
    assert ATOMIC_KINDS == {
        EventKind.SWING_CONFIRMED,
        EventKind.STRUCTURAL_LEG_CREATED,
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.LEVEL_TOUCHED,
        EventKind.LEVEL_PENETRATED,
        EventKind.SWEEP_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
        EventKind.DISPLACEMENT_OBSERVED,
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.QUALIFIED_BOS,
        EventKind.PROTECTED_SWING_ASSIGNED,
        EventKind.MSS_CORE_CONFIRMED,
        EventKind.FVG_CREATED,
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
        EventKind.DEALING_RANGE_CREATED,
        EventKind.DEALING_RANGE_ACTIVATED,
        EventKind.DEALING_RANGE_INVALIDATED,
        EventKind.DEALING_RANGE_REPLACED,
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
    }
    assert not {
        EventKind.BAR_COMPLETED,
        EventKind.MARKET_EPOCH_RESET,
        EventKind.TIMEFRAME_STATE_CHANGED,
        EventKind.RELATION_STATE_CHANGED,
        EventKind.SESSION_STATE_CHANGED,
        EventKind.FVG_TOUCHED,
        EventKind.FVG_EXPIRED,
        EventKind.DEALING_RANGE_EXTENDED,
        EventKind.DELIVERY_PHASE_CHANGED,
        EventKind.ORIGIN_ZONE_TOUCHED,
    } & ATOMIC_KINDS

    registry = SemanticRegistry.from_file()
    assert registry.canonical_emitted_event_kinds == ATOMIC_KINDS
    assert REGISTERED_ATOMIC_POPULATION == {
        kind.value for kind in ATOMIC_KINDS
    }
    assert "displacement_observed:active" not in REGISTERED_ATOMIC_POPULATION


def test_runtime_atomic_population_is_checked_against_registry_statuses() -> None:
    registry = SemanticRegistry.from_file()

    _validate_registry_atomic_population(registry)

    stale = SimpleNamespace(
        canonical_emitted_event_kinds=(
            registry.canonical_emitted_event_kinds
            | {EventKind.FVG_EXPIRED}
        )
    )
    with pytest.raises(ResearchContractError, match="canonical_emitted union"):
        _validate_registry_atomic_population(stale)


def test_runner_loads_historical_registry_at_manifest_semantic_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "historical_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "semantic_version": "smc_semantics_v1.1",
                "semantic_registry": "semantics/registry.yaml",
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    class RegistryReadObserved(RuntimeError):
        pass

    def observe_registry_read(
        source: str | Path,
        *,
        required_version: str,
    ) -> None:
        observed["source"] = Path(source)
        observed["required_version"] = required_version
        raise RegistryReadObserved

    monkeypatch.setattr(
        SemanticRegistry,
        "from_file",
        staticmethod(observe_registry_read),
    )

    with pytest.raises(RegistryReadObserved):
        _load_contract_and_registry(manifest)

    assert observed == {
        "source": ROOT / "semantics/registry.yaml",
        "required_version": "smc_semantics_v1.1",
    }


def test_runner_requires_manifest_semantic_version_before_registry_read(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "missing_semantic_version.json"
    manifest.write_text(
        json.dumps({"semantic_registry": "semantics/registry.yaml"}),
        encoding="utf-8",
    )

    with pytest.raises(ResearchContractError, match="manifest semantic_version"):
        _load_contract_and_registry(manifest)


def test_source_linked_stage_persists_auditable_chain_edge() -> None:
    previous = {
        "event_id": "touch:1",
        "kind": "level_touched",
        "known_at": _clock(0),
        "direction": "short",
        "timeframe": "1m",
        "lineage_tokens": ("event:touch:1",),
    }
    current = {
        "event_id": "sweep:1",
        "kind": "sweep_confirmed",
        "known_at": _clock(8),
        "direction": "short",
        "timeframe": "1m",
        "lineage_tokens": ("event:touch:1",),
    }

    admitted, chains, ledger = _source_linked_stage(
        [previous],
        [current],
        completed_index={_clock(0): 0, _clock(8): 1},
        maximum_completed_bars=1,
        previous_kind="level_touched",
        previous_timeframe="1m",
        current_kind="sweep_confirmed",
        current_timeframe="1m",
        stage="E2_sweep",
        prior_chains={"touch:1": ("touch:1",)},
    )

    assert admitted == [current]
    assert chains == {"sweep:1": ("touch:1", "sweep:1")}
    assert ledger[0]["completed_bars"] == 1
    assert ledger[0]["shared_lineage_tokens"] == ("event:touch:1",)


def _matching_row(
    minute: int,
    *,
    atomic_event_count: int,
) -> dict[str, object]:
    return {
        "asof": _clock(minute),
        "symbol": "NQ",
        "instrument_id": 123,
        "session_phase": "new_york_am",
        "m1_direction": "long",
        "atr": 1.0,
        "nearest_distance_atr": 0.5,
        "relative_volume": 1.0,
        "atomic_event_count": atomic_event_count,
    }


def test_matched_control_and_treatment_use_the_same_paired_cohort() -> None:
    touches = [
        {
            "event_id": "touch:1",
            "known_at": _clock(0),
            "symbol": "NQ",
            "instrument_id": 123,
            "direction": "long",
            "timeframe": "1m",
        },
        {
            "event_id": "touch:2",
            "known_at": _clock(2),
            "symbol": "NQ",
            "instrument_id": 123,
            "direction": "long",
            "timeframe": "1m",
        },
    ]
    rows = [
        _matching_row(0, atomic_event_count=1),
        _matching_row(1, atomic_event_count=0),
        _matching_row(2, atomic_event_count=1),
    ]

    controls, matched_treatments, pairs = _matched_controls(touches, rows)

    assert len(controls) == len(matched_treatments) == len(pairs) == 1
    assert pairs[0]["treatment_event_id"] == matched_treatments[0]["event_id"]
    assert pairs[0]["control_event_id"] == controls[0]["event_id"]
    assert pairs[0]["stratum"]["instrument_id"] == 123


def test_summary_includes_time_to_invalidation() -> None:
    rows = [
        {"asof": _clock(0), "close": 100.0, "high": 100.0, "low": 100.0, "atr": 1.0},
        {"asof": _clock(1), "close": 100.0, "high": 100.5, "low": 99.5, "atr": 1.0},
        {"asof": _clock(2), "close": 99.0, "high": 100.4, "low": 98.75, "atr": 1.0},
    ]

    value = _fixture_summary(
        [{"known_at": _clock(0), "direction": "long"}],
        rows,
        {row["asof"]: index for index, row in enumerate(rows)},
        outcome_parameters={"horizon": 2},
    )

    assert value["median_time_to_invalidation_completed_bars"] == 2
    assert value["signal_half_life_completed_bars"] == 2
    assert value["mean_continuation_distance_atr"] == 0.0
    assert value["mean_retracement_depth_atr"] == pytest.approx(1.0)


def test_next_structural_context_persists_identity_and_next_bos_direction() -> None:
    signal = {
        "event_id": "signal:1",
        "kind": "sweep_confirmed",
        "known_at": _clock(0),
        "event_time": _clock(0),
        "symbol": "NQ",
        "instrument_id": 123,
        "direction": "long",
    }
    same_clock_bos = {
        "event_id": "bos:same-clock",
        "kind": EventKind.QUALIFIED_BOS.value,
        "known_at": _clock(0),
        "event_time": _clock(0),
        "symbol": "NQ",
        "instrument_id": 123,
        "direction": "short",
    }
    next_raw = {
        "event_id": "raw:next",
        "kind": EventKind.RAW_BOUNDARY_BREAK.value,
        "known_at": _clock(1),
        "event_time": _clock(1),
        "symbol": "NQ",
        "instrument_id": 123,
        "direction": "short",
    }
    next_bos = {
        "event_id": "bos:next",
        "kind": EventKind.QUALIFIED_BOS.value,
        "known_at": _clock(2),
        "event_time": _clock(1),
        "symbol": "NQ",
        "instrument_id": 123,
        "direction": "long",
    }

    _enrich_next_structural_context(
        [signal],
        [same_clock_bos, next_bos, next_raw],
    )

    assert signal["next_structural_event_id"] == "raw:next"
    assert signal["next_structural_event_kind"] == "raw_boundary_break"
    assert signal["next_structural_event_direction"] == "short"
    assert signal["next_structural_event_time"] == _clock(1)
    assert signal["next_structural_event_known_at"] == _clock(1)
    assert signal["next_structural_direction_match"] is False
    assert signal["next_qualified_bos_direction"] == "long"
    assert signal["next_qualified_bos_direction_match"] is True

    rows = [
        {
            "asof": _clock(0),
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "atr": 1.0,
            "symbol": "NQ",
            "instrument_id": 123,
        },
        {
            "asof": _clock(1),
            "close": 101.0,
            "high": 101.2,
            "low": 100.0,
            "atr": 1.0,
            "symbol": "NQ",
            "instrument_id": 123,
        },
    ]
    outcome = _fixture_outcome(
        signal,
        rows,
        {row["asof"]: index for index, row in enumerate(rows)},
    )
    assert outcome is not None
    assert outcome["next_structural_event_id"] == "raw:next"
    assert outcome["next_structural_event_kind"] == "raw_boundary_break"
    assert outcome["next_structural_event_direction"] == "short"
    assert outcome["next_qualified_bos_direction"] == "long"


def test_result_identity_excludes_elapsed_seconds_recursively() -> None:
    left = {
        "coverage": {"signals": 3},
        "run_metadata": {"elapsed_seconds": 1.0},
        "result_identity": "old",
    }
    right = {
        "coverage": {"signals": 3},
        "run_metadata": {"elapsed_seconds": 999.0},
        "result_identity": "different",
    }

    assert canonical_result_identity(left) == canonical_result_identity(right)


def test_jsonl_ledger_is_deterministic_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "event_study.jsonl"
    records = [
        {"event_id": "event:2", "known_at": _clock(2)},
        {"event_id": "event:1", "known_at": _clock(1)},
    ]

    _write_jsonl(path, records)
    first = path.read_bytes()
    _write_jsonl(path, records)

    assert path.read_bytes() == first
    assert [json.loads(line)["event_id"] for line in first.splitlines()] == [
        "event:2",
        "event:1",
    ]


def test_legacy_and_unfrozen_template_manifests_fail_closed() -> None:
    registry = SemanticRegistry.from_file(
        ROOT / "semantics/registry.yaml",
        required_version="smc_semantics_v1.1",
    )

    for manifest in (
        ROOT / "experiments/manifests/smc_semantic_v1_2024_01_signal_diagnostic.yaml",
        ROOT / "experiments/manifests/semantic_event_study_v2_template.yaml",
    ):
        with pytest.raises(ResearchContractError):
            load_frozen_research_contract(
                manifest,
                root=ROOT,
                actual_semantic_registry_identity=registry.identity,
            )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_template_preserves_historical_numbered_runtime_paths() -> None:
    expected_runtime = {
        "runtime_package_init": "smc_trader/__init__.py",
        "runtime_artifact_stream": "smc_trader/artifact_stream.py",
        "runtime_causal": "smc_trader/causal.py",
        "runtime_io": "smc_trader/io.py",
        "runtime_market_clock": "smc_trader/market_clock.py",
        "runtime_model": "smc_trader/model.py",
        "runtime_observation": "smc_trader/observation.py",
        "runtime_semantics": "smc_trader/semantics.py",
        "runtime_event_store": "smc_trader/event_store.py",
        "runtime_market_state": "smc_trader/market_state.py",
        "runtime_structure": "smc_trader/structure.py",
        "runtime_liquidity": "smc_trader/liquidity.py",
        "runtime_displacement": "smc_trader/displacement.py",
        "runtime_displacement_observer": "smc_trader/displacement_observer.py",
        "runtime_group3": "smc_trader/zone.py",
        "runtime_group4": "smc_trader/range_auction.py",
        "runtime_group5": "smc_trader/group5.py",
        "runtime_scene_graph": "smc_trader/scene_graph.py",
        "runtime_signal_research": "smc_trader/signal_research.py",
        "runtime_validation": "smc_trader/validation.py",
    }
    template = json.loads(
        (ROOT / "experiments/manifests/semantic_event_study_v2_template.yaml").read_text(
            encoding="utf-8"
        )
    )
    bindings = template["identity_bindings"]

    assert dict(REQUIRED_RUNTIME_CODE_BINDINGS) == expected_runtime
    assert set(bindings) == REQUIRED_IDENTITY_BINDINGS
    for name, relative_path in REQUIRED_RUNTIME_CODE_BINDINGS.items():
        historical_path = {
            "runtime_group3": "smc_trader/group3.py",
            "runtime_group4": "smc_trader/group4.py",
        }.get(name, relative_path)
        assert bindings[name] == {"path": historical_path, "sha256": None}
        assert (ROOT / relative_path).is_file()


def test_frozen_contract_binds_every_registered_file_exactly(tmp_path: Path) -> None:
    bindings: dict[str, dict[str, str]] = {}
    for name in sorted(REQUIRED_IDENTITY_BINDINGS):
        relative_path = REQUIRED_RUNTIME_CODE_BINDINGS.get(name, f"{name}.json")
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        bindings[name] = {"path": relative_path, "sha256": _sha(path)}
    dataset = tmp_path / "ohlcv.parquet"
    dataset.write_bytes(b"registered-diagnostic-data")
    split = Path(bindings["split_registry"]["path"])
    payload = {
        "schema_version": 2,
        "status": "frozen_development_diagnostic_not_oos_not_trading_authority",
        "frozen_before_run": True,
        "parameter_search_space": {},
        "authority": {
            "diagnostic_only": True,
            "artifact_fit_allowed": False,
            "trading_authority": False,
            "oos_opened": False,
        },
        "out_of_sample_period": "not_opened",
        "semantic_registry_identity": "a" * 64,
        "semantic_registry": bindings["semantic_registry"]["path"],
        "identity_bindings": bindings,
        "dataset_version": {
            "path": dataset.name,
            "sha256": _sha(dataset),
            "manifest_path": bindings["dataset_manifest"]["path"],
            "manifest_sha256": bindings["dataset_manifest"]["sha256"],
            "split_role": "brain_calibration_trial",
            "split_registry": str(split),
            "split_registry_sha256": bindings["split_registry"]["sha256"],
        },
        "warmup_period": {
            "start": "2023-12-24T00:00:00-05:00",
            "outcomes_opened": False,
        },
        "diagnostic_period": {
            "start": "2024-01-01T00:00:00-05:00",
            "end_exclusive": "2024-02-01T00:00:00-05:00",
        },
        "allowed_split_roles": {
            "warmup": ["brain_validation"],
            "diagnostic": ["brain_calibration_trial"],
        },
    }
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    contract = load_frozen_research_contract(
        manifest,
        root=tmp_path,
        actual_semantic_registry_identity="a" * 64,
    )
    assert set(contract.identity_paths) == REQUIRED_IDENTITY_BINDINGS

    missing_runtime = json.loads(json.dumps(payload))
    del missing_runtime["identity_bindings"]["runtime_event_store"]
    manifest.write_text(json.dumps(missing_runtime), encoding="utf-8")
    with pytest.raises(ResearchContractError, match="exact registered research files"):
        load_frozen_research_contract(
            manifest,
            root=tmp_path,
            actual_semantic_registry_identity="a" * 64,
        )

    substituted_runtime = json.loads(json.dumps(payload))
    substituted_runtime["identity_bindings"]["runtime_model"] = dict(
        bindings["runtime_observation"]
    )
    manifest.write_text(json.dumps(substituted_runtime), encoding="utf-8")
    with pytest.raises(ResearchContractError, match="runtime_model must bind"):
        load_frozen_research_contract(
            manifest,
            root=tmp_path,
            actual_semantic_registry_identity="a" * 64,
        )

    for forbidden_role in ("rolling_oof", "sealed_holdout"):
        forbidden = json.loads(json.dumps(payload))
        forbidden["dataset_version"]["split_role"] = forbidden_role
        manifest.write_text(json.dumps(forbidden), encoding="utf-8")
        with pytest.raises(ResearchContractError, match="brain_calibration_trial"):
            load_frozen_research_contract(
                manifest,
                root=tmp_path,
                actual_semantic_registry_identity="a" * 64,
            )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runtime_path = tmp_path / bindings["runtime_observation"]["path"]
    runtime_bytes = runtime_path.read_bytes()
    runtime_path.write_text("changed runtime", encoding="utf-8")
    with pytest.raises(ResearchContractError, match="runtime_observation"):
        load_frozen_research_contract(
            manifest,
            root=tmp_path,
            actual_semantic_registry_identity="a" * 64,
        )
    runtime_path.write_bytes(runtime_bytes)

    (tmp_path / bindings["model_config"]["path"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ResearchContractError, match="model_config"):
        load_frozen_research_contract(
            manifest,
            root=tmp_path,
            actual_semantic_registry_identity="a" * 64,
        )


def test_link_contract_rejects_wall_clock_or_optional_lineage() -> None:
    template = json.loads(
        (ROOT / "experiments/manifests/semantic_event_study_v2_template.yaml").read_text(
            encoding="utf-8"
        )
    )
    template["event_definition"]["chain_edges"]["E2_to_E3"][
        "maximum_completed_bars"
    ] = "30 minutes"

    with pytest.raises(ResearchContractError, match="completed-bar"):
        _validated_link_contracts(template)


def test_non_nested_mss_comparison_requires_prior_displacement() -> None:
    template = json.loads(
        (ROOT / "experiments/manifests/semantic_event_study_v2_template.yaml").read_text(
            encoding="utf-8"
        )
    )

    _, non_nested = _validated_link_contracts(template)

    assert "displacement_prior_mss" not in non_nested
    assert non_nested["mss_prior_displacement"] == {
        "previous_kind": "displacement_observed",
        "previous_timeframe": "5m",
        "current_kind": "mss_core_confirmed",
        "current_timeframe": "5m",
        "maximum_completed_bars": 30,
        "source_lineage_required": True,
    }


def test_research_design_requires_matched_cohort_and_persistent_ledgers() -> None:
    template = json.loads(
        (ROOT / "experiments/manifests/semantic_event_study_v2_template.yaml").read_text(
            encoding="utf-8"
        )
    )
    template["experiment_id"] = "diagnostic_contract_test"
    template["frozen_at"] = "2026-08-20T00:00:00-04:00"

    assert set(template["event_definition"]["atomic_population"]) == (
        REGISTERED_ATOMIC_POPULATION
    )
    assert template["event_definition"]["signal_direction_assignment"] == (
        SIGNAL_DIRECTION_ASSIGNMENT
    )
    assert template["event_definition"]["research_selection"] == (
        RESEARCH_SELECTION
    )
    assert template["event_definition"]["range_invalidation_variants"] == (
        RANGE_INVALIDATION_VARIANT_CONTRACT
    )
    assert template["event_definition"]["range_invalidation_pooling"] == (
        RANGE_INVALIDATION_POOLING
    )
    assert template["event_definition"]["parent_relation_priority"] == (
        PARENT_RELATION_PRIORITY
    )
    assert template["event_definition"]["snapshot_authority"] == (
        SNAPSHOT_AUTHORITY
    )
    assert template["input_census"]["diagnostic_data_gap_policy"] == (
        DIAGNOSTIC_DATA_GAP_POLICY
    )
    assert template["secondary_outcomes"] == SECONDARY_OUTCOME_DEFINITIONS
    assert template["limitations"] == list(REQUIRED_RESEARCH_LIMITATIONS)
    for name, value in PRIMARY_PATH_SCAN_CONTRACT.items():
        assert template["primary_outcome"][name] == value

    changed_population = json.loads(json.dumps(template))
    changed_population["event_definition"]["atomic_population"].append(
        "fvg_touched"
    )
    with pytest.raises(ResearchContractError, match="atomic population"):
        _validated_research_design(changed_population)

    changed_selection = json.loads(json.dumps(template))
    changed_selection["event_definition"]["research_selection"][
        "displacement_observed"
    ]["predicate"] = "evidence.lifecycle == 'started'"
    with pytest.raises(ResearchContractError, match="research selection"):
        _validated_research_design(changed_selection)

    changed_range_variant = json.loads(json.dumps(template))
    changed_range_variant["event_definition"]["range_invalidation_variants"][
        "active_acceptance"
    ]["required_source_kinds"] = ["dealing_range_created", "bar_completed"]
    with pytest.raises(ResearchContractError, match="range invalidation"):
        _validated_research_design(changed_range_variant)

    outcome_parameters, minimum = _validated_research_design(template)

    assert outcome_parameters == {
        "horizon": 60,
        "target_atr": 1.0,
        "invalidation_atr": 1.0,
    }
    assert minimum == 30

    template["control_definition"]["same_matched_cohort_required"] = False
    with pytest.raises(ResearchContractError, match="matched-control"):
        _validated_research_design(template)


def test_research_design_freezes_direction_and_secondary_outcome_definitions() -> None:
    template = json.loads(
        (ROOT / "experiments/manifests/semantic_event_study_v2_template.yaml").read_text(
            encoding="utf-8"
        )
    )
    template["experiment_id"] = "diagnostic_contract_test"
    template["frozen_at"] = "2026-08-20T00:00:00-04:00"

    changed_direction = json.loads(json.dumps(template))
    changed_direction["event_definition"]["signal_direction_assignment"][
        "level_touched_above"
    ] = "long"
    with pytest.raises(ResearchContractError, match="direction assignment"):
        _validated_research_design(changed_direction)

    changed_parent_priority = json.loads(json.dumps(template))
    changed_parent_priority["event_definition"]["parent_relation_priority"][
        "5m"
    ] = ["15m", "1h"]
    with pytest.raises(ResearchContractError, match="parent relation priority"):
        _validated_research_design(changed_parent_priority)

    changed_snapshot_authority = json.loads(json.dumps(template))
    changed_snapshot_authority["event_definition"]["snapshot_authority"] = (
        "frame_projection"
    )
    with pytest.raises(ResearchContractError, match="snapshot authority"):
        _validated_research_design(changed_snapshot_authority)

    changed_census = json.loads(json.dumps(template))
    changed_census["input_census"]["diagnostic_data_gap_policy"] = "allow"
    with pytest.raises(ResearchContractError, match="input census"):
        _validated_research_design(changed_census)

    changed_outcome = json.loads(json.dumps(template))
    changed_outcome["secondary_outcomes"]["mfe_atr"] = "changed"
    with pytest.raises(ResearchContractError, match="secondary outcome"):
        _validated_research_design(changed_outcome)

    changed_limitation = json.loads(json.dumps(template))
    changed_limitation["limitations"] = []
    with pytest.raises(ResearchContractError, match="limitations"):
        _validated_research_design(changed_limitation)


def test_full_input_census_rejects_natural_eof_and_accepts_exact_window() -> None:
    expected = {
        "expected_emitted_bars_including_warmup": 10,
        "expected_diagnostic_completed_bars": 6,
        "expected_diagnostic_real_rows": 5,
        "expected_diagnostic_ready_real_rows": 5,
        "expected_diagnostic_synthetic_bars": 1,
        "expected_warmup_data_gap_resets": 0,
        "expected_diagnostic_data_gap_resets": 0,
        "expected_contract_changes": 0,
        "expected_last_processed_asof": _clock(10),
        "expected_last_diagnostic_asof": _clock(9),
    }
    actual = {
        "emitted_bars_including_warmup": 10,
        "diagnostic_completed_bars": 6,
        "diagnostic_real_rows": 5,
        "diagnostic_ready_real_rows": 5,
        "diagnostic_synthetic_bars": 1,
        "warmup_data_gap_resets": 0,
        "diagnostic_data_gap_resets": 0,
        "contract_changes": 0,
        "last_processed_asof": _clock(10),
        "last_diagnostic_asof": _clock(9),
    }

    _assert_full_input_census(expected, actual)

    incomplete = {**actual, "emitted_bars_including_warmup": 9}
    with pytest.raises(ResearchContractError, match="emitted_bars"):
        _assert_full_input_census(expected, incomplete)


def test_directional_event_without_exact_real_entry_row_fails_closed() -> None:
    event = {
        "event_id": "event:synthetic-clock",
        "known_at": _clock(1),
        "direction": "long",
        "symbol": "NQ",
        "instrument_id": 1,
    }
    rows = [
        {
            "asof": _clock(0),
            "symbol": "NQ",
            "instrument_id": 1,
        }
    ]

    with pytest.raises(ResearchContractError, match="exact real known_at row"):
        _validate_directional_entry_rows([event], rows, {_clock(0): 0})


def test_any_max_bars_run_remains_an_incomplete_smoke_at_natural_eof() -> None:
    manifest_status = "frozen_development_diagnostic_not_oos_not_trading_authority"

    assert _artifact_status(manifest_status, None) == manifest_status
    assert _artifact_status(manifest_status, 36000) == (
        "incomplete_smoke_not_experiment_result"
    )
