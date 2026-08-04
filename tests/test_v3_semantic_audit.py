from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

import smc_trader.semantic_audit as semantic_audit_module
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    ExecutionObservation,
    FrameObservation,
    MarketObservation,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
    Timeframe,
    content_hash,
    to_primitive,
)
from smc_trader.semantic_audit import (
    _lock_blind_review_transaction,
    FixedSemanticCaseSelector,
    SemanticCase,
    SemanticCaseVisualizer,
    classify_bos_case,
    compute_semantic_review_gate,
    load_audit_contract,
    lock_blind_review,
    lock_review_set,
    materialize_blind_review_export,
    materialize_blind_unit,
    materialize_semantic_review_scorecard,
    materialize_semantic_truth_overlay,
    selection_score,
    semantic_clock_certificate,
    semantic_event_id,
)
from smc_trader.semantic_discovery_runner import CasePacketTransaction


def _bos(
    *,
    asof: pd.Timestamp,
    lifecycle: BOSLifecycle,
    scope: BOSScope,
    suffix: str,
) -> BreakOfStructureState:
    pending_at = asof - pd.Timedelta(minutes=2)
    return BreakOfStructureState(
        bos_id=f"bos-{suffix}",
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        lifecycle=lifecycle,
        scope=scope,
        target_swing_id=f"swing-{suffix}",
        source_structure_id=(
            None if scope is BOSScope.LOCAL else f"structure-{suffix}"
        ),
        target_price=100.0,
        target_ticks=400,
        pending_at=pending_at,
        resolved_at=(
            None if lifecycle is BOSLifecycle.PENDING else asof
        ),
        age_bars=2,
        attempt_count=(
            1 if lifecycle is BOSLifecycle.PENDING else 0
        ),
        last_attempt_at=(
            asof if lifecycle is BOSLifecycle.PENDING else None
        ),
        attempt_clocks=(
            (asof,)
            if lifecycle is BOSLifecycle.PENDING
            else ()
        ),
        failure_reason=(
            "data_gap_reset"
            if lifecycle is BOSLifecycle.FAILED
            else None
        ),
    )


def _observation(
    asof: pd.Timestamp,
    *,
    bos: BreakOfStructureState | None = None,
) -> MarketObservation:
    swing = None
    structure = None
    if bos is not None:
        swing = SwingPoint(
            swing_id=bos.target_swing_id,
            timeframe=bos.timeframe,
            symbol="NQH7",
            instrument_id=1,
            side=(
                SwingSide.HIGH
                if bos.direction is Direction.LONG
                else SwingSide.LOW
            ),
            price=bos.target_price,
            price_ticks=bos.target_ticks,
            pivot_start=asof - pd.Timedelta(minutes=5),
            pivot_end=asof - pd.Timedelta(minutes=4),
            observed_at=asof - pd.Timedelta(minutes=3),
            confirmed_at=asof - pd.Timedelta(minutes=3),
            lifecycle=(
                SwingLifecycle.BROKEN
                if bos.lifecycle is BOSLifecycle.CONFIRMED
                else SwingLifecycle.CONFIRMED
            ),
            relation=(
                SwingRelation.HH
                if bos.direction is Direction.LONG
                else SwingRelation.LL
            ),
            age_bars=3,
            broken_at=(
                asof
                if bos.lifecycle is BOSLifecycle.CONFIRMED
                else None
            ),
            failure_reason=(
                "close_beyond_swing"
                if bos.lifecycle is BOSLifecycle.CONFIRMED
                else None
            ),
        )
        if bos.source_structure_id is not None:
            source_direction = (
                (
                    Direction.SHORT
                    if bos.direction is Direction.LONG
                    else Direction.LONG
                )
                if bos.scope is BOSScope.OPPOSED
                else bos.direction
            )
            broken = bos.scope is BOSScope.OPPOSED
            structure = StructureSequenceState(
                structure_id=bos.source_structure_id,
                timeframe=bos.timeframe,
                direction=source_direction,
                lifecycle=(
                    StructureLifecycle.BROKEN
                    if broken
                    else StructureLifecycle.CONFIRMED
                ),
                formed_at=asof - pd.Timedelta(minutes=8),
                confirmed_at=asof - pd.Timedelta(minutes=6),
                broken_at=asof if broken else None,
                high_run=2,
                low_run=2,
                sequence_count=2,
                latest_high_id=(
                    swing.swing_id
                    if swing.side is SwingSide.HIGH
                    else "fixture-high"
                ),
                latest_low_id=(
                    swing.swing_id
                    if swing.side is SwingSide.LOW
                    else "fixture-low"
                ),
                protected_swing_id="fixture-protected",
                protected_price=99.0,
                cumulative_magnitude_atr=2.0,
                age_bars=4,
                failure_reason=(
                    "protected_level_close_break" if broken else None
                ),
            )
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=asof,
            bars=20,
            metrics={},
            ready=True,
            swings=(
                (swing,)
                if swing is not None and swing.timeframe is timeframe
                else ()
            ),
            structures=(
                (structure,)
                if (
                    structure is not None
                    and structure.timeframe is timeframe
                )
                else ()
            ),
            structure_breaks=(
                (bos,)
                if bos is not None and bos.timeframe is timeframe
                else ()
            ),
        )
        for timeframe in Timeframe
    }
    return MarketObservation(
        asof=asof,
        symbol="NQH7",
        instrument_id=1,
        price=100.0,
        frames=frames,
        recent_events=(),
        event_durations_minutes={},
        execution=ExecutionObservation(
            spread_points=0.25,
            expected_slippage_points=0.0,
            expected_round_trip_cost_points=0.25,
            minutes_to_deadline=60,
            fillability=1.0,
            data_age_seconds=0.0,
            size_available=None,
            source="semantic-test",
        ),
    )


def _histories(asof: pd.Timestamp):
    minutes = {
        Timeframe.H4: 240,
        Timeframe.H1: 60,
        Timeframe.M5: 5,
        Timeframe.M1: 1,
    }
    histories = {
        timeframe: (
            Candle(
                timeframe=timeframe,
                start=asof - pd.Timedelta(minutes=duration),
                end=asof,
                open=99.5,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=10.0,
                symbol="NQH7",
                instrument_id=1,
                observed_minutes=duration,
                expected_minutes=duration,
                complete=True,
            ),
        )
        for timeframe, duration in minutes.items()
    }
    histories[Timeframe.M1] = tuple(
        Candle(
            timeframe=Timeframe.M1,
            start=asof - pd.Timedelta(minutes=6 - index),
            end=asof - pd.Timedelta(minutes=5 - index),
            open=99.5,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
            symbol="NQH7",
            instrument_id=1,
            observed_minutes=1,
            expected_minutes=1,
            complete=True,
        )
        for index in range(6)
    )
    return histories


def _contract():
    return load_audit_contract(verify_bound_files=False)


def _single_case_contract() -> dict:
    contract = _contract()
    return {
        **contract,
        "audit_id": f"{contract['audit_id']}-SYNTHETIC-ONE-CASE",
        "selection": {
            "timeframes": ["1m"],
            "directions": ["long"],
            "case_classes": ["confirmed_bos"],
            "calendar_years": [2017],
            "cases_per_bucket": 1,
            "bucket_count": 1,
            "total_cases": 1,
        },
        "release_gates": {
            **contract["release_gates"],
            "formula_and_causal_agreement": "1/1",
            "blind_strict_case_agreement": (
                "at least 1/1; missing and uncertain count as disagreement"
            ),
        },
    }


def _selected_case(
    item: BreakOfStructureState,
    *,
    asof: pd.Timestamp,
    case_class: str,
) -> dict:
    event_id = semantic_event_id(
        item,
        case_class=case_class,
        case_clock=asof,
    )
    value = SemanticCase(
        semantic_event_id=event_id,
        bos_id=item.bos_id,
        timeframe=item.timeframe,
        direction=item.direction,
        case_class=case_class,
        calendar_year=asof.year,
        case_clock=asof,
        selection_score=selection_score(
            _contract()["bindings"]["primitive_protocol_sha256"],
            event_id,
        ),
        semantic_state_hash=content_hash(item),
        target_swing_id=item.target_swing_id,
        source_structure_id=item.source_structure_id,
    )
    return to_primitive(value)


@pytest.mark.historical_frozen
def test_frozen_contract_bindings_match_current_semantic_core() -> None:
    contract = load_audit_contract()
    assert contract["selection"]["total_cases"] == 240
    assert contract["completed_candle_interpretation"]["policy"] == (
        "real_completed_only_with_registered_synthetic_clock"
    )


def test_unauthorized_causal_source_is_never_hashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    forbidden = (
        semantic_audit_module.ROOT
        / contract["source"]["path"]
    ).resolve()
    original = semantic_audit_module.sha256_file
    frozen_bound_hashes = {
        (
            semantic_audit_module.ROOT / relative
        ).resolve(): contract["bindings"][name]
        for name, relative in contract["bound_files"].items()
    }

    def guarded(path):
        resolved = Path(path).resolve()
        if resolved == forbidden:
            raise AssertionError("unauthorized causal source was touched")
        if resolved in frozen_bound_hashes:
            return frozen_bound_hashes[resolved]
        return original(path)

    monkeypatch.setattr(
        semantic_audit_module,
        "sha256_file",
        guarded,
    )
    loaded = load_audit_contract()
    assert loaded["source"]["access_authorized"] is False


def test_formal_216_of_240_gate_has_exact_boundaries() -> None:
    contract = _contract()
    assert compute_semantic_review_gate(
        contract,
        case_count=240,
        strict_case_agreement_count=215,
    )["strict_case_agreement_passed"] is False
    assert compute_semantic_review_gate(
        contract,
        case_count=240,
        strict_case_agreement_count=216,
    )["strict_case_agreement_passed"] is True
    assert compute_semantic_review_gate(
        contract,
        case_count=240,
        strict_case_agreement_count=217,
    )["strict_case_agreement_passed"] is True
    with pytest.raises(ValueError, match="not computable"):
        compute_semantic_review_gate(
            contract,
            case_count=239,
            strict_case_agreement_count=216,
        )


def test_forming_swing_is_included_in_raw_recoverable_truth() -> None:
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    histories = _histories(asof)
    h1 = histories[Timeframe.H1][0]
    forming = SwingPoint(
        swing_id="forming-h1",
        timeframe=Timeframe.H1,
        symbol="NQH7",
        instrument_id=1,
        side=SwingSide.HIGH,
        price=101.0,
        price_ticks=404,
        pivot_start=h1.start,
        pivot_end=h1.end,
        observed_at=h1.end,
        confirmed_at=None,
        lifecycle=SwingLifecycle.FORMING,
    )
    observation = _observation(asof)
    observation = replace(
        observation,
        frames={
            **observation.frames,
            Timeframe.H1: replace(
                observation.frame(Timeframe.H1),
                swings=(forming,),
            ),
        },
    )
    raw = semantic_audit_module._blind_raw_evidence(
        histories,
        opaque_case_id="forming-case",
        case_clock=asof,
        tick_size=0.25,
        prefix_source_sha256="0" * 64,
        history_capacity=1024,
        reset_epoch=0,
        last_reset_at=None,
        last_reset_reason=None,
    )
    truth = semantic_audit_module._enumerated_structure_truth(
        observation,
        raw,
    )
    assert truth["swings"] == [
        {
            "timeframe": "1H",
            "side": "high",
            "pivot_index": 0,
            "prior_same_side_index": None,
            "relation": "none",
            "observed_index": 0,
            "confirmation_index": None,
            "broken_index": None,
            "lifecycle": "forming",
            "failure_reason": None,
        }
    ]
    assert truth["context"]["context_truncated"] is False


def test_evicted_prior_does_not_silently_drop_current_swing() -> None:
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    histories = _histories(asof)
    m1 = histories[Timeframe.M1]
    pivot = m1[2]
    confirmed_at = m1[4].end
    current = SwingPoint(
        swing_id="current-m1",
        timeframe=Timeframe.M1,
        symbol="NQH7",
        instrument_id=1,
        side=SwingSide.HIGH,
        price=101.0,
        price_ticks=404,
        pivot_start=pivot.start,
        pivot_end=pivot.end,
        observed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SwingLifecycle.CONFIRMED,
        prior_same_side_id="evicted-prior-h1",
        relation=SwingRelation.HH,
    )
    observation = _observation(asof)
    observation = replace(
        observation,
        frames={
            **observation.frames,
            Timeframe.M1: replace(
                observation.frame(Timeframe.M1),
                swings=(current,),
            ),
        },
    )
    raw = semantic_audit_module._blind_raw_evidence(
        histories,
        opaque_case_id="evicted-prior-case",
        case_clock=asof,
        tick_size=0.25,
        prefix_source_sha256="0" * 64,
        history_capacity=1024,
        reset_epoch=0,
        last_reset_at=None,
        last_reset_reason=None,
    )
    truth = semantic_audit_module._enumerated_structure_truth(
        observation,
        raw,
    )
    assert len(truth["swings"]) == 1
    assert truth["swings"][0]["prior_same_side_index"] is None
    assert (
        truth["context"]["unavailable_prior_same_side_count"] == 1
    )
    assert truth["context"]["context_truncated"] is True


def test_reviewer_physical_creation_order_uses_only_random_identity() -> None:
    event_order = [
        {"semantic_event_id": "event-a"},
        {"semantic_event_id": "event-b"},
        {"semantic_event_id": "event-c"},
    ]
    authority = {
        "event-a": {"review_unit_id": "f" * 64},
        "event-b": {"review_unit_id": "8" * 64},
        "event-c": {"review_unit_id": "0" * 64},
    }
    ordered = semantic_audit_module._reviewer_export_creation_order(
        event_order,
        authority,
    )
    assert [
        item["semantic_event_id"] for item in ordered
    ] == ["event-c", "event-b", "event-a"]


def test_visual_coverage_maps_old_cross_timeframe_and_attempt_markers() -> None:
    raw = {
        "panels": {
            timeframe.value: {
                "bars": [
                    {"bar_index": index}
                    for index in range(100)
                ]
            }
            for timeframe in Timeframe
        }
    }
    swings = [
        {
            "timeframe": "4H",
            "side": "high",
            "pivot_index": 0,
            "prior_same_side_index": None,
            "relation": "none",
            "observed_index": 0,
            "confirmation_index": None,
            "broken_index": None,
            "lifecycle": "forming",
            "failure_reason": None,
        }
    ]
    bos_candidates = [
        {
            "timeframe": "1m",
            "direction": "long",
            "target_pivot_index": 1,
            "target_price_ticks": 400,
            "pending_index": 2,
            "wick_attempt_indices": [3, 75],
            "resolved_index": 99,
            "lifecycle": "failed",
            "scope": "local",
            "failure_reason": "superseded",
        }
    ]
    coverage = semantic_audit_module._semantic_visual_coverage(
        raw,
        {
            "swings": swings,
            "bos_candidates": bos_candidates,
        },
    )
    assert coverage["all_typed_candidates_in_index_maps"] is True
    assert coverage["price_panels"]["4H"] == {
        "first_history_index": 80,
        "last_history_index": 99,
        "bar_count": 20,
        "maximum_bars": 20,
        "axis_compressed": False,
    }
    assert coverage["candidate_index_maps"]["4H"][
        "swing_candidates"
    ][0]["pivot_index"] == 0
    assert coverage["candidate_index_maps"]["1m"][
        "bos_candidates"
    ][0]["wick_attempt_indices"] == [3, 75]


def test_case_classes_are_mutually_exclusive_and_score_is_exact() -> None:
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    opposed = _bos(
        asof=asof,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.OPPOSED,
        suffix="opposed",
    )
    continuation = _bos(
        asof=asof,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        suffix="continuation",
    )
    pending = _bos(
        asof=asof,
        lifecycle=BOSLifecycle.PENDING,
        scope=BOSScope.CONTINUATION,
        suffix="wick",
    )
    assert classify_bos_case(opposed, case_clock=asof) == (
        "broken_or_opposed"
    )
    assert classify_bos_case(continuation, case_clock=asof) == (
        "confirmed_bos"
    )
    assert classify_bos_case(pending, case_clock=asof) == (
        "wick_only_no_close"
    )
    primitive_hash = _contract()["bindings"]["primitive_protocol_sha256"]
    event_id = semantic_event_id(
        continuation,
        case_class="confirmed_bos",
        case_clock=asof,
    )
    import hashlib

    assert selection_score(primitive_hash, event_id) == hashlib.sha256(
        f"{primitive_hash}|{event_id}".encode("utf-8")
    ).hexdigest()


def test_selector_is_order_independent_and_shortage_fails_closed() -> None:
    contract = _contract()
    candidates = []
    for minute in (3, 1, 2):
        asof = pd.Timestamp(
            f"2017-01-05 10:0{minute}",
            tz="America/New_York",
        )
        candidates.append(
            _observation(
                asof,
                bos=_bos(
                    asof=asof,
                    lifecycle=BOSLifecycle.CONFIRMED,
                    scope=BOSScope.CONTINUATION,
                    suffix=str(minute),
                ),
            )
        )
    first = FixedSemanticCaseSelector(contract)
    second = FixedSemanticCaseSelector(contract)
    for item in candidates:
        first.observe(item)
    for item in reversed(candidates):
        second.observe(item)
    kwargs = {
        "source_sha256": contract["bindings"]["causal_source_sha256"],
        "source_start": pd.Timestamp(contract["source"]["start"]),
        "source_end_exclusive": pd.Timestamp(
            contract["source"]["end_exclusive"]
        ),
        "implementation_hashes": {"test": "0" * 64},
    }
    first_manifest = first.manifest(**kwargs)
    second_manifest = second.manifest(**kwargs)
    assert first_manifest["cases"] == second_manifest["cases"]
    assert first_manifest["status"] == "unavailable"
    assert first_manifest["selected_case_count"] == 2
    assert len(first_manifest["missing_buckets"]) == 119


def test_semantic_clock_certificate_rejects_future_input() -> None:
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    observation = _observation(asof)
    certificate = semantic_clock_certificate(
        observation,
        _histories(asof),
        case_clock=asof,
        max_source_time_loaded=asof - pd.Timedelta(minutes=1),
        max_engine_time_processed=asof,
        prefix_source_sha256="0" * 64,
    )
    assert certificate["maximum_market_time"] == asof
    with pytest.raises(ValueError, match="source row start"):
        semantic_clock_certificate(
            observation,
            _histories(asof),
            case_clock=asof,
            max_source_time_loaded=asof,
            max_engine_time_processed=asof,
            prefix_source_sha256="0" * 64,
        )
    future = dict(_histories(asof))
    future[Timeframe.M1] = (
        *future[Timeframe.M1],
        Candle(
            timeframe=Timeframe.M1,
            start=asof,
            end=asof + pd.Timedelta(minutes=1),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
            symbol="NQH7",
            instrument_id=1,
            observed_minutes=1,
            expected_minutes=1,
            complete=True,
        ),
    )
    with pytest.raises(ValueError, match="future candles"):
        semantic_clock_certificate(
            observation,
            future,
            case_clock=asof,
            max_source_time_loaded=asof - pd.Timedelta(minutes=1),
            max_engine_time_processed=asof,
            prefix_source_sha256="0" * 64,
        )


def test_blind_image_does_not_depend_on_semantic_truth(tmp_path) -> None:
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    histories = _histories(asof)
    plain = _observation(asof)
    labeled = _observation(
        asof,
        bos=_bos(
            asof=asof,
            lifecycle=BOSLifecycle.CONFIRMED,
            scope=BOSScope.CONTINUATION,
            suffix="truth",
        ),
    )
    visualizer = SemanticCaseVisualizer()
    first = visualizer.render_blind(
        plain,
        histories,
        tmp_path / "plain.png",
        opaque_case_id="opaque",
        tick_size=0.25,
    )
    second = visualizer.render_blind(
        labeled,
        histories,
        tmp_path / "labeled.png",
        opaque_case_id="opaque",
        tick_size=0.25,
    )
    assert first.sha256 == second.sha256
    assert first.maximum_market_time == asof


def _packet(tmp_path: Path):
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    item = _bos(
        asof=asof,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        suffix="packet",
    )
    observation = _observation(asof, bos=item)
    authority = tmp_path / "authority"
    blind = tmp_path / "blind"
    materialize_blind_unit(
        contract=_contract(),
        selected_case=_selected_case(
            item,
            asof=asof,
            case_class="confirmed_bos",
        ),
        observation=observation,
        histories=_histories(asof),
        authority_root=authority,
        blind_root=blind,
        max_source_time_loaded=asof - pd.Timedelta(minutes=1),
        max_engine_time_processed=asof,
        prefix_source_sha256="0" * 64,
        implementation_hashes={"semantic_audit": "0" * 64},
        tick_size=0.25,
        history_capacity=1024,
        reset_epoch=0,
        last_reset_at=None,
        last_reset_reason=None,
    )
    return authority, blind


def test_pass2_rebuilds_selected_bos_metadata_from_replay(
    tmp_path: Path,
) -> None:
    asof = pd.Timestamp("2017-01-05 10:00", tz="America/New_York")
    item = _bos(
        asof=asof,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        suffix="metadata",
    )
    tampered = {
        **_selected_case(
            item,
            asof=asof,
            case_class="confirmed_bos",
        ),
        "target_swing_id": "forged-target",
    }
    with pytest.raises(ValueError, match="selected metadata"):
        materialize_blind_unit(
            contract=_contract(),
            selected_case=tampered,
            observation=_observation(asof, bos=item),
            histories=_histories(asof),
            authority_root=tmp_path / "authority",
            blind_root=tmp_path / "blind",
            max_source_time_loaded=asof - pd.Timedelta(minutes=1),
            max_engine_time_processed=asof,
            prefix_source_sha256="0" * 64,
            implementation_hashes={"semantic_audit": "0" * 64},
            tick_size=0.25,
            history_capacity=1024,
            reset_epoch=0,
            last_reset_at=None,
            last_reset_reason=None,
        )


def test_authority_and_blind_packet_are_separated(tmp_path) -> None:
    authority, blind = _packet(tmp_path)
    authority_payload = json.loads(
        (authority / "authority.json").read_text(encoding="utf-8")
    )
    blind_payload = json.loads(
        (blind / "blind_manifest.json").read_text(encoding="utf-8")
    )
    assert authority_payload["selected_case"]["case_class"] == (
        "confirmed_bos"
    )
    serialized_blind = json.dumps(blind_payload, sort_keys=True)
    for forbidden in ("case_class", "bos_id", "direction", "timeframe"):
        assert forbidden not in serialized_blind
    assert blind_payload["maximum_market_time"] <= blind_payload["case_clock"]
    raw = json.loads(
        (blind / "blind_raw_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    m1 = raw["panels"]["1m"]["bars"]
    assert [item["bar_index"] for item in m1] == list(range(6))
    assert all(item["real_completed"] for item in m1)
    assert all(item["synthetic_minutes"] == 0 for item in m1)
    assert raw["reset_epoch"] == 0


def test_review_lock_is_hash_bound_create_once_and_action_free(tmp_path) -> None:
    _, blind = _packet(tmp_path)
    review_root = tmp_path / "reviews" / "unit"
    packet_bytes = {
        path.relative_to(blind): path.read_bytes()
        for path in blind.iterdir()
        if path.is_file()
    }
    template = json.loads(
        (blind / "review_template.json").read_text(encoding="utf-8")
    )
    review = {
        **template,
        "reviewer_id": "semantic-reviewer",
        "reviewed_at": "2026-07-28T12:00:00-04:00",
        "no_future_attestation": True,
        "judgment": {
            "swings": [
                {
                    "timeframe": "1m",
                    "side": "high",
                    "pivot_index": 1,
                    "prior_same_side_index": None,
                    "relation": "HH",
                    "observed_index": 2,
                    "confirmation_index": 2,
                    "broken_index": 5,
                    "lifecycle": "broken",
                    "failure_reason": "close_beyond_swing",
                }
            ],
            "bos_candidates": [
                {
                    "timeframe": "1m",
                    "direction": "long",
                    "target_pivot_index": 1,
                    "target_price_ticks": 400,
                    "pending_index": 3,
                    "wick_attempt_indices": [],
                    "resolved_index": 5,
                    "lifecycle": "confirmed",
                    "scope": "continuation",
                    "failure_reason": None,
                }
            ],
            "confidence": 0.8,
            "issue_codes": [],
        },
    }
    prohibited = dict(review)
    prohibited["reviewer_id"] = "should enter"
    with pytest.raises(ValueError, match="action language"):
        _lock_blind_review_transaction(
            blind,
            tmp_path / "reviews" / "prohibited",
            prohibited,
            contract=_contract(),
        )
    invalid_same_clock = json.loads(json.dumps(review))
    invalid_same_clock["judgment"]["bos_candidates"][0].update(
        {
            "wick_attempt_indices": [5],
            "lifecycle": "failed",
            "failure_reason": "data_gap_reset",
        }
    )
    with pytest.raises(ValueError, match="same-clock"):
        _lock_blind_review_transaction(
            blind,
            tmp_path / "reviews" / "invalid-same-clock",
            invalid_same_clock,
            contract=_contract(),
        )
    lock = _lock_blind_review_transaction(
        blind,
        review_root,
        review,
        contract=_contract(),
    )
    assert lock.is_file()
    assert {
        path.relative_to(blind): path.read_bytes()
        for path in blind.iterdir()
        if path.is_file()
    } == packet_bytes
    with pytest.raises(FileExistsError):
        _lock_blind_review_transaction(
            blind,
            review_root,
            review,
            contract=_contract(),
        )


def test_tampered_blind_image_cannot_be_reviewed(tmp_path) -> None:
    _, blind = _packet(tmp_path)
    with (blind / "case.png").open("ab") as handle:
        handle.write(b"tamper")
    template = json.loads(
        (blind / "review_template.json").read_text(encoding="utf-8")
    )
    with pytest.raises(
        ValueError,
        match="blind packet artifact changed: case.png",
    ):
        _lock_blind_review_transaction(
            blind,
            tmp_path / "reviews" / "tampered",
            {
                **template,
                "reviewer_id": "reviewer",
                "reviewed_at": "2026-07-28T12:00:00-04:00",
                "no_future_attestation": True,
            },
            contract=_contract(),
        )


def test_exact_blind_set_lock_and_truth_overlay_are_pass2_bound(
    tmp_path,
) -> None:
    contract = _single_case_contract()
    engine_root = tmp_path / "engine-output"
    reviewer_root = tmp_path / "reviewer-output"
    reviews_root = tmp_path / "review-transactions"
    truth_root = tmp_path / "truth-output"
    governance_root = tmp_path / "governance"
    engine_root.mkdir()
    asof = pd.Timestamp(
        "2017-01-05 10:00",
        tz="America/New_York",
    )
    item = _bos(
        asof=asof,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        suffix="set",
    )
    secondary = replace(
        _bos(
            asof=asof,
            lifecycle=BOSLifecycle.CONFIRMED,
            scope=BOSScope.LOCAL,
            suffix="set-secondary",
        ),
        direction=Direction.SHORT,
    )
    primary_observation = _observation(asof, bos=item)
    secondary_observation = _observation(asof, bos=secondary)
    secondary_swing = replace(
        secondary_observation.frame(Timeframe.M1).swings[0],
        pivot_start=asof - pd.Timedelta(minutes=6),
        pivot_end=asof - pd.Timedelta(minutes=5),
        observed_at=asof - pd.Timedelta(minutes=4),
        confirmed_at=asof - pd.Timedelta(minutes=4),
    )
    m1_frame = primary_observation.frame(Timeframe.M1)
    observation = replace(
        primary_observation,
        frames={
            **primary_observation.frames,
            Timeframe.M1: replace(
                m1_frame,
                swings=(
                    *m1_frame.swings,
                    secondary_swing,
                ),
                structure_breaks=(
                    *m1_frame.structure_breaks,
                    secondary,
                ),
            ),
        },
    )
    selected = {
        **_selected_case(
            item,
            asof=asof,
            case_class="confirmed_bos",
        ),
        "case_bar_synthetic": False,
        "case_source_row_start": asof - pd.Timedelta(minutes=1),
        "case_source_row_ordinal": 5,
        "case_source_row_sha256": "6" * 64,
        "case_bar_sha256": "7" * 64,
        "source_prefix_root": "8" * 64,
        "produced_bar_prefix_root": "9" * 64,
        "source_rows_admitted": 6,
        "produced_bars": 6,
        "reset_epoch": 0,
    }
    event_id = selected["semantic_event_id"]
    implementation_hashes = {
        "semantic_discovery_runner_sha256": "1" * 64,
        "semantic_audit_sha256": "2" * 64,
        "io_sha256": "3" * 64,
        "causal_sha256": "4" * 64,
        "market_clock_sha256": "5" * 64,
        "observation_sha256": "6" * 64,
        "structure_sha256": "7" * 64,
    }
    iterator_bindings = {
        "maximum_history": 1024,
        "maximum_no_trade_gap_minutes": 5,
        "allow_data_gap_reset": True,
        "source_batch_rows": 1024,
    }
    selection = {
        "format_version": 1,
        "artifact": "v3_structure_bos_selection_authority",
        "audit_id": contract["audit_id"],
        "status": "complete",
        "audit_contract_sha256": "4" * 64,
        "runner_contract_sha256": "5" * 64,
        "engine_output_root": str(engine_root.resolve()),
        "source_sha256": "3" * 64,
        "source_start": "2017-01-01T00:00:00-05:00",
        "source_end_exclusive": "2018-01-01T00:00:00-05:00",
        "source_rows": 6,
        "produced_bars": 6,
        "source_prefix_root": "8" * 64,
        "produced_bar_prefix_root": "9" * 64,
        "iterator_bindings": iterator_bindings,
        "implementation_hashes": implementation_hashes,
        "expected_bucket_count": 1,
        "cases_per_bucket": 1,
        "expected_case_count": 1,
        "selected_case_count": 1,
        "missing_buckets": [],
        "cases": [selected],
    }
    selection_path = engine_root / "selection_authority.json"
    selection_path.write_text(
        json.dumps(to_primitive(selection), sort_keys=True),
        encoding="utf-8",
    )
    import hashlib

    selection_sha = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    transaction = CasePacketTransaction(engine_root / "packets")

    def produce(authority: Path, blind: Path) -> None:
        materialize_blind_unit(
            contract=contract,
            selected_case=selected,
            observation=observation,
            histories=_histories(asof),
            authority_root=authority,
            blind_root=blind,
            max_source_time_loaded=(
                asof - pd.Timedelta(minutes=1)
            ),
            max_engine_time_processed=asof,
            prefix_source_sha256="0" * 64,
            implementation_hashes=implementation_hashes,
            tick_size=0.25,
            history_capacity=1024,
            reset_epoch=0,
            last_reset_at=None,
            last_reset_reason=None,
        )

    case_root, completion_sha = transaction.publish(
        semantic_event_id_value=event_id,
        completion_metadata={
            "selection_manifest_sha256": selection_sha,
            "case_clock": asof,
        },
        producer=produce,
    )
    pass2 = {
        "format_version": 1,
        "artifact": "v3_semantic_pass2_manifest",
        "audit_id": contract["audit_id"],
        "engine_output_root": str(engine_root.resolve()),
        "selection_manifest_sha256": selection_sha,
        "source_sha256": "3" * 64,
        "audit_contract_sha256": "4" * 64,
        "runner_contract_sha256": "5" * 64,
        "implementation_hashes": implementation_hashes,
        "iterator_bindings": iterator_bindings,
        "case_count": 1,
        "final_source_rows_admitted": 6,
        "final_produced_bars": 6,
        "final_source_prefix_root": "8" * 64,
        "final_produced_bar_prefix_root": "9" * 64,
        "final_source_row_ordinal": 5,
        "final_source_start": asof - pd.Timedelta(minutes=1),
        "final_causal_clock": asof,
        "final_checkpoint_state_sha256": "a" * 64,
        "case_commit_hashes": {
            event_id: completion_sha,
        },
    }
    pass2_path = engine_root / "PASS2_MANIFEST.json"
    pass2_path.write_text(
        json.dumps(to_primitive(pass2), sort_keys=True),
        encoding="utf-8",
    )
    copied_engine = tmp_path / "copied-engine"
    copied_engine.mkdir()
    copied_selection = copied_engine / selection_path.name
    copied_pass2 = copied_engine / pass2_path.name
    shutil.copyfile(selection_path, copied_selection)
    shutil.copyfile(pass2_path, copied_pass2)
    with pytest.raises(ValueError, match="registered complete selection"):
        materialize_blind_review_export(
            pass2_manifest_path=copied_pass2,
            selection_manifest_path=copied_selection,
            packets_root=transaction.root,
            destination=tmp_path / "copied-reviewer-output",
            contract=contract,
        )
    assert not (tmp_path / "copied-reviewer-output").exists()
    with pytest.raises(ValueError, match="outside immutable"):
        materialize_blind_review_export(
            pass2_manifest_path=pass2_path,
            selection_manifest_path=selection_path,
            packets_root=transaction.root,
            destination=transaction.root / "review-export",
            contract=contract,
        )
    blind_set_path = materialize_blind_review_export(
        pass2_manifest_path=pass2_path,
        selection_manifest_path=selection_path,
        packets_root=transaction.root,
        destination=reviewer_root,
        contract=contract,
    )
    blind_set = json.loads(
        blind_set_path.read_text(encoding="utf-8")
    )
    assert blind_set["case_count"] == 1
    assert "semantic_event_id" not in json.dumps(
        blind_set,
        sort_keys=True,
    )
    assert "case_completion_sha256" not in json.dumps(
        blind_set,
        sort_keys=True,
    )
    assert not any(
        path.name == "authority.json"
        for path in blind_set_path.parent.rglob("*")
    )
    unit = blind_set["units"][0]
    blind_root = blind_set_path.parent / unit["unit_path"]
    assert [
        path.name for path in (blind_set_path.parent / "units").iterdir()
    ] == sorted(
        path.name
        for path in (blind_set_path.parent / "units").iterdir()
    )
    for visible_json in blind_set_path.parent.rglob("*.json"):
        assert event_id not in visible_json.read_text(encoding="utf-8")
        assert completion_sha not in visible_json.read_text(encoding="utf-8")
    public_candidate_ids = {
        hashlib.sha256(
            f"{contract['audit_id']}|{candidate}".encode("utf-8")
        ).hexdigest()
        for candidate in (
            event_id,
            f"{item.bos_id}|wick_only_no_close|{asof.isoformat()}",
            f"{item.bos_id}|broken_or_opposed|{asof.isoformat()}",
        )
    }
    assert unit["opaque_case_id"] not in public_candidate_ids
    assert unit["review_unit_id"] not in public_candidate_ids
    polluted_root = tmp_path / "polluted-reviewer-output"
    shutil.copytree(reviewer_root, polluted_root)
    (polluted_root / "authority.json").write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="authority mapping|top-level tree",
    ):
        lock_review_set(
            [],
            governance_root / "POLLUTED_REVIEWS_LOCK.json",
            blind_set_manifest=(
                polluted_root / "BLIND_SET_MANIFEST.json"
            ),
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            contract=contract,
        )
    template = json.loads(
        (blind_root / "review_template.json").read_text(
            encoding="utf-8"
        )
    )
    authority_payload = json.loads(
        (case_root / "authority" / "authority.json").read_text(
            encoding="utf-8"
        )
    )
    review = {
        **template,
        "reviewer_id": "blind-reviewer",
        "reviewed_at": "2026-07-29T12:00:00-04:00",
        "no_future_attestation": True,
        "judgment": {
            "swings": authority_payload[
                "enumerated_truth"
            ]["swings"],
            "bos_candidates": authority_payload[
                "enumerated_truth"
            ]["bos_candidates"],
            "confidence": 0.5,
            "issue_codes": [],
        },
    }
    with pytest.raises(ValueError, match="outside immutable"):
        lock_blind_review(
            blind_root,
            reviewer_root / "reviews" / unit["review_unit_id"],
            review,
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            contract=contract,
        )
    assert not (reviewer_root / "reviews").exists()
    engine_review = engine_root / "reviews" / unit["review_unit_id"]
    with pytest.raises(ValueError, match="outside immutable"):
        lock_blind_review(
            blind_root,
            engine_review,
            review,
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            contract=contract,
        )
    assert not (engine_root / "reviews").exists()
    review_root = reviews_root / unit["review_unit_id"]
    lock_blind_review(
        blind_root,
        review_root,
        review,
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        contract=contract,
    )
    forged_root = reviews_root / "forged"
    shutil.copytree(review_root, forged_root)
    forged_review_path = forged_root / "review.json"
    forged_review = json.loads(
        forged_review_path.read_text(encoding="utf-8")
    )
    forged_review["judgment"]["swings"] = []
    forged_review_path.write_text(
        json.dumps(forged_review, sort_keys=True),
        encoding="utf-8",
    )
    forged_lock_path = forged_root / "LOCKED_REVIEW.json"
    forged_lock = json.loads(
        forged_lock_path.read_text(encoding="utf-8")
    )
    forged_lock["review_sha256"] = hashlib.sha256(
        forged_review_path.read_bytes()
    ).hexdigest()
    forged_lock_path.write_text(
        json.dumps(forged_lock, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="enumerate"):
        lock_review_set(
            [forged_root],
            governance_root / "FORGED_REVIEWS_LOCK.json",
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            contract=contract,
        )
    with pytest.raises(ValueError, match="all registered"):
        lock_review_set(
            [],
            governance_root / "MISSING_REVIEWS_LOCK.json",
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            contract=contract,
        )
    with pytest.raises(ValueError, match="duplicated"):
        lock_review_set(
            [review_root, review_root],
            governance_root / "DUPLICATE_REVIEWS_LOCK.json",
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            contract=contract,
        )
    global_lock = lock_review_set(
        [review_root],
        governance_root / "ALL_REVIEWS_LOCKED.json",
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        contract=contract,
    )
    with pytest.raises(ValueError, match="outside immutable"):
        materialize_semantic_truth_overlay(
            authority_path=case_root / "authority" / "authority.json",
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            global_review_lock=global_lock,
            review_unit_id=unit["review_unit_id"],
            destination=transaction.root / "truth",
            contract=contract,
        )
    truth_lock = materialize_semantic_truth_overlay(
        authority_path=case_root / "authority" / "authority.json",
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        global_review_lock=global_lock,
        review_unit_id=unit["review_unit_id"],
        destination=truth_root / "exact",
        contract=contract,
    )
    assert truth_lock.is_file()
    truth = json.loads(
        (truth_lock.parent / "semantic_truth.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["future_present"] is False
    assert truth["action_present"] is False
    assert truth["raw_index_mapping"]["target_pivot_index"] == 1
    assert truth["strict_case_agreement"] is True
    assert truth["visualization_coverage"][
        "all_typed_candidates_in_index_maps"
    ] is True
    assert truth["visualization_coverage"][
        "price_axis_compression"
    ] is False
    scorecard_path = materialize_semantic_review_scorecard(
        [truth_lock.parent],
        governance_root / "SEMANTIC_SCORECARD.json",
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        global_review_lock=global_lock,
        contract=contract,
    )
    scorecard = json.loads(
        scorecard_path.read_text(encoding="utf-8")
    )
    assert scorecard["strict_case_agreement_count"] == 1
    assert scorecard["strict_case_agreement_passed"] is True
    omission_review_root = reviews_root / "omission"
    retained_bos = authority_payload[
        "enumerated_truth"
    ]["bos_candidates"][:1]
    retained_swings = [
        value
        for value in authority_payload[
            "enumerated_truth"
        ]["swings"]
        if (
            value["timeframe"],
            value["pivot_index"],
        )
        == (
            retained_bos[0]["timeframe"],
            retained_bos[0]["target_pivot_index"],
        )
    ]
    omission_review = {
        **template,
        "reviewer_id": "omission-reviewer",
        "reviewed_at": "2026-07-29T12:30:00-04:00",
        "no_future_attestation": True,
        "judgment": {
            "swings": retained_swings,
            "bos_candidates": retained_bos,
            "confidence": 0.5,
            "issue_codes": ["MULTIPLE_CANDIDATES_OMITTED"],
        },
    }
    lock_blind_review(
        blind_root,
        omission_review_root,
        omission_review,
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        contract=contract,
    )
    omission_global_lock = lock_review_set(
        [omission_review_root],
        governance_root / "OMISSION_REVIEWS_LOCKED.json",
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        contract=contract,
    )
    omission_truth_lock = materialize_semantic_truth_overlay(
        authority_path=case_root / "authority" / "authority.json",
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        global_review_lock=omission_global_lock,
        review_unit_id=unit["review_unit_id"],
        destination=truth_root / "omission",
        contract=contract,
    )
    omission_truth = json.loads(
        (
            omission_truth_lock.parent / "semantic_truth.json"
        ).read_text(encoding="utf-8")
    )
    assert omission_truth["strict_case_agreement"] is False
    assert omission_truth["computed_issue_codes"] == [
        "MULTIPLE_CANDIDATES_OMITTED"
    ]
    omission_scorecard = materialize_semantic_review_scorecard(
        [omission_truth_lock.parent],
        governance_root / "OMISSION_SCORECARD.json",
        blind_set_manifest=blind_set_path,
        selection_manifest=selection_path,
        pass2_manifest=pass2_path,
        global_review_lock=omission_global_lock,
        contract=contract,
    )
    assert json.loads(
        omission_scorecard.read_text(encoding="utf-8")
    )["strict_case_agreement_passed"] is False

    with (review_root / "review.json").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="review"):
        materialize_semantic_truth_overlay(
            authority_path=(
                case_root / "authority" / "authority.json"
            ),
            blind_set_manifest=blind_set_path,
            selection_manifest=selection_path,
            pass2_manifest=pass2_path,
            global_review_lock=global_lock,
            review_unit_id=unit["review_unit_id"],
            destination=truth_root / "after-tamper",
            contract=contract,
        )
