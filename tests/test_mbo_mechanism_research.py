from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.research_orchestration

from smc_trader.model import Direction, EventKind, EventOrigin, MarketEvent, Timeframe
import scripts.run_mbo_mechanism_research as phase6_runner

from smc_trader.mbo_mechanism_research import (
    EpisodeMatchContextUnavailable,
    EpisodeWindowUnavailable,
    EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY,
    EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY,
    LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    MechanismEpisode,
    PHASE6_FIXED_FAMILY,
    PRIMARY_METRIC_BY_HYPOTHESIS,
    Phase6ResearchError,
    REQUIRED_PHASE6_IDENTITY_BINDINGS,
    SECONDARY_METRICS_BY_HYPOTHESIS,
    STABILITY_POLICY,
    STRICT_PRIOR_CONTEXT_CENSOR_REASONS,
    STRICT_PRIOR_CONTEXT_UNAVAILABLE_FIELD_POLICY,
    SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    aggregate_episode_feature_window,
    canonical_identity,
    canonicalize_mechanism_episodes,
    evaluate_descriptive_sensitivity,
    evaluate_fixed_mechanism_family,
    first_active_displacement_episodes,
    first_fvg_lifecycle_episodes,
    load_frozen_phase6_contract,
    match_mechanism_controls,
    pack_nonoverlapping_mechanism_controls,
    strict_prior_match_context_clock,
    validate_minute_feature_frame,
    validate_phase6_design,
    validate_registered_synthetic_mbo_flow,
    validate_underpowered_extension_gate,
)
from scripts.run_mbo_mechanism_research import (
    ACTIVE_SYNTHETIC_CLOCK_SCOPE,
    PRIOR_WARMUP_SYNTHETIC_CLOCK_SCOPE,
    _bind_synthetic_exception_audit_scope,
    _candidate_episode,
    _blocked_synthetic_exception_lineage,
    _extension_warmup_synthetic_registration,
    _legacy_synthetic_exception_audit_projection,
    _scoped_displacement_monotonicity,
    _source_only_episode_m5_bars,
    _strict_prior_context_censor_counts,
    _strict_prior_match_context,
    _synthetic_semantic_exception_audit,
    _validate_reader_census,
    run as run_phase6,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/research/mbo_mechanism_phase6_template.yaml"


def _clock(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-06-03T13:30:00Z") + pd.Timedelta(minutes=minute)


def test_strict_prior_context_counts_exclude_synthetic_lineage_reasons() -> None:
    assert _strict_prior_context_censor_counts(
        [
            {
                "reason": (
                    "strict_prior_real_completed_m1_context_atr_unavailable"
                )
            },
            {
                "reason": (
                    "synthetic_semantic_exception_descendant_excluded_from_samples"
                )
            },
            {
                "reason": (
                    "strict_prior_real_completed_m1_context_atr_unavailable"
                )
            },
        ]
    ) == {
        "strict_prior_real_completed_m1_context_atr_unavailable": 2
    }


def _feature_frame(minutes: int = 40) -> pd.DataFrame:
    rows = []
    for minute in range(minutes):
        change_valid = minute > 0
        rows.append(
            {
                "decision_time": _clock(minute),
                "symbol": "NQM4",
                "instrument_id": 13743,
                "book_observed_at": _clock(minute) - pd.Timedelta(milliseconds=1),
                "book_valid": True,
                "book_change_valid": change_valid,
                "aggressor_buy_volume": float(10 + minute),
                "aggressor_sell_volume": 5.0,
                "aggressor_unknown_volume": 0.0,
                "aggressor_buy_trade_count": float(2 + minute),
                "aggressor_sell_trade_count": 1.0,
                "aggressor_unknown_trade_count": 0.0,
                "passive_bid_fill_volume": 2.0,
                "passive_ask_fill_volume": 1.0,
                "passive_unknown_fill_volume": 0.0,
                "passive_bid_fill_count": 1.0,
                "passive_ask_fill_count": 1.0,
                "passive_unknown_fill_count": 0.0,
                "displayed_bid_add_volume": 10.0,
                "displayed_ask_add_volume": 8.0,
                "displayed_bid_cancel_volume": 3.0,
                "displayed_ask_cancel_volume": 4.0,
                "best_level_ofi_contracts": float(minute) if change_valid else None,
                "mid_change_ticks": 1.0 if change_valid else None,
                "spread_ticks": 1.0,
                "book_valid_clock_fraction": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _episode(
    identity: str,
    minute: int,
    *,
    hypothesis: str = "displacement_impact",
    kind: str = "displacement_observed",
    control_kind: str | None = None,
    event_variant: str | None = None,
    half: str = "first_half",
    entity_id: str | None = None,
    direction: str = "long",
    score: float | None = None,
) -> MechanismEpisode:
    return MechanismEpisode(
        episode_id=identity,
        hypothesis=hypothesis,
        event_kind=kind,
        entity_id=identity if entity_id is None else entity_id,
        known_at=_clock(minute),
        event_time=_clock(minute - 4),
        source_bar_clocks=(_clock(minute),),
        source_m5_bar_event_ids=(f"bar:m5:{minute}",),
        symbol="NQM4",
        instrument_id=13743,
        timeframe="5m",
        direction=direction,
        session_phase="opening_expansion",
        half_week=half,
        match_fields={
            "study_week": "week_1",
            "half_week": half,
            "volatility_bucket": "atr_ticks_log2_5",
            "trend_relation": "with_m5_trend",
            "relative_volume_bucket": "rv_1_to_1_5",
        },
        control_kind=control_kind,
        event_variant=event_variant,
        score=score,
        match_context_clock=_clock(minute - 5),
    )


def test_template_is_valid_design_but_fails_closed_before_data_open() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    validate_phase6_design(payload)
    assert set(payload["identity_bindings"]) == REQUIRED_PHASE6_IDENTITY_BINDINGS
    assert payload["reader_census_contract"] == {
        "window_id": "2024-06-week-1",
        "completed_clocks": 6900,
        "real_completed": 6899,
        "synthetic_no_trade": 1,
        "synthetic_decision_clocks": ["2024-06-07T03:10:00Z"],
        "synthetic_event_policy": (
            "normalized_M1_clock_root_with_only_registered_terminal_censor_semantic_exception"
        ),
        "synthetic_m5_source_or_control_eligible": False,
        "mbo_response_window_eligible": True,
    }
    fvg = payload["hypothesis_design"]["fvg_retest_response"]
    assert fvg["primary_control_variant"] == "failed_retest"
    assert "excluded_from_holm_and_phase7" in fvg["pseudo_zone_role"]
    assert payload["matching_context"]["same_clock_or_post_event_fallback"] is False
    assert payload["matching_context"]["unavailable_field_policy"] == dict(
        STRICT_PRIOR_CONTEXT_UNAVAILABLE_FIELD_POLICY
    )
    assert payload["matching_context"]["censor_reasons"] == list(
        STRICT_PRIOR_CONTEXT_CENSOR_REASONS
    )
    assert payload["synthetic_semantic_exception_policy"] == (
        SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
    )
    assert payload["extension_warmup_synthetic_exception_policy"] == (
        EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY
    )
    assert payload["extension_displacement_monotonicity_policy"] == (
        EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY
    )
    assert payload["stability"] == STABILITY_POLICY
    assert "maximum_cardinality" not in payload["matching"]["algorithm"]

    with pytest.raises(Phase6ResearchError, match="incomplete/unfrozen"):
        load_frozen_phase6_contract(TEMPLATE, root=ROOT)


def test_template_rejects_fvg_pooling_and_post_event_match_context() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    pooled = json.loads(json.dumps(payload))
    pooled["hypothesis_design"]["fvg_retest_response"][
        "pseudo_zone_role"
    ] = "primary_control"
    with pytest.raises(Phase6ResearchError, match="roles must remain disjoint"):
        validate_phase6_design(pooled)

    post_event = json.loads(json.dumps(payload))
    post_event["matching_context"]["same_clock_or_post_event_fallback"] = True
    with pytest.raises(Phase6ResearchError, match="strict-prior"):
        validate_phase6_design(post_event)

    unavailable_policy = json.loads(json.dumps(payload))
    unavailable_policy["matching_context"]["unavailable_field_policy"][
        "relative_volume"
    ] = "fill_zero"
    with pytest.raises(Phase6ResearchError, match="strict-prior"):
        validate_phase6_design(unavailable_policy)

    reasons = json.loads(json.dumps(payload))
    reasons["matching_context"]["censor_reasons"].pop()
    with pytest.raises(Phase6ResearchError, match="strict-prior"):
        validate_phase6_design(reasons)

    synthetic_exception = json.loads(json.dumps(payload))
    synthetic_exception["synthetic_semantic_exception_policy"][
        "registered_allowlist"
    ][0]["lifecycle"] = "active"
    with pytest.raises(Phase6ResearchError, match="synthetic semantic"):
        validate_phase6_design(synthetic_exception)

    source_data_policy = json.loads(json.dumps(payload))
    source_data_policy["synthetic_semantic_exception_policy"][
        "source_data_policy"
    ]["terminal_source_data_ids"] = "unchecked"
    with pytest.raises(Phase6ResearchError, match="synthetic semantic"):
        validate_phase6_design(source_data_policy)

    extension = json.loads(json.dumps(payload))
    extension["study_mode"] = "primary_plus_registered_underpowered_extension"
    extension["reader_census_contract"]["window_id"] = "2024-06-week-2"
    extension["reader_census_contract"]["synthetic_decision_clocks"] = [
        "2024-06-10T04:14:00Z"
    ]
    extension["extension_warmup_synthetic_exception_policy"][
        "source"
    ] = "unbound_result"
    with pytest.raises(Phase6ResearchError, match="extension warmup"):
        validate_phase6_design(extension)

    pooled_stability = json.loads(json.dumps(payload))
    pooled_stability["stability"]["extension_half_week_strata"] = (
        "half_week_cross_week_pooling"
    )
    with pytest.raises(Phase6ResearchError, match="stability stratification"):
        validate_phase6_design(pooled_stability)


def _synthetic_semantic_gate_fixture() -> tuple[
    MarketEvent,
    MarketEvent,
    MarketEvent,
    dict[str, MarketEvent],
]:
    source_clock = _clock(15)
    synthetic_clock = _clock(20)
    source = MarketEvent(
        event_id="real-m5-source",
        kind=EventKind.BAR_COMPLETED,
        observed_at=source_clock,
        timeframe=Timeframe.M5,
        side=None,
        price=100.0,
        strength=0.0,
        event_time=source_clock,
        known_at=source_clock,
        evidence={
            "real_completed": True,
            "clock_only": False,
            "detector_candle_id": "real-m5-candle",
        },
        source_data_ids=("real-m5-data",),
        origin=EventOrigin.NORMALIZED_DATA,
    )
    clock_root = MarketEvent(
        event_id="synthetic-m1-clock-root",
        kind=EventKind.BAR_COMPLETED,
        observed_at=synthetic_clock,
        timeframe=Timeframe.M1,
        side=None,
        price=100.0,
        strength=0.0,
        event_time=synthetic_clock,
        known_at=synthetic_clock,
        evidence={
            "real_completed": False,
            "clock_only": True,
            "detector_candle_id": "synthetic-m1-candle",
        },
        source_data_ids=("synthetic-m1-data",),
        origin=EventOrigin.NORMALIZED_DATA,
    )
    terminal = MarketEvent(
        event_id="synthetic-displacement-terminal",
        kind=EventKind.DISPLACEMENT_OBSERVED,
        observed_at=synthetic_clock,
        timeframe=Timeframe.M5,
        side="above",
        price=None,
        strength=0.5,
        source_ids=(source.event_id,),
        event_time=_clock(10),
        known_at=synthetic_clock,
        direction=Direction.LONG,
        evidence={
            "lifecycle": "censored",
            "terminal_reason": "synthetic_interruption",
        },
        source_event_ids=(source.event_id,),
        source_data_ids=("real-m5-candle",),
        source_entity_ids=("displacement:1",),
        context_event_ids=(clock_root.event_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    return terminal, source, clock_root, {
        source.event_id: source,
        clock_root.event_id: clock_root,
        terminal.event_id: terminal,
    }


def test_exact_synthetic_terminal_exception_is_context_only_and_audited() -> None:
    terminal, source, clock_root, events = _synthetic_semantic_gate_fixture()
    audit = _synthetic_semantic_exception_audit(
        terminal,
        observation_clock=terminal.known_at,
        registered_synthetic_clocks=frozenset({terminal.known_at}),
        event_get=events.get,
    )
    assert audit["source_event_ids"] == (source.event_id,)
    assert audit["context_event_ids"] == (clock_root.event_id,)
    assert audit["last_real_source_known_at"] == source.known_at
    assert audit["terminal_detector_candle_ids_producer_order"] == (
        "real-m5-candle",
    )
    assert audit["canonical_real_M5_detector_candle_id_union"] == (
        "real-m5-candle",
    )
    assert audit["synthetic_context_event_ids"] == (clock_root.event_id,)
    assert audit["synthetic_context_clocks"] == (clock_root.known_at,)
    assert audit["synthetic_context_roots"] == (
        {
            "event_id": clock_root.event_id,
            "known_at": clock_root.known_at,
            "source_data_ids": ("synthetic-m1-data",),
            "detector_candle_id": "synthetic-m1-candle",
        },
    )
    assert audit["dag_node_count"] == 3
    assert audit["dag_edge_count"] == 2
    assert len(audit["dag_sha256"]) == 64
    assert audit["disposition"] == (
        "allowed_but_excluded_from_all_analysis_samples"
    )


def test_interior_synthetic_root_scopes_terminal_on_next_real_clock() -> None:
    terminal, source, clock_root, _ = _synthetic_semantic_gate_fixture()
    root_clock = terminal.known_at - pd.Timedelta(minutes=1)
    interior_root = MarketEvent(
        **{
            **clock_root.__dict__,
            "event_id": "interior-synthetic-m1-clock-root",
            "observed_at": root_clock,
            "event_time": root_clock,
            "known_at": root_clock,
        }
    )
    terminal = MarketEvent(
        **{
            **terminal.__dict__,
            "context_event_ids": (interior_root.event_id,),
        }
    )
    events = {
        source.event_id: source,
        interior_root.event_id: interior_root,
        terminal.event_id: terminal,
    }
    audit = _synthetic_semantic_exception_audit(
        terminal,
        observation_clock=terminal.known_at,
        registered_synthetic_clocks=frozenset({root_clock}),
        event_get=events.get,
    )
    scoped = _bind_synthetic_exception_audit_scope(
        audit,
        observation_clock=terminal.known_at,
        active_registered_clocks=frozenset({root_clock}),
        prior_registration={
            "registered_clocks": frozenset(),
            "expected_audits_by_event_id": {},
        },
    )

    assert audit["known_at"] == terminal.known_at
    assert audit["synthetic_context_clocks"] == (root_clock,)
    assert scoped["clock_scope"] == ACTIVE_SYNTHETIC_CLOCK_SCOPE


def test_synthetic_terminal_audit_requires_all_registered_constituent_roots() -> None:
    terminal, _, clock_root, events = _synthetic_semantic_gate_fixture()
    missing_clock = terminal.known_at - pd.Timedelta(minutes=1)

    with pytest.raises(
        Phase6ResearchError,
        match="exactly cover the registered synthetic constituent clocks",
    ):
        _synthetic_semantic_exception_audit(
            terminal,
            observation_clock=terminal.known_at,
            registered_synthetic_clocks=frozenset(
                {missing_clock, clock_root.known_at}
            ),
            event_get=events.get,
        )


def test_synthetic_terminal_audit_accepts_all_registered_constituent_roots() -> None:
    terminal, source, clock_root, _ = _synthetic_semantic_gate_fixture()
    interior_clock = terminal.known_at - pd.Timedelta(minutes=1)
    interior_root = MarketEvent(
        **{
            **clock_root.__dict__,
            "event_id": "interior-synthetic-m1-clock-root-2",
            "observed_at": interior_clock,
            "event_time": interior_clock,
            "known_at": interior_clock,
            "source_data_ids": ("interior-synthetic-m1-data",),
            "evidence": {
                **dict(clock_root.evidence),
                "detector_candle_id": "interior-synthetic-m1-candle",
            },
        }
    )
    terminal = MarketEvent(
        **{
            **terminal.__dict__,
            "context_event_ids": (
                interior_root.event_id,
                clock_root.event_id,
            ),
        }
    )
    events = {
        source.event_id: source,
        interior_root.event_id: interior_root,
        clock_root.event_id: clock_root,
        terminal.event_id: terminal,
    }

    audit = _synthetic_semantic_exception_audit(
        terminal,
        observation_clock=terminal.known_at,
        registered_synthetic_clocks=frozenset(
            {interior_clock, terminal.known_at}
        ),
        event_get=events.get,
    )

    assert audit["synthetic_context_clocks"] == (
        interior_clock,
        terminal.known_at,
    )
    assert audit["synthetic_context_root_count"] == 2


def test_synthetic_terminal_audit_rejects_duplicate_context_root_clock() -> None:
    terminal, source, clock_root, _ = _synthetic_semantic_gate_fixture()
    duplicate_root = MarketEvent(
        **{
            **clock_root.__dict__,
            "event_id": "duplicate-synthetic-m1-clock-root",
            "source_data_ids": ("duplicate-synthetic-m1-data",),
            "evidence": {
                **dict(clock_root.evidence),
                "detector_candle_id": "duplicate-synthetic-m1-candle",
            },
        }
    )
    ordered_ids = tuple(
        sorted((clock_root.event_id, duplicate_root.event_id))
    )
    terminal = MarketEvent(
        **{
            **terminal.__dict__,
            "context_event_ids": ordered_ids,
        }
    )
    events = {
        source.event_id: source,
        clock_root.event_id: clock_root,
        duplicate_root.event_id: duplicate_root,
        terminal.event_id: terminal,
    }

    with pytest.raises(
        Phase6ResearchError,
        match="exactly cover the registered synthetic constituent clocks",
    ):
        _synthetic_semantic_exception_audit(
            terminal,
            observation_clock=terminal.known_at,
            registered_synthetic_clocks=frozenset({terminal.known_at}),
            event_get=events.get,
        )


def _extension_contract_with_prior_synthetic_audit(
    audit: dict[str, object],
) -> SimpleNamespace:
    legacy_audit = _legacy_synthetic_exception_audit_projection(audit)
    prior = {
        "study_mode": "primary_week_only",
        "coverage": {
            "reader_active_census": {
                "synthetic_decision_clocks": [audit["known_at"]],
                "synthetic": 1,
            },
            "synthetic_semantic_exception_gate": {
                "policy": LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
                "allowed_count": 1,
                "rejected_count": 0,
                "allowed": [legacy_audit],
                "rejected": [],
                "blocked_lineage_event_count": 2,
                "blocked_lineage_event_ids": [
                    legacy_audit["event_id"],
                    legacy_audit["current_synthetic_context_event_id"],
                ],
            },
        },
    }
    prior["result_identity"] = canonical_identity(prior)
    return SimpleNamespace(
        prior_week1_result=prior,
        payload={
            "study_mode": "primary_plus_registered_underpowered_extension",
            "extension_warmup_synthetic_exception_policy": (
                EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY
            ),
        },
        primary_window=SimpleNamespace(
            start=pd.Timestamp("2024-06-02T22:00:00Z"),
            end_exclusive=pd.Timestamp("2024-06-07T21:01:00Z"),
        ),
    )


def test_extension_warmup_exception_is_exactly_hash_bound_and_scoped() -> None:
    terminal, _, clock_root, events = _synthetic_semantic_gate_fixture()
    audit = _synthetic_semantic_exception_audit(
        terminal,
        observation_clock=terminal.known_at,
        registered_synthetic_clocks=frozenset({terminal.known_at}),
        event_get=events.get,
    )
    contract = _extension_contract_with_prior_synthetic_audit(audit)
    registration = _extension_warmup_synthetic_registration(contract)

    assert registration["registered_clocks"] == frozenset({terminal.known_at})
    assert registration["blocked_lineage_event_ids"] == frozenset(
        {terminal.event_id, clock_root.event_id}
    )
    prior_scoped = _bind_synthetic_exception_audit_scope(
        audit,
        observation_clock=terminal.known_at,
        active_registered_clocks=frozenset(
            {pd.Timestamp("2024-06-10T04:14:00Z")}
        ),
        prior_registration=registration,
    )
    assert prior_scoped["clock_scope"] == PRIOR_WARMUP_SYNTHETIC_CLOCK_SCOPE

    active_clock = pd.Timestamp("2024-06-10T04:14:00Z")
    active_audit = {
        **audit,
        "known_at": active_clock,
        "synthetic_context_clocks": (active_clock,),
    }
    active_scoped = _bind_synthetic_exception_audit_scope(
        active_audit,
        observation_clock=active_clock,
        active_registered_clocks=frozenset(
            {pd.Timestamp("2024-06-10T04:14:00Z")}
        ),
        prior_registration=registration,
    )
    assert active_scoped["clock_scope"] == ACTIVE_SYNTHETIC_CLOCK_SCOPE


def test_extension_warmup_exception_rejects_unbound_or_changed_audit() -> None:
    terminal, _, clock_root, events = _synthetic_semantic_gate_fixture()
    audit = _synthetic_semantic_exception_audit(
        terminal,
        observation_clock=terminal.known_at,
        registered_synthetic_clocks=frozenset({terminal.known_at}),
        event_get=events.get,
    )
    registration = _extension_warmup_synthetic_registration(
        _extension_contract_with_prior_synthetic_audit(audit)
    )
    with pytest.raises(Phase6ResearchError, match="differs from the hash-bound"):
        _bind_synthetic_exception_audit_scope(
            {**audit, "dag_sha256": "0" * 64},
            observation_clock=terminal.known_at,
            active_registered_clocks=frozenset(),
            prior_registration=registration,
        )
    with pytest.raises(Phase6ResearchError, match="no registered clock scope"):
        _bind_synthetic_exception_audit_scope(
            {
                **audit,
                "known_at": pd.Timestamp("2024-06-08T00:00:00Z"),
                "synthetic_context_clocks": (
                    pd.Timestamp("2024-06-08T00:00:00Z"),
                ),
            },
            observation_clock=pd.Timestamp("2024-06-08T00:00:00Z"),
            active_registered_clocks=frozenset(),
            prior_registration=registration,
        )

    tampered_contract = _extension_contract_with_prior_synthetic_audit(audit)
    tampered_contract.prior_week1_result["coverage"][
        "synthetic_semantic_exception_gate"
    ]["blocked_lineage_event_ids"] = [terminal.event_id]
    tampered_contract.prior_week1_result["result_identity"] = canonical_identity(
        tampered_contract.prior_week1_result
    )
    with pytest.raises(Phase6ResearchError, match="blocked lineage"):
        _extension_warmup_synthetic_registration(tampered_contract)


def test_extension_displacement_monotonicity_is_never_false_combined() -> None:
    prior = {
        "displacement_continuous_monotonicity": {
            "n": 53,
            "spearman_rho": 0.025,
            "threshold_selected": False,
        }
    }
    scoped = _scoped_displacement_monotonicity(
        {"n": 41, "spearman_rho": 0.1, "threshold_selected": False},
        prior_week1_result=prior,
        registered_policy=EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY,
    )
    assert scoped["prior_week1"] == prior[
        "displacement_continuous_monotonicity"
    ]
    assert scoped["current_week2"]["n"] == 41
    assert scoped["combined"] is None
    assert scoped["holm_included"] is False
    assert scoped["phase7_evidence_admission"] is False

    with pytest.raises(Phase6ResearchError, match="policy is absent"):
        _scoped_displacement_monotonicity(
            {"n": 41, "spearman_rho": 0.1, "threshold_selected": False},
            prior_week1_result=prior,
            registered_policy="pool_both_weeks",
        )


def test_synthetic_terminal_preserves_producer_order_but_audits_canonical_set() -> None:
    terminal, source, clock_root, _ = _synthetic_semantic_gate_fixture()
    earlier = MarketEvent(
        **{
            **source.__dict__,
            "event_id": "real-m5-source-earlier",
            "observed_at": _clock(10),
            "event_time": _clock(10),
            "known_at": _clock(10),
            "evidence": {
                **dict(source.evidence),
                "detector_candle_id": "z-earlier-detector",
            },
            "source_data_ids": ("raw-earlier-digest",),
        }
    )
    later = MarketEvent(
        **{
            **source.__dict__,
            "event_id": "real-m5-source-later",
            "evidence": {
                **dict(source.evidence),
                "detector_candle_id": "a-later-detector",
            },
            "source_data_ids": ("raw-later-digest",),
        }
    )
    producer_order = ("z-earlier-detector", "a-later-detector")
    terminal = MarketEvent(
        **{
            **terminal.__dict__,
            "source_ids": (earlier.event_id, later.event_id),
            "source_event_ids": (earlier.event_id, later.event_id),
            "source_data_ids": producer_order,
        }
    )
    events = {
        earlier.event_id: earlier,
        later.event_id: later,
        clock_root.event_id: clock_root,
        terminal.event_id: terminal,
    }

    audit = _synthetic_semantic_exception_audit(
        terminal,
        observation_clock=terminal.known_at,
        registered_synthetic_clocks=frozenset({terminal.known_at}),
        event_get=events.get,
    )

    assert audit["terminal_detector_candle_ids_producer_order"] == producer_order
    assert audit["canonical_real_M5_detector_candle_id_union"] == (
        "a-later-detector",
        "z-earlier-detector",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("active", "outside the exact allowlist"),
        ("creation", "outside the exact allowlist"),
        ("synthetic_source", "non-real or non-M5 source BAR"),
        ("context_missing", "unique and context-only"),
        ("synthetic_raw_data_id", "identifier leaked"),
        ("synthetic_detector_id", "identifier leaked"),
        ("missing_real_detector_id", "do not exactly equal"),
        ("extra_real_detector_id", "do not exactly equal"),
    ),
)
def test_synthetic_semantic_gate_rejects_every_non_allowlisted_shape(
    mutation: str,
    message: str,
) -> None:
    terminal, source, clock_root, events = _synthetic_semantic_gate_fixture()
    if mutation == "active":
        terminal = MarketEvent(
            **{
                **terminal.__dict__,
                "evidence": {
                    "lifecycle": "active",
                    "terminal_reason": None,
                },
            }
        )
    elif mutation == "creation":
        terminal = MarketEvent(
            **{**terminal.__dict__, "kind": EventKind.FVG_CREATED}
        )
    elif mutation == "synthetic_source":
        source = MarketEvent(
            **{
                **source.__dict__,
                "evidence": {"real_completed": False, "clock_only": True},
            }
        )
        events[source.event_id] = source
    elif mutation == "context_missing":
        terminal = MarketEvent(
            **{**terminal.__dict__, "context_event_ids": (),}
        )
    elif mutation == "synthetic_raw_data_id":
        terminal = MarketEvent(
            **{
                **terminal.__dict__,
                "source_data_ids": (
                    "real-m5-candle",
                    "synthetic-m1-data",
                ),
            }
        )
    elif mutation == "synthetic_detector_id":
        terminal = MarketEvent(
            **{
                **terminal.__dict__,
                "source_data_ids": (
                    "real-m5-candle",
                    "synthetic-m1-candle",
                ),
            }
        )
    elif mutation == "missing_real_detector_id":
        terminal = MarketEvent(
            **{**terminal.__dict__, "source_data_ids": (),}
        )
    elif mutation == "extra_real_detector_id":
        terminal = MarketEvent(
            **{
                **terminal.__dict__,
                "source_data_ids": (
                    "real-m5-candle",
                    "extra-real-candle",
                ),
            }
        )
    events[terminal.event_id] = terminal

    with pytest.raises(Phase6ResearchError, match=message):
        _synthetic_semantic_exception_audit(
            terminal,
            observation_clock=terminal.known_at,
            registered_synthetic_clocks=frozenset({terminal.known_at}),
            event_get=events.get,
        )


def test_market_event_rejects_duplicate_terminal_detector_identities() -> None:
    terminal, _, _, _ = _synthetic_semantic_gate_fixture()
    with pytest.raises(
        ValueError,
        match="source_data_ids identities must be unique",
    ):
        MarketEvent(
            **{
                **terminal.__dict__,
                "source_data_ids": (
                    "real-m5-candle",
                    "real-m5-candle",
                ),
            }
        )


def test_raw_and_fvg_descendants_of_synthetic_terminal_are_blocked() -> None:
    terminal, source, clock_root, events = _synthetic_semantic_gate_fixture()
    raw = MarketEvent(
        event_id="descendant-raw-break",
        kind=EventKind.RAW_BOUNDARY_BREAK,
        observed_at=_clock(25),
        timeframe=Timeframe.M5,
        side="above",
        price=101.0,
        strength=0.5,
        source_ids=(source.event_id,),
        event_time=_clock(25),
        known_at=_clock(25),
        direction=Direction.LONG,
        evidence={"boundary": 101.0},
        source_event_ids=(source.event_id,),
        context_event_ids=(terminal.event_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    fvg = MarketEvent(
        event_id="descendant-fvg-created",
        kind=EventKind.FVG_CREATED,
        observed_at=_clock(30),
        timeframe=Timeframe.M5,
        side="below",
        price=100.0,
        strength=0.5,
        source_ids=(raw.event_id,),
        event_time=_clock(30),
        known_at=_clock(30),
        direction=Direction.LONG,
        evidence={"fvg_id": "fvg:descendant"},
        source_event_ids=(raw.event_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    events.update(
        {
            raw.event_id: raw,
            fvg.event_id: fvg,
        }
    )
    blocked = frozenset({terminal.event_id, clock_root.event_id})

    assert _blocked_synthetic_exception_lineage(
        raw,
        event_get=events.get,
        blocked_event_ids=blocked,
    ) == tuple(sorted(blocked))
    assert _blocked_synthetic_exception_lineage(
        fvg,
        event_get=events.get,
        blocked_event_ids=blocked,
    ) == tuple(sorted(blocked))


@pytest.mark.parametrize(
    ("label", "origin", "kind"),
    (
        ("legacy_structure", EventOrigin.LEGACY_TRANSPORT, EventKind.STRUCTURE_STATE),
        ("legacy_fvg", EventOrigin.LEGACY_TRANSPORT, EventKind.FVG_STATE),
        ("projection", EventOrigin.STATE_PROJECTION, EventKind.STRUCTURE_STATE),
        ("normalized", EventOrigin.NORMALIZED_DATA, EventKind.BAR_COMPLETED),
    ),
)
def test_descendant_scan_treats_non_atomic_namespaces_as_terminal(
    label: str,
    origin: EventOrigin,
    kind: EventKind,
) -> None:
    clock = _clock(25)
    leaf = MarketEvent(
        event_id=f"{label}-leaf",
        kind=kind,
        observed_at=clock,
        timeframe=Timeframe.M5,
        side=None,
        price=100.0,
        strength=0.0,
        event_time=clock,
        known_at=clock,
        source_ids=(f"opaque-{label}-token",),
        source_event_ids=(f"opaque-{label}-token",),
        origin=origin,
    )
    raw = MarketEvent(
        event_id=f"raw-with-{label}-context",
        kind=EventKind.RAW_BOUNDARY_BREAK,
        observed_at=clock,
        timeframe=Timeframe.M5,
        side="above",
        price=101.0,
        strength=0.5,
        event_time=clock,
        known_at=clock,
        direction=Direction.LONG,
        context_event_ids=(leaf.event_id,),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )
    events = {leaf.event_id: leaf, raw.event_id: raw}

    assert _blocked_synthetic_exception_lineage(
        raw,
        event_get=events.get,
        blocked_event_ids={"unrelated-synthetic-terminal"},
    ) == ()
    assert _blocked_synthetic_exception_lineage(
        leaf,
        event_get=events.get,
        blocked_event_ids={leaf.event_id},
    ) == (leaf.event_id,)


def test_descendant_scan_fails_closed_on_missing_atomic_parent() -> None:
    clock = _clock(25)
    event = MarketEvent(
        event_id="atomic-with-missing-parent",
        kind=EventKind.FVG_CREATED,
        observed_at=clock,
        timeframe=Timeframe.M5,
        side="above",
        price=101.0,
        strength=0.5,
        event_time=clock,
        known_at=clock,
        direction=Direction.LONG,
        source_event_ids=("missing-canonical-parent",),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )

    with pytest.raises(Phase6ResearchError, match="missing source/context parent"):
        _blocked_synthetic_exception_lineage(
            event,
            event_get={event.event_id: event}.get,
            blocked_event_ids={"unrelated-synthetic-terminal"},
        )


def test_episode_window_separates_known_at_and_retrospective_features() -> None:
    frame = _feature_frame()
    episode = _episode("disp:1", 15)

    window = aggregate_episode_feature_window(frame, episode)

    assert window.formation_clocks == tuple(_clock(value) for value in range(11, 16))
    assert window.pre_clocks == tuple(_clock(value) for value in range(6, 11))
    assert window.post_clocks == tuple(_clock(value) for value in range(16, 21))
    assert window.formation_available_at == episode.known_at
    assert window.post_available_at > episode.known_at
    assert window.post_retrospective_only is True
    assert "post_displayed_defense_net_add_per_contract" in window.metrics
    assert not any("replenishment" in name for name in window.metrics)


def test_invalid_book_change_censors_episode_instead_of_zero_fill() -> None:
    frame = _feature_frame()
    frame.loc[12, "book_change_valid"] = False
    frame.loc[12, ["best_level_ofi_contracts", "mid_change_ticks"]] = None

    with pytest.raises(EpisodeWindowUnavailable) as captured:
        aggregate_episode_feature_window(frame, _episode("disp:2", 15))

    assert captured.value.reason == "book_change_invalid_in_required_window"


def test_feature_contract_rejects_future_book_duplicate_and_bad_delta() -> None:
    frame = _feature_frame(20)
    validated = validate_minute_feature_frame(
        frame,
        start=_clock(0),
        end_exclusive=_clock(20),
        symbol="NQM4",
        instrument_id=13743,
        expected_rows=20,
    )
    assert len(validated) == 20

    future = frame.copy()
    future.loc[2, "book_observed_at"] = _clock(2) + pd.Timedelta(seconds=1)
    with pytest.raises(Phase6ResearchError, match="future clock"):
        validate_minute_feature_frame(
            future,
            start=_clock(0),
            end_exclusive=_clock(20),
            symbol="NQM4",
            instrument_id=13743,
            expected_rows=20,
        )

    duplicate = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    with pytest.raises(Phase6ResearchError, match="duplicate"):
        validate_minute_feature_frame(
            duplicate,
            start=_clock(0),
            end_exclusive=_clock(20),
            symbol="NQM4",
            instrument_id=13743,
            expected_rows=21,
        )

    invalid_delta = frame.copy()
    invalid_delta.loc[0, "best_level_ofi_contracts"] = 0.0
    with pytest.raises(Phase6ResearchError, match="must remain null"):
        validate_minute_feature_frame(
            invalid_delta,
            start=_clock(0),
            end_exclusive=_clock(20),
            symbol="NQM4",
            instrument_id=13743,
            expected_rows=20,
        )


def test_source_clock_and_fvg_alias_leakage_fail_closed() -> None:
    with pytest.raises(Phase6ResearchError, match="source BAR"):
        MechanismEpisode(
            **{
                **_episode("disp:3", 15).__dict__,
                "source_bar_clocks": (_clock(16),),
            }
        )
    with pytest.raises(Phase6ResearchError, match="FVG_TOUCHED"):
        _episode(
            "fvg:alias",
            15,
            hypothesis="fvg_retest_response",
            kind="fvg_touched",
            event_variant="successful_retest",
        )


def test_fvg_first_lifecycle_keeps_variant_but_never_pools_entities() -> None:
    successful = _episode(
        "fvg:partial",
        15,
        hypothesis="fvg_retest_response",
        kind="fvg_partially_filled",
        event_variant="successful_retest",
    )
    later = MechanismEpisode(
        **{
            **successful.__dict__,
            "episode_id": "fvg:full",
            "known_at": _clock(20),
            "event_time": _clock(20),
            "source_bar_clocks": (_clock(20),),
            "source_m5_bar_event_ids": ("bar:m5:20",),
            "match_context_clock": _clock(15),
            "event_kind": "fvg_fully_filled",
        }
    )
    failed = _episode(
        "fvg:failed",
        25,
        hypothesis="fvg_retest_response",
        kind="fvg_invalidated",
        event_variant="failed_retest",
    )
    failed = MechanismEpisode(**{**failed.__dict__, "entity_id": "another_fvg"})

    selected = first_fvg_lifecycle_episodes([later, failed, successful])

    assert [item.episode_id for item in selected] == ["fvg:partial", "fvg:failed"]
    assert {item.event_variant for item in selected} == {
        "successful_retest",
        "failed_retest",
    }


def test_fvg_conflicting_lifecycle_at_earliest_known_at_fails_closed() -> None:
    successful = _episode(
        "fvg:partial",
        15,
        hypothesis="fvg_retest_response",
        kind="fvg_partially_filled",
        entity_id="fvg:same",
        event_variant="successful_retest",
    )
    failed = _episode(
        "fvg:invalidated",
        15,
        hypothesis="fvg_retest_response",
        kind="fvg_invalidated",
        entity_id="fvg:same",
        event_variant="failed_retest",
    )

    for episodes in ((successful, failed), (failed, successful)):
        with pytest.raises(Phase6ResearchError, match="conflicting lifecycle"):
            first_fvg_lifecycle_episodes(episodes)


def test_fvg_same_lifecycle_signature_with_conflicting_payload_fails_closed() -> None:
    first = _episode(
        "fvg:partial-a",
        15,
        hypothesis="fvg_retest_response",
        kind="fvg_partially_filled",
        entity_id="fvg:same-payload-clock",
        event_variant="successful_retest",
        direction="long",
    )
    conflicting = _episode(
        "fvg:partial-b",
        15,
        hypothesis="fvg_retest_response",
        kind="fvg_partially_filled",
        entity_id="fvg:same-payload-clock",
        event_variant="successful_retest",
        direction="short",
    )

    with pytest.raises(Phase6ResearchError, match="conflicting duplicate"):
        first_fvg_lifecycle_episodes((first, conflicting))


def test_fvg_first_lifecycle_is_invariant_to_later_lifecycle_changes() -> None:
    first = _episode(
        "fvg:first-partial",
        15,
        hypothesis="fvg_retest_response",
        kind="fvg_partially_filled",
        entity_id="fvg:frozen",
        event_variant="successful_retest",
    )
    later_success = _episode(
        "fvg:later-full",
        20,
        hypothesis="fvg_retest_response",
        kind="fvg_fully_filled",
        entity_id="fvg:frozen",
        event_variant="successful_retest",
    )
    later_failure = _episode(
        "fvg:later-invalidated",
        20,
        hypothesis="fvg_retest_response",
        kind="fvg_invalidated",
        entity_id="fvg:frozen",
        event_variant="failed_retest",
    )

    frozen = first_fvg_lifecycle_episodes((first,))

    assert first_fvg_lifecycle_episodes((later_success, first)) == frozen
    assert first_fvg_lifecycle_episodes((later_failure, first)) == frozen
    assert frozen[0].episode_id == "fvg:first-partial"
    assert frozen[0].event_variant == "successful_retest"


def test_matching_requires_same_week_half_session_and_future_nonoverlap() -> None:
    treatment = _episode("treatment", 10)
    good = _episode(
        "control:good",
        25,
        control_kind="non_active_displacement_clock",
        kind="completed_m5_clock",
    )
    wrong_half = _episode(
        "control:wrong-half",
        26,
        control_kind="non_active_displacement_clock",
        kind="completed_m5_clock",
        half="second_half",
    )
    index = {_clock(value): value for value in range(40)}
    fields = (
        "symbol",
        "instrument_id",
        "study_week",
        "half_week",
        "timeframe",
        "session_phase",
        "direction",
        "volatility_bucket",
        "trend_relation",
        "relative_volume_bucket",
    )

    result = match_mechanism_controls(
        [treatment],
        [wrong_half, good],
        completed_index=index,
        exact_fields=fields,
        maximum_completed_minute_offset=30,
    )

    assert result.matched == 1
    assert result.pairs[0].candidate_id == "control:good"
    assert result.pairs[0].completed_bar_offset >= 11


def _paired_rows(count: int) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for hypothesis in PHASE6_FIXED_FAMILY:
        primary = PRIMARY_METRIC_BY_HYPOTHESIS[hypothesis]
        metrics = (primary, *SECONDARY_METRICS_BY_HYPOTHESIS[hypothesis])
        rows = []
        for index in range(count):
            treatment = {name: 2.0 for name in metrics}
            control = {name: 0.0 for name in metrics}
            half = "first_half" if index < count // 2 else "second_half"
            rows.append(
                {
                    "treatment_metrics": treatment,
                    "control_metrics": control,
                    "treatment_study_week": "week_1",
                    "control_study_week": "week_1",
                    "treatment_half_week": half,
                    "control_half_week": half,
                    "treatment_session_phase": "morning_delivery",
                    "control_session_phase": "morning_delivery",
                }
            )
        result[hypothesis] = rows
    return result


def test_fixed_holm_family_separates_support_engineering_and_extension() -> None:
    rows = _paired_rows(30)
    evaluated = evaluate_fixed_mechanism_family(
        rows,
        bootstrap_replicates=1000,
    )

    assert evaluated["engineering_status"] == "pass"
    assert evaluated["extension_required"] is False
    assert evaluated["phase7_evidence_allowlist"] == list(PHASE6_FIXED_FAMILY)
    assert all(
        evaluated["mechanisms"][name]["status"] == "supported"
        for name in PHASE6_FIXED_FAMILY
    )

    rows["mss_flow_shift"] = rows["mss_flow_shift"][:29]
    underpowered = evaluate_fixed_mechanism_family(
        rows,
        bootstrap_replicates=1000,
    )
    assert underpowered["engineering_status"] == "pass"
    assert underpowered["extension_required"] is True
    assert underpowered["mechanisms"]["mss_flow_shift"]["status"] == "underpowered"
    assert underpowered["holm"]["raw_p_values"]["mss_flow_shift"] == 1.0
    assert "mss_flow_shift" not in underpowered["phase7_evidence_allowlist"]


def test_stability_stratum_labels_must_match_on_both_sides() -> None:
    rows = _paired_rows(30)
    rows["sweep_rejection"][0]["control_half_week"] = "second_half"

    with pytest.raises(Phase6ResearchError, match="paired stratum mismatch"):
        evaluate_fixed_mechanism_family(rows, bootstrap_replicates=1000)

    rows = _paired_rows(30)
    rows["sweep_rejection"][0]["control_study_week"] = "week_2"
    with pytest.raises(Phase6ResearchError, match="paired stratum mismatch"):
        evaluate_fixed_mechanism_family(rows, bootstrap_replicates=1000)


def test_extension_stability_never_pools_same_half_labels_across_weeks() -> None:
    rows = _paired_rows(30)
    target = rows["sweep_rejection"]
    for index, item in enumerate(target):
        if index >= 20:
            item["treatment_study_week"] = "week_2"
            item["control_study_week"] = "week_2"
            item["treatment_half_week"] = (
                "first_half" if index < 25 else "second_half"
            )
            item["control_half_week"] = item["treatment_half_week"]
            for metric in item["treatment_metrics"]:
                item["treatment_metrics"][metric] = -1.0

    evaluated = evaluate_fixed_mechanism_family(
        rows,
        bootstrap_replicates=1000,
    )
    sweep = evaluated["mechanisms"]["sweep_rejection"]
    assert set(sweep["strata"]["study_week_x_half_week"]) == {
        "week_1:first_half",
        "week_1:second_half",
        "week_2:first_half",
        "week_2:second_half",
    }
    assert sweep["systematic_sign_reversal"] is True
    assert sweep["status"] == "unsupported"


def test_extension_gate_uses_sample_only_not_effect_or_p_value() -> None:
    mechanisms = {
        name: {
            "primary_matched_n": 30,
            "status": "unsupported",
            "holm_adjusted_p_value": 1.0,
        }
        for name in PHASE6_FIXED_FAMILY
    }
    result = {
        "manifest_sha256": "a" * 64,
        "study_mode": "primary_week_only",
        "raw_partition_hashes_verified": True,
        "mechanisms": mechanisms,
    }
    with pytest.raises(Phase6ResearchError, match="underpowered"):
        validate_underpowered_extension_gate(
            result,
            expected_manifest_sha256="a" * 64,
        )

    mechanisms["fvg_retest_response"]["primary_matched_n"] = 29
    validate_underpowered_extension_gate(
        result,
        expected_manifest_sha256="a" * 64,
    )

    with pytest.raises(Phase6ResearchError, match="raw partition hashes"):
        validate_underpowered_extension_gate(
            {**result, "raw_partition_hashes_verified": False},
            expected_manifest_sha256="a" * 64,
        )


def test_runner_cannot_skip_raw_rehash_and_cli_has_no_bypass(tmp_path: Path) -> None:
    with pytest.raises(Phase6ResearchError, match="requires raw partition hash"):
        run_phase6(
            manifest_path=tmp_path / "does-not-exist.yaml",
            output=tmp_path / "unused.json",
            verify_raw_partition_hashes=False,
        )
    runner_source = (
        ROOT / "scripts/run_mbo_mechanism_research.py"
    ).read_text(encoding="utf-8")
    assert "--skip-raw-partition-rehash" not in runner_source
    assert '"raw_partition_hashes_verified": True' in runner_source
    assert "non_authoritative_derived_display" in runner_source


def test_strict_prior_context_is_formation_prior_and_never_same_clock() -> None:
    source_clock = _clock(20)
    expected_clock = _clock(15)
    context_by_clock = {
        expected_clock: {
            "atr": 8.0,
            "relative_volume": 1.2,
            "trend_direction": "long",
            "source": "cached_real_completed_M1_at_strict_prior_formation_clock",
        },
        # A same-clock shock must have no influence on matching covariates.
        source_clock: {
            "atr": 10_000.0,
            "relative_volume": 100.0,
            "trend_direction": "short",
            "source": "cached_real_completed_M1_at_strict_prior_formation_clock",
        },
    }
    context_clock, context = _strict_prior_match_context(
        (source_clock,),
        context_by_clock,
        episode_id="candidate",
    )
    assert context_clock == expected_clock
    assert strict_prior_match_context_clock((source_clock,)) == expected_clock
    base = {
        "known_at": source_clock,
        "symbol": "NQM4",
        "instrument_id": 13743,
        "session_phase": "opening_expansion",
        "study_week": "week_1",
        "half_week": "first_half",
        "atr": context["atr"],
        "relative_volume": context["relative_volume"],
        "trend_direction": context["trend_direction"],
        "tick_size": 0.25,
        "match_context_clock": context_clock,
        "match_context_source": context["source"],
        "source_m5_bar_event_id": "bar:m5:20",
    }
    candidate = _candidate_episode(
        base,
        hypothesis="displacement_impact",
        direction="long",
        candidate_id="control:20",
        event_kind="completed_m5_clock",
        control_kind="non_active_displacement_clock",
    )
    assert candidate.match_context_clock == expected_clock
    assert candidate.match_fields["trend_relation"] == "with_m5_trend"
    assert candidate.match_fields["relative_volume_bucket"] == "rv_1_to_1_5"
    assert candidate.match_fields["volatility_bucket"] != "atr_ticks_log2_15"

    with pytest.raises(EpisodeMatchContextUnavailable) as captured:
        _strict_prior_match_context(
            (source_clock,),
            {source_clock: context_by_clock[source_clock]},
            episode_id="missing-prior",
        )
    assert captured.value.reason == (
        "strict_prior_real_completed_m1_context_missing"
    )


def test_cold_start_nullable_context_is_censored_without_zero_fill() -> None:
    source_clock = _clock(20)
    with pytest.raises(EpisodeMatchContextUnavailable) as captured:
        _strict_prior_match_context(
            (source_clock,),
            {},
            episode_id="cold-start-relative-volume-none",
        )
    assert captured.value.reason == (
        "strict_prior_real_completed_m1_context_missing"
    )

    with pytest.raises(EpisodeMatchContextUnavailable) as captured:
        _strict_prior_match_context(
            (source_clock,),
            {
                _clock(15): {
                    "atr": 8.0,
                    "relative_volume": None,
                    "trend_direction": None,
                }
            },
            episode_id="cold-start-relative-volume-none",
        )
    assert captured.value.reason == (
        "strict_prior_real_completed_m1_context_relative_volume_unavailable"
    )

    with pytest.raises(EpisodeMatchContextUnavailable) as captured:
        _strict_prior_match_context(
            (source_clock,),
            {
                _clock(15): {
                    "atr": None,
                    "relative_volume": 1.0,
                    "trend_direction": None,
                }
            },
            episode_id="cold-start-atr-none",
        )
    assert captured.value.reason == (
        "strict_prior_real_completed_m1_context_atr_unavailable"
    )

    context_clock, complete = _strict_prior_match_context(
        (source_clock,),
        {
            _clock(15): {
                "atr": 8.0,
                "relative_volume": 0.0,
                "trend_direction": None,
            }
        },
        episode_id="valid-zero-relative-volume",
    )
    assert context_clock == _clock(15)
    assert complete == {
        "atr": 8.0,
        "relative_volume": 0.0,
        "trend_direction": None,
        "source": "cached_real_completed_M1_at_strict_prior_formation_clock",
    }


def test_collect_eye_inputs_accepts_cold_start_relative_volume_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold_clock = _clock(0)
    m5_clock = _clock(5)
    completed_m1 = SimpleNamespace(
        real_completed=True,
        symbol="NQM4",
        instrument_id=13743,
    )
    m5_candle = SimpleNamespace(
        end=m5_clock,
        real_completed=True,
        symbol="NQM4",
        instrument_id=13743,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
    )
    normalized_m5 = SimpleNamespace(
        event_id="bar:m5:cold-start-test",
        origin=EventOrigin.NORMALIZED_DATA,
        kind=EventKind.BAR_COMPLETED,
        timeframe=Timeframe.M5,
        known_at=m5_clock,
        evidence={"real_completed": True, "clock_only": False},
    )
    updates = (
        SimpleNamespace(
            completed_1m=completed_m1,
            newly_completed={},
            anomalies=(),
        ),
        SimpleNamespace(
            completed_1m=completed_m1,
            newly_completed={Timeframe.M5: (m5_candle,)},
            anomalies=(),
        ),
    )

    def snapshot(relative_volume: float | None) -> SimpleNamespace:
        return SimpleNamespace(
            authority=phase6_runner.MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER,
            timeframe_states={
                Timeframe.M5: SimpleNamespace(
                    structure=SimpleNamespace(internal_direction=None)
                )
            },
            session=SimpleNamespace(
                relative_volume=relative_volume,
                phase="overnight_delivery",
            ),
        )

    observations = (
        SimpleNamespace(
            market_snapshot=snapshot(None),
            asof=cold_clock,
            semantic_events_this_update=(),
            frame=lambda timeframe: SimpleNamespace(metrics={"atr": 1.0}),
        ),
        SimpleNamespace(
            market_snapshot=snapshot(1.0),
            asof=m5_clock,
            semantic_events_this_update=(normalized_m5,),
            frame=lambda timeframe: SimpleNamespace(metrics={"atr": 1.0}),
        ),
    )
    reader = SimpleNamespace(on_bar=lambda bar: updates[bar])
    observer = SimpleNamespace(
        observe=lambda value: observations[updates.index(value)],
        audit_store=SimpleNamespace(
            get=lambda event_id: (
                normalized_m5
                if event_id == normalized_m5.event_id
                else None
            )
        ),
    )
    monkeypatch.setattr(
        phase6_runner,
        "_build_eye",
        lambda model_path: (reader, observer),
    )
    monkeypatch.setattr(
        phase6_runner,
        "iter_completed_bars",
        lambda frame, allow_data_gap_reset: iter((0, 1)),
    )
    monkeypatch.setattr(
        phase6_runner,
        "_validate_reader_census",
        lambda **kwargs: None,
    )

    result = phase6_runner._collect_eye_inputs(
        loaded=SimpleNamespace(frame=object()),
        feature_clocks=frozenset({m5_clock}),
        contract=SimpleNamespace(
            model_path=Path("unused"),
            payload={
                "reader_census_contract": {"synthetic_decision_clocks": []},
                "synthetic_semantic_exception_policy": (
                    SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
                ),
            },
            active_window=SimpleNamespace(window_id="cold-start-test"),
            symbol="NQM4",
            instrument_id=13743,
        ),
        tick_size=0.25,
    )

    assert result["emitted_bars"] == 2
    assert result["context_exclusions"] == [
        {
            "hypothesis": "shared_m5_control_base",
            "applies_to_hypotheses": (
                "displacement_impact",
                "fvg_retest_response",
            ),
            "role": "shared_m5_candidate_control_base",
            "episode_id": f"completed_m5:{m5_clock.isoformat()}",
            "context_clock": cold_clock,
            "context_source": (
                "cached_real_completed_M1_at_strict_prior_formation_clock"
            ),
            "reason": (
                "strict_prior_real_completed_m1_context_relative_volume_unavailable"
            ),
        }
    ]


def test_collection_failure_leaves_no_partial_result_or_ledgers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = _clock(0)
    active_window = SimpleNamespace(
        start=clock,
        end_exclusive=clock + pd.Timedelta(minutes=1),
        expected_rows=1,
    )
    contract = SimpleNamespace(
        payload={
            "reader_census_contract": {
                "synthetic_decision_clocks": [clock.isoformat()]
            }
        },
        active_window=active_window,
        symbol="NQM4",
        instrument_id=13743,
        warmup_start=clock - pd.Timedelta(minutes=1),
        ohlcv_path=tmp_path / "ohlcv.parquet",
        model_path=tmp_path / "model.json",
    )
    features = pd.DataFrame({"decision_time": [clock]})
    loaded = SimpleNamespace(
        warnings=(),
        contract_selection_causal=True,
        frame=pd.DataFrame(
            columns=("symbol", "instrument_id"),
            index=pd.DatetimeIndex([], tz="UTC"),
        ),
    )
    monkeypatch.setattr(phase6_runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        phase6_runner,
        "load_frozen_phase6_contract",
        lambda *args, **kwargs: contract,
    )
    monkeypatch.setattr(
        phase6_runner,
        "_load_feature_artifact",
        lambda value: (features, "feature-protocol"),
    )
    monkeypatch.setattr(
        phase6_runner,
        "validate_minute_feature_frame",
        lambda *args, **kwargs: features,
    )
    monkeypatch.setattr(
        phase6_runner,
        "validate_registered_synthetic_mbo_flow",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(phase6_runner, "load_ohlcv", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(phase6_runner, "_json", lambda path: {"tick_size": 0.25})
    monkeypatch.setattr(
        phase6_runner,
        "_collect_eye_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            Phase6ResearchError("cold-start context collection failed")
        ),
    )
    output = tmp_path / "outputs/research/cold_start.json"

    with pytest.raises(Phase6ResearchError, match="cold-start context"):
        phase6_runner.run(
            manifest_path=tmp_path / "manifest.yaml",
            output=output,
        )

    expected_paths = {
        output,
        output.with_suffix(".md"),
        output.with_name(f"{output.stem}.episodes.jsonl"),
        output.with_name(f"{output.stem}.matched_pairs.jsonl"),
        output.with_name(f"{output.stem}.unmatched.jsonl"),
    }
    assert not any(path.exists() for path in expected_paths)


def test_displacement_entity_first_then_clock_canonicalization() -> None:
    first = _episode("disp:a:first", 15, entity_id="disp:a", score=0.4)
    later = _episode("disp:a:later", 20, entity_id="disp:a", score=0.9)
    other = _episode("disp:b:first", 20, entity_id="disp:b", score=0.7)

    entity_first = first_active_displacement_episodes([later, other, first])
    assert [item.episode_id for item in entity_first] == [
        "disp:a:first",
        "disp:b:first",
    ]
    canonical, exclusions = canonicalize_mechanism_episodes(
        entity_first,
        analysis_hypothesis="displacement_impact",
        control_kind=None,
    )
    assert exclusions == ()
    assert len(canonical) == 2
    assert canonical[1].constituent_event_ids == ("disp:b:first",)


def test_clock_canonicalization_collapses_levels_and_excludes_ambiguity() -> None:
    first = _episode("sweep:a", 15, hypothesis="sweep_rejection", kind="sweep_confirmed")
    second = _episode("sweep:b", 15, hypothesis="sweep_rejection", kind="sweep_confirmed")
    later = _episode("sweep:c", 20, hypothesis="sweep_rejection", kind="sweep_confirmed")
    canonical, exclusions = canonicalize_mechanism_episodes(
        [second, later, first],
        analysis_hypothesis="sweep_rejection",
        control_kind=None,
    )
    assert exclusions == ()
    assert len(canonical) == 2
    assert canonical[0].constituent_event_ids == ("sweep:a", "sweep:b")
    assert canonical[0].constituent_entity_ids == ("sweep:a", "sweep:b")

    opposite = _episode(
        "sweep:opposite",
        15,
        hypothesis="sweep_rejection",
        kind="sweep_confirmed",
        direction="short",
    )
    ambiguous, exclusions = canonicalize_mechanism_episodes(
        [first, opposite, later],
        analysis_hypothesis="sweep_rejection",
        control_kind=None,
    )
    assert [item.constituent_event_ids for item in ambiguous] == [("sweep:c",)]
    assert exclusions[0]["reason"] == "ambiguous_opposite_direction_same_clock"


def test_fvg_pseudo_sensitivity_can_never_enter_holm_or_phase7() -> None:
    primary = PRIMARY_METRIC_BY_HYPOTHESIS["fvg_retest_response"]
    rows = [
        {
            "treatment_metrics": {primary: 1.0},
            "control_metrics": {primary: 0.0},
        }
        for _ in range(30)
    ]
    value = evaluate_descriptive_sensitivity(
        rows,
        hypothesis="fvg_retest_response",
        comparison="fvg_successful_vs_pseudo_zone_descriptive_sensitivity",
        bootstrap_replicates=1000,
    )
    assert value["holm_included"] is False
    assert value["phase7_evidence_admission"] is False
    assert value["support_verdict"] == "not_applicable_descriptive_only"
    assert "exact_sign_p_value" not in value


def test_hypothesis_local_packing_never_reuses_formation_or_post_clocks() -> None:
    frame = _feature_frame(60)
    first_treatment = _episode("treatment:1", 10)
    overlapping_treatment = _episode("treatment:2", 12)
    first_control = _episode(
        "control:1",
        25,
        control_kind="non_active_displacement_clock",
        kind="completed_m5_clock",
    )
    second_control = _episode(
        "control:2",
        40,
        control_kind="non_active_displacement_clock",
        kind="completed_m5_clock",
    )
    episodes = (
        first_treatment,
        overlapping_treatment,
        first_control,
        second_control,
    )
    windows = {
        item.episode_id: aggregate_episode_feature_window(frame, item)
        for item in episodes
    }
    result = pack_nonoverlapping_mechanism_controls(
        [overlapping_treatment, first_treatment],
        [second_control, first_control],
        windows=windows,
        completed_index={_clock(index): index for index in range(60)},
        exact_fields=(
            "symbol",
            "instrument_id",
            "study_week",
            "half_week",
            "timeframe",
            "session_phase",
            "direction",
            "volatility_bucket",
            "trend_relation",
            "relative_volume_bucket",
        ),
        maximum_completed_minute_offset=3000,
        embargo_minutes=5,
    )
    assert result.matched == 1
    assert result.pairs[0].treatment_id == "treatment:1"
    assert result.pairs[0].candidate_id == "control:1"
    assert result.overlap_exclusions == 1
    assert result.unmatched["treatment:2"] == (
        "global_inference_clock_overlap_or_candidate_capacity"
    )


@pytest.mark.parametrize("event_id", ["mss:interval", "fvg:interval"])
def test_source_only_m5_ancestry_is_clipped_to_closed_episode_interval(
    monkeypatch: pytest.MonkeyPatch,
    event_id: str,
) -> None:
    ancestors = {}
    for minute in (5, 10, 15, 20, 25):
        identity = f"bar:{minute}"
        ancestors[identity] = SimpleNamespace(
            event_id=identity,
            origin=EventOrigin.NORMALIZED_DATA,
            kind=EventKind.BAR_COMPLETED,
            timeframe=Timeframe.M5,
            known_at=_clock(minute),
            evidence={"real_completed": True, "clock_only": False},
        )
    observer = SimpleNamespace(
        audit_store=SimpleNamespace(get=lambda identity: ancestors.get(identity))
    )
    event = SimpleNamespace(
        event_id=event_id,
        event_time=_clock(10),
        known_at=_clock(20),
    )
    monkeypatch.setattr(
        "scripts.run_mbo_mechanism_research.resolve_source_lineage_tokens",
        lambda *_args, **_kwargs: frozenset(
            f"event:{identity}" for identity in ancestors
        ),
    )
    event_ids, clocks = _source_only_episode_m5_bars(event, observer)
    assert event_ids == ("bar:10", "bar:15", "bar:20")
    assert clocks == (_clock(10), _clock(15), _clock(20))


def test_registered_synthetic_clock_requires_zero_trade_and_fill_flow() -> None:
    frame = _feature_frame(5)
    zero_fields = (
        "aggressor_buy_volume",
        "aggressor_sell_volume",
        "aggressor_unknown_volume",
        "aggressor_buy_trade_count",
        "aggressor_sell_trade_count",
        "aggressor_unknown_trade_count",
        "passive_bid_fill_volume",
        "passive_ask_fill_volume",
        "passive_unknown_fill_volume",
        "passive_bid_fill_count",
        "passive_ask_fill_count",
        "passive_unknown_fill_count",
    )
    frame.loc[2, list(zero_fields)] = 0.0
    # Displayed A/C is book flow and is explicitly allowed at this clock.
    assert frame.loc[2, "displayed_bid_add_volume"] > 0.0
    validate_registered_synthetic_mbo_flow(frame, (_clock(2),))

    invalid = frame.copy()
    invalid.loc[2, "aggressor_buy_trade_count"] = 1.0
    with pytest.raises(Phase6ResearchError, match="nonzero MBO trade/fill"):
        validate_registered_synthetic_mbo_flow(invalid, (_clock(2),))


def test_reader_census_requires_the_exact_registered_synthetic_clock() -> None:
    feature_clocks = frozenset({_clock(0), _clock(1), _clock(2)})
    registered = {
        "window_id": "test-window",
        "completed_clocks": 3,
        "real_completed": 2,
        "synthetic_no_trade": 1,
        "synthetic_decision_clocks": [_clock(1).isoformat()],
    }
    values = {
        "registered": registered,
        "active_window_id": "test-window",
        "feature_clocks": feature_clocks,
        "observed_clocks": set(feature_clocks),
        "completed": 3,
        "real": 2,
        "synthetic": 1,
        "synthetic_clocks": {_clock(1)},
        "first": _clock(0),
        "last": _clock(2),
        "contracts": {("NQM4", 13743)},
        "expected_contract": ("NQM4", 13743),
        "data_gap_resets": 0,
        "contract_changes": 0,
    }
    _validate_reader_census(**values)
    with pytest.raises(Phase6ResearchError, match="replay census"):
        _validate_reader_census(
            **{**values, "synthetic_clocks": {_clock(2)}}
        )
