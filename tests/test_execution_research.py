from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.execution import TopOfBook
from smc_trader.execution_research import (
    EXECUTION_RESEARCH_PROTOCOL_SHA256,
    ExecutionMethod,
    ExecutionResearchConfig,
    ExecutionResearchError,
    ExecutionResearchIntent,
    LIMIT_EXECUTION_METHODS,
    MinuteExecutionInput,
    OrderResearchStatus,
    evaluate_execution_research_intent,
    load_execution_research_config,
    minute_execution_inputs_from_phase6_frame,
    summarize_execution_research,
)
from smc_trader.mbo_mechanism import MBO_MECHANISM_COLUMNS
from smc_trader.model import Bar, Direction
from scripts.check_phase8_execution_readiness import audit_execution_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/execution_research_v1.json"
TEMPLATE = ROOT / "experiments/manifests/execution_research_phase8_v1_template.yaml"
CONFIG_SHA256 = "8212939f9dc00b11c285063a78d56f8ddbc5257a728b02829017a6d60e0d08b7"
MAPPING_ID = "instrument-map:NQ-front-to-NQM4-13743:v1"
MAPPING_SHA256 = "c" * 64


def _clock(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-06-03T13:30:00Z") + pd.Timedelta(minute, unit="m")


def _intent(
    *,
    quantity: int = 1,
    expires_minute: int = 6,
    analysis_ends_minute: int | None = None,
    prices: dict[ExecutionMethod, float] | None = None,
    instrument_id: str | int = 13743,
    source_trade_intent_id: str = "trade-intent:fixed",
) -> ExecutionResearchIntent:
    prices = prices or {method: 99.5 for method in LIMIT_EXECUTION_METHODS}
    return ExecutionResearchIntent(
        source_trade_intent_id=source_trade_intent_id,
        created_at=_clock(0),
        expires_at=_clock(expires_minute),
        symbol="NQM4",
        instrument_id=instrument_id,
        vendor_instrument_id=13743,
        instrument_mapping_id=MAPPING_ID,
        instrument_mapping_sha256=MAPPING_SHA256,
        side=Direction.LONG,
        quantity=quantity,
        tick_size=0.25,
        point_value=20.0,
        arrival_mid=100.125,
        invalidation_price=99.0,
        target_price=102.0,
        method_prices=tuple((method, prices[method]) for method in LIMIT_EXECUTION_METHODS),
        source_identity_ids=("signal:1", source_trade_intent_id),
        paired_stratum=("dfp", "opening_expansion", "long"),
        analysis_ends_at=(
            None
            if analysis_ends_minute is None
            else _clock(analysis_ends_minute)
        ),
    )


def _short_intent() -> ExecutionResearchIntent:
    return ExecutionResearchIntent(
        source_trade_intent_id="trade-intent:short",
        created_at=_clock(0),
        expires_at=_clock(6),
        symbol="NQM4",
        instrument_id=13743,
        vendor_instrument_id=13743,
        instrument_mapping_id=MAPPING_ID,
        instrument_mapping_sha256=MAPPING_SHA256,
        side=Direction.SHORT,
        quantity=1,
        tick_size=0.25,
        point_value=20.0,
        arrival_mid=100.125,
        invalidation_price=101.0,
        target_price=98.0,
        method_prices=tuple((method, 100.75) for method in LIMIT_EXECUTION_METHODS),
        source_identity_ids=("signal:short", "trade-intent:short"),
        paired_stratum=("lsr", "opening_expansion", "short"),
    )


def _input(
    minute: int,
    *,
    low: float = 99.75,
    high: float = 100.5,
    bid: float = 100.0,
    ask: float = 100.25,
    bid_size: float = 10.0,
    ask_size: float = 10.0,
    passive_bid: float = 0.0,
    passive_ask: float = 0.0,
    book_valid: bool = True,
    invalid_reason: str | None = None,
    stale_seconds: float = 0.0,
    source_reset: bool = False,
    synthetic_source: bool = False,
    mechanism_missing: bool = False,
    symbol: str = "NQM4",
    logical_instrument_id: str | int = 13743,
) -> MinuteExecutionInput:
    clock = _clock(minute)
    bar = Bar(
        start=clock - pd.Timedelta(1, unit="m"),
        open=100.0,
        high=high,
        low=low,
        close=100.0,
        volume=0.0 if synthetic_source else 100.0,
        symbol=symbol,
        instrument_id=13743,
        synthetic_no_trade=synthetic_source,
    )
    book = (
        TopOfBook(
            observed_at=clock - pd.Timedelta(stale_seconds, unit="s"),
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )
        if book_valid
        else None
    )
    flow = None if mechanism_missing else 0.0
    return MinuteExecutionInput(
        decision_time=clock,
        symbol=symbol,
        instrument_id=logical_instrument_id,
        vendor_instrument_id=13743,
        instrument_mapping_id=MAPPING_ID,
        instrument_mapping_sha256=MAPPING_SHA256,
        bar=bar,
        book=book,
        book_valid=book_valid,
        invalid_reason=invalid_reason,
        passive_bid_fill_volume=None if mechanism_missing else passive_bid,
        passive_ask_fill_volume=None if mechanism_missing else passive_ask,
        displayed_bid_add_volume=flow,
        displayed_ask_add_volume=flow,
        displayed_bid_cancel_volume=flow,
        displayed_ask_cancel_volume=flow,
        aggressor_buy_volume=flow,
        aggressor_sell_volume=flow,
        source_reset=source_reset,
        synthetic_source=synthetic_source,
        source_artifact_id="phase6:mbo:week1",
        source_artifact_sha256="a" * 64,
        source_row_sha256=f"{minute:064x}",
    )


def _ledger(**overrides) -> tuple[MinuteExecutionInput, ...]:
    return tuple(_input(minute, **overrides) for minute in range(7))


def _outcome(study, method: ExecutionMethod):
    return next(item for item in study.outcomes if item.method is method)


def test_versioned_config_and_incomplete_template_are_identity_bound() -> None:
    config = load_execution_research_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config.authority == "research_only_never_submit"
    assert config.config_id == f"execution-research-config:{EXECUTION_RESEARCH_PROTOCOL_SHA256}"
    assert tuple(config.ordered_methods) == tuple(ExecutionMethod)

    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["status"] == "template_incomplete_not_authorized_to_run"
    assert template["identity_bindings"]["protocol_config"]["sha256"] == CONFIG_SHA256
    assert template["preregistration"]["canonical_config_id"] == (
        config.config_id
    )
    assert template["preregistration"]["fixed_contrasts"] == 6
    assert template["preregistration"]["tuning_allowed"] is False
    assert template["authority"]["queue_truth_claimed"] is False
    assert template["frozen_before_run"] is False
    assert template["experiment_id"] is None
    assert all(value is None for value in template["outputs"].values())
    assert template["preregistration"]["cross_intent_independence_claimed"] is False
    assert template["preregistration"]["confirmatory_claim"] is False
    assert template["formal_input_schema"]["fixture_rows_authorized_as_evidence"] is False
    assert all(
        template["method_price_rules"][method.value] is None
        for method in LIMIT_EXECUTION_METHODS
    )
    assert all(
        template["variants"][family] == []
        for family in ("wait_time", "cancel_rule", "stop", "target")
    )
    readiness = audit_execution_readiness(TEMPLATE)
    assert readiness["ready"] is False
    assert "missing_trade_intent_ledger" in readiness["readiness_blockers"]

    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["tick_size"] = 0.5
    with pytest.raises(ExecutionResearchError, match="version bump"):
        ExecutionResearchConfig.from_payload(changed)
    with pytest.raises(ExecutionResearchError, match="identity drift"):
        replace(config, minimum_complete_pairs=1)
    intent = _intent()
    assert intent.entry_expires_at == intent.expires_at
    assert intent.analysis_ends_at == intent.entry_expires_at
    with pytest.raises(ExecutionResearchError, match="contract is invalid"):
        replace(intent, schema_version="phase8_execution_research_v1.0")
    with pytest.raises(ExecutionResearchError, match="contract is invalid"):
        _intent(expires_minute=6, analysis_ends_minute=5)


def test_phase6_frame_adapter_preserves_exact_minute_grain_and_row_identity() -> None:
    row = {column: 0.0 for column in MBO_MECHANISM_COLUMNS}
    row.update(
        {
            "decision_time": _clock(0),
            "symbol": "NQM4",
            "instrument_id": 13743,
            "book_observed_at": _clock(0),
            "publisher_id": 2,
            "sequence": 1,
            "bid": 100.0,
            "ask": 100.25,
            "bid_size": 10.0,
            "ask_size": 12.0,
            "top5_bid_size": 20.0,
            "top5_ask_size": 22.0,
            "depth_imbalance": -2.0 / 42.0,
            "book_valid": True,
            "invalid_reason": "",
            "book_age_seconds": 0.0,
            "mid_price": 100.125,
            "spread_points": 0.25,
            "spread_ticks": 1.0,
            "book_change_valid": False,
            "best_level_ofi_contracts": None,
            "depth_imbalance_change": None,
            "mid_change_points": None,
            "mid_change_ticks": None,
            "absolute_mid_impact_ticks_per_aggressor_contract": None,
            "signed_mid_impact_ticks_per_net_aggressor_contract": None,
            "book_valid_clock_fraction": 1.0,
        }
    )
    bar = Bar(
        start=_clock(0) - pd.Timedelta(1, unit="m"),
        open=100.0,
        high=100.5,
        low=99.75,
        close=100.0,
        volume=100.0,
        symbol="NQM4",
        instrument_id=13743,
    )
    inputs = minute_execution_inputs_from_phase6_frame(
        pd.DataFrame([row]),
        (bar,),
        source_artifact_id="phase6:mbo:week1",
        source_artifact_sha256="a" * 64,
        logical_instrument_id=13743,
        instrument_mapping_id=MAPPING_ID,
        instrument_mapping_sha256=MAPPING_SHA256,
    )

    assert len(inputs) == 1
    assert inputs[0].decision_time == _clock(0)
    assert len(inputs[0].source_row_sha256) == 64
    assert inputs[0].book == TopOfBook(
        observed_at=_clock(0), bid=100.0, ask=100.25, bid_size=10.0, ask_size=12.0
    )
    changed = dict(row)
    changed["passive_bid_fill_volume"] = 1.0
    changed_inputs = minute_execution_inputs_from_phase6_frame(
        pd.DataFrame([changed]),
        (bar,),
        source_artifact_id="phase6:mbo:week1",
        source_artifact_sha256="a" * 64,
        logical_instrument_id=13743,
        instrument_mapping_id=MAPPING_ID,
        instrument_mapping_sha256=MAPPING_SHA256,
    )
    assert changed_inputs[0].source_row_sha256 != inputs[0].source_row_sha256
    assert changed_inputs[0].input_id != inputs[0].input_id


def test_future_book_is_rejected_at_the_causal_input_boundary() -> None:
    clock = _clock(0)
    with pytest.raises(ExecutionResearchError, match="future BBO"):
        MinuteExecutionInput(
            decision_time=clock,
            symbol="NQM4",
            instrument_id=13743,
            vendor_instrument_id=13743,
            instrument_mapping_id=MAPPING_ID,
            instrument_mapping_sha256=MAPPING_SHA256,
            bar=Bar(
                start=clock - pd.Timedelta(1, unit="m"),
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.0,
                volume=1.0,
                symbol="NQM4",
                instrument_id=13743,
            ),
            book=TopOfBook(
                observed_at=clock + pd.Timedelta(1, unit="ms"),
                bid=100.0,
                ask=100.25,
                bid_size=1.0,
                ask_size=1.0,
            ),
            book_valid=True,
            invalid_reason=None,
            passive_bid_fill_volume=0.0,
            passive_ask_fill_volume=0.0,
            displayed_bid_add_volume=0.0,
            displayed_ask_add_volume=0.0,
            displayed_bid_cancel_volume=0.0,
            displayed_ask_cancel_volume=0.0,
            aggressor_buy_volume=0.0,
            aggressor_sell_volume=0.0,
            source_reset=False,
            synthetic_source=False,
            source_artifact_id="phase6:mbo:week1",
            source_artifact_sha256="a" * 64,
            source_row_sha256="b" * 64,
        )


def test_symbolic_logical_instrument_uses_explicit_hash_bound_vendor_mapping() -> None:
    config = load_execution_research_config(CONFIG)
    intent = _intent(instrument_id="NQ:front")
    inputs = tuple(
        _input(minute, logical_instrument_id="NQ:front")
        for minute in range(7)
    )
    study = evaluate_execution_research_intent(intent, inputs, config)
    assert study.research_intent_id == intent.research_intent_id

    with pytest.raises(ExecutionResearchError, match="intent-parallel"):
        evaluate_execution_research_intent(
            intent,
            tuple(_input(minute, logical_instrument_id=13743) for minute in range(7)),
            config,
        )
    with pytest.raises(ExecutionResearchError, match="intent-parallel"):
        evaluate_execution_research_intent(
            intent,
            tuple(
                replace(
                    _input(minute, logical_instrument_id="NQ:front"),
                    instrument_mapping_id="instrument-map:wrong",
                )
                for minute in range(7)
            ),
            config,
        )


def test_intent_cost_contract_and_arrival_mid_cannot_drift_from_protocol_or_bbo() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = _ledger()
    with pytest.raises(ExecutionResearchError, match="tick_size/point_value"):
        evaluate_execution_research_intent(
            replace(_intent(), tick_size=0.5), inputs, config
        )
    with pytest.raises(ExecutionResearchError, match="tick_size/point_value"):
        evaluate_execution_research_intent(
            replace(_intent(), point_value=50.0), inputs, config
        )
    with pytest.raises(ExecutionResearchError, match="first causal BBO"):
        evaluate_execution_research_intent(
            replace(_intent(), arrival_mid=100.0), inputs, config
        )


@pytest.mark.parametrize(
    ("inputs", "reason"),
    [
        ((_input(0, book_valid=False, invalid_reason="missing_top_of_book"),), "missing_top_of_book"),
        ((_input(0, book_valid=False, invalid_reason="crossed_book"),), "crossed_top_of_book"),
        ((_input(0, stale_seconds=61.0),), "stale_top_of_book"),
        ((_input(0, source_reset=True),), "reset_source"),
        ((_input(0, synthetic_source=True),), "synthetic_source"),
        ((_input(0, mechanism_missing=True),), "missing_mbo_mechanism"),
    ],
)
def test_invalid_sources_are_censored_never_counted_as_unfilled(inputs, reason) -> None:
    config = load_execution_research_config(CONFIG)
    study = evaluate_execution_research_intent(_intent(), inputs, config)
    assert all(item.terminal_status is OrderResearchStatus.CENSORED for item in study.outcomes)
    assert {item.terminal_reason for item in study.outcomes} == {reason}
    summary = summarize_execution_research((study,), config)
    assert all(item.eligible_non_censored == 0 for item in summary.method_summaries)
    assert all(item.full_fill_probability is None for item in summary.method_summaries)


def test_same_intent_methods_share_exact_input_ledger_and_are_deterministic() -> None:
    config = load_execution_research_config(CONFIG)
    prices = {method: 100.25 for method in LIMIT_EXECUTION_METHODS}
    intent = _intent(prices=prices)
    inputs = _ledger()
    left = evaluate_execution_research_intent(intent, inputs, config)
    right = evaluate_execution_research_intent(intent, tuple(reversed(inputs)), config)

    assert left.study_id == right.study_id
    assert tuple(item.method for item in left.outcomes) == tuple(ExecutionMethod)
    assert all(item.research_intent_id == intent.research_intent_id for item in left.outcomes)
    assert len({item.source_input_ids for item in left.outcomes}) == 1
    assert all(item.terminal_status is OrderResearchStatus.FILLED for item in left.outcomes)
    assert all(item.queue_truth_available is False for item in left.outcomes)


def test_limit_fill_uses_f_proxy_and_same_bar_is_adverse_first() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=98.75, high=102.25, passive_bid=1.0),
        *(_input(minute) for minute in range(2, 7)),
    )
    study = evaluate_execution_research_intent(_intent(), inputs, config)
    fvg = _outcome(study, ExecutionMethod.FVG_50_LIMIT)

    assert fvg.terminal_status is OrderResearchStatus.FILLED
    assert fvg.first_fill_at == _clock(1)
    assert fvg.average_fill_price == 99.5
    assert fvg.position_outcome == "same_bar_ambiguous_invalidation_first"
    assert fvg.target_before_invalidation is False
    assert fvg.mae_points == pytest.approx(0.75)
    assert fvg.mfe_points == 0.0


def test_short_side_uses_bid_market_and_passive_ask_fill_symmetrically() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=97.75, high=101.25, passive_ask=1.0),
        *(_input(minute) for minute in range(2, 7)),
    )
    study = evaluate_execution_research_intent(_short_intent(), inputs, config)
    market = _outcome(study, ExecutionMethod.MARKET)
    fvg = _outcome(study, ExecutionMethod.FVG_50_LIMIT)

    assert market.average_fill_price == 100.0
    assert fvg.average_fill_price == 100.75
    assert fvg.position_outcome == "same_bar_ambiguous_invalidation_first"
    assert fvg.target_before_invalidation is False
    assert fvg.mae_points == pytest.approx(0.5)


def test_partial_proxy_fill_expires_and_is_not_promoted_to_full_fill() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = tuple(
        _input(
            minute,
            low=99.4 if minute == 1 else 99.75,
            passive_bid=1.0 if minute == 1 else 0.0,
        )
        for minute in range(4)
    )
    study = evaluate_execution_research_intent(
        _intent(quantity=2, expires_minute=3), inputs, config
    )
    fvg = _outcome(study, ExecutionMethod.FVG_50_LIMIT)

    assert fvg.terminal_status is OrderResearchStatus.EXPIRED
    assert fvg.terminal_reason == "good_til_time_expired"
    assert fvg.filled_quantity == 1.0
    assert fvg.fill_fraction == 0.5
    assert fvg.partial_fill is True
    assert fvg.full_fill_at is None


def test_multiple_passive_proxy_fills_conserve_remaining_quantity_to_full_fill() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = tuple(
        _input(
            minute,
            low=99.4 if minute in {1, 2} else 99.75,
            passive_bid=1.0 if minute in {1, 2} else 0.0,
        )
        for minute in range(7)
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_intent(quantity=2), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.terminal_status is OrderResearchStatus.FILLED
    assert fvg.filled_quantity == 2.0
    assert fvg.fill_fraction == 1.0
    assert fvg.partial_fill is False
    assert fvg.first_fill_at == _clock(1)
    assert fvg.full_fill_at == _clock(2)


def test_partial_proxy_fill_is_preserved_when_later_source_censors_remainder() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.4, passive_bid=1.0),
        _input(2, book_valid=False, invalid_reason="missing_top_of_book"),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_intent(quantity=2), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.terminal_status is OrderResearchStatus.CENSORED
    assert fvg.terminal_reason == "missing_top_of_book"
    assert fvg.filled_quantity == 1.0
    assert fvg.fill_fraction == 0.5
    assert fvg.partial_fill is True


def test_fractional_passive_f_capacity_censors_instead_of_becoming_zero_unfilled() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.4, passive_bid=0.9),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_intent(), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )
    assert fvg.terminal_status is OrderResearchStatus.CENSORED
    assert fvg.terminal_reason == "fractional_contract_capacity"
    assert fvg.filled_quantity == 0.0


def test_prior_partial_position_resolves_bar_before_later_clock_quote_fill() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.4, passive_bid=1.0),
        _input(2, low=99.75, high=102.25, bid=99.25, ask=99.5),
        *(_input(minute) for minute in range(3, 7)),
    )
    study = evaluate_execution_research_intent(_intent(quantity=2), inputs, config)
    fvg = _outcome(study, ExecutionMethod.FVG_50_LIMIT)

    assert fvg.terminal_status is OrderResearchStatus.CANCELLED
    assert fvg.terminal_reason == "target_cancelled_remainder"
    assert fvg.filled_quantity == 1.0
    assert fvg.position_outcome == "target"
    assert fvg.target_before_invalidation is True


def test_prior_partial_stop_cancels_remainder_before_end_clock_marketable_quote() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.4, passive_bid=1.0),
        _input(2, low=98.75, high=100.5, bid=99.25, ask=99.5),
        *(_input(minute) for minute in range(3, 7)),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_intent(quantity=2), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.terminal_status is OrderResearchStatus.CANCELLED
    assert fvg.terminal_reason == "invalidation_cancelled_remainder"
    assert fvg.filled_quantity == 1.0
    assert fvg.fill_fraction == 0.5
    assert fvg.full_fill_at is None
    assert fvg.position_outcome == "invalidation"


def test_target_on_new_resting_fill_bar_is_not_credited_until_a_later_bar() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.4, high=102.25, passive_bid=1.0),
        _input(2, low=99.75, high=102.25),
        *(_input(minute) for minute in range(3, 7)),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_intent(), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.first_fill_at == _clock(1)
    assert fvg.position_outcome == "target"
    assert fvg.target_before_invalidation is True
    assert fvg.last_evaluated_at == _clock(2)


def test_target_before_pending_proxy_fill_terminates_and_cannot_later_fill() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.75, high=102.25),
        _input(2, low=99.4, passive_bid=1.0),
        *(_input(minute) for minute in range(3, 7)),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_intent(), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.terminal_status is OrderResearchStatus.CANCELLED
    assert fvg.terminal_reason == "target_before_proxy_fill"
    assert fvg.filled_quantity == 0.0
    assert fvg.first_fill_at is None
    assert fvg.last_evaluated_at == _clock(1)


def test_short_target_before_pending_proxy_fill_is_symmetric() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=97.75, high=100.5),
        _input(2, high=101.0, passive_ask=1.0),
        *(_input(minute) for minute in range(3, 7)),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(_short_intent(), inputs, config),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.terminal_status is OrderResearchStatus.CANCELLED
    assert fvg.terminal_reason == "target_before_proxy_fill"
    assert fvg.filled_quantity == 0.0
    assert fvg.last_evaluated_at == _clock(1)


def test_post_gtt_rows_track_partial_position_but_never_add_entry_fills() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (
        _input(0),
        _input(1, low=99.4, passive_bid=1.0),
        _input(2),
        _input(3, low=99.4, passive_bid=1.0),
        _input(4, low=99.4, passive_bid=1.0),
        _input(5, low=99.4, high=102.25, passive_bid=1.0),
    )
    fvg = _outcome(
        evaluate_execution_research_intent(
            _intent(quantity=2, expires_minute=2, analysis_ends_minute=5),
            inputs,
            config,
        ),
        ExecutionMethod.FVG_50_LIMIT,
    )

    assert fvg.terminal_status is OrderResearchStatus.EXPIRED
    assert fvg.terminal_reason == "good_til_time_expired"
    assert fvg.filled_quantity == 1.0
    assert fvg.fill_fraction == 0.5
    assert fvg.full_fill_at is None
    assert fvg.position_outcome == "target"
    assert fvg.target_before_invalidation is True
    assert fvg.last_evaluated_at == _clock(5)


def test_primary_pair_ignores_unavailable_secondary_horizon_metric() -> None:
    config = load_execution_research_config(CONFIG)
    prices = {method: 100.25 for method in LIMIT_EXECUTION_METHODS}
    study = evaluate_execution_research_intent(
        _intent(
            prices=prices,
            expires_minute=1,
            analysis_ends_minute=2,
        ),
        tuple(_input(minute) for minute in range(3)),
        config,
    )

    assert all(
        outcome.implementation_shortfall_points is not None
        and outcome.realized_spread_points is None
        and "realized_spread_horizon_unavailable"
        in outcome.analysis_censor_reasons
        for outcome in study.outcomes
    )
    summary = summarize_execution_research((study,), config)
    assert all(item.complete_pairs == 1 for item in summary.paired_contrasts)
    assert all(
        item.mean_implementation_shortfall_points is not None
        and item.mean_realized_spread_points is None
        for item in summary.method_summaries
    )


def test_insufficient_market_depth_records_partial_then_censors_unknown_remainder() -> None:
    config = load_execution_research_config(CONFIG)
    study = evaluate_execution_research_intent(
        _intent(quantity=2), _ledger(ask_size=0.5), config
    )
    market = _outcome(study, ExecutionMethod.MARKET)

    assert market.terminal_status is OrderResearchStatus.CENSORED
    assert market.terminal_reason == "fractional_contract_capacity"
    assert market.filled_quantity == 0.0
    assert market.partial_fill is False


def test_off_tick_method_is_rejected_without_price_fallback() -> None:
    config = load_execution_research_config(CONFIG)
    prices = {method: 99.5 for method in LIMIT_EXECUTION_METHODS}
    prices[ExecutionMethod.FVG_50_LIMIT] = 99.6
    study = evaluate_execution_research_intent(_intent(prices=prices), _ledger(), config)
    fvg = _outcome(study, ExecutionMethod.FVG_50_LIMIT)
    ob = _outcome(study, ExecutionMethod.OB_50_LIMIT)

    assert fvg.terminal_status is OrderResearchStatus.REJECTED
    assert fvg.terminal_reason == "off_tick_price"
    assert fvg.order_price == 99.6
    assert ob.order_price == 99.5
    assert ob.terminal_status is not OrderResearchStatus.REJECTED


@pytest.mark.parametrize(
    ("field_name", "price"),
    (("invalidation_price", 99.1), ("target_price", 102.1)),
)
def test_intent_rejects_off_tick_structural_prices(
    field_name: str,
    price: float,
) -> None:
    with pytest.raises(ExecutionResearchError, match="contract is invalid"):
        replace(_intent(), **{field_name: price})


def test_missing_minute_censors_working_limits_but_does_not_rewrite_market_fill() -> None:
    config = load_execution_research_config(CONFIG)
    inputs = (_input(0), _input(2), _input(3), _input(4), _input(5), _input(6))
    study = evaluate_execution_research_intent(_intent(), inputs, config)

    assert _outcome(study, ExecutionMethod.FVG_50_LIMIT).terminal_reason == "missing_clock"
    market = _outcome(study, ExecutionMethod.MARKET)
    assert market.terminal_status is OrderResearchStatus.FILLED
    assert "missing_clock" in market.analysis_censor_reasons


def test_summary_uses_six_fixed_same_intent_holm_contrasts() -> None:
    config = load_execution_research_config(CONFIG)
    prices = {method: 100.25 for method in LIMIT_EXECUTION_METHODS}
    study = evaluate_execution_research_intent(
        _intent(prices=prices), _ledger(), config
    )
    summary = summarize_execution_research((study,), config)

    assert len(summary.method_summaries) == 7
    assert len(summary.paired_contrasts) == 6
    assert all(item.complete_pairs == 1 for item in summary.paired_contrasts)
    assert all(
        item.status == "underpowered_descriptive_only"
        for item in summary.paired_contrasts
    )
    assert all(
        item.cross_intent_independence_claimed is False
        and item.confirmatory_claim is False
        for item in summary.paired_contrasts
    )
    assert all(0.0 <= item.holm_adjusted_pvalue <= 1.0 for item in summary.paired_contrasts)
    assert summary.summary_id.startswith("execution-research-summary:")

    with pytest.raises(ExecutionResearchError, match="duplicate-source"):
        summarize_execution_research((study, study), config)
    with pytest.raises(ExecutionResearchError, match="intent/input parity"):
        replace(study, source_trade_intent_id="trade-intent:forged")

    fvg = _outcome(study, ExecutionMethod.FVG_50_LIMIT)
    with pytest.raises(ExecutionResearchError, match="outcome invariants"):
        replace(
            fvg,
            filled_quantity=0.0,
            fill_fraction=1.0,
            full_fill_at=None,
            average_fill_price=None,
            implementation_shortfall_points=-999.0,
        )

    half_quantity_forge = replace(
        fvg,
        intended_quantity=2,
        terminal_status=OrderResearchStatus.EXPIRED,
        terminal_reason="good_til_time_expired",
        fill_fraction=0.5,
        partial_fill=True,
        full_fill_at=None,
        time_to_full_fill_seconds=None,
    )
    forged_outcomes = tuple(
        half_quantity_forge if item.method is ExecutionMethod.FVG_50_LIMIT else item
        for item in study.outcomes
    )
    with pytest.raises(ExecutionResearchError, match="intent/input parity"):
        replace(study, outcomes=forged_outcomes)

    order_price_forge = replace(
        fvg,
        order_price=99.5,
        slippage_points=fvg.side.sign * (fvg.average_fill_price - 99.5),
    )
    forged_outcomes = tuple(
        order_price_forge if item.method is ExecutionMethod.FVG_50_LIMIT else item
        for item in study.outcomes
    )
    with pytest.raises(ExecutionResearchError, match="intent/input parity"):
        replace(study, outcomes=forged_outcomes)

    unfilled_study = evaluate_execution_research_intent(_intent(), _ledger(), config)
    unfilled = _outcome(unfilled_study, ExecutionMethod.FVG_50_LIMIT)
    with pytest.raises(ExecutionResearchError, match="outcome invariants"):
        replace(
            unfilled,
            terminal_status=OrderResearchStatus.CENSORED,
            terminal_reason="full_fill",
        )


def test_same_source_trade_intent_cannot_be_recounted_as_independent_study() -> None:
    config = load_execution_research_config(CONFIG)
    first_prices = {method: 100.25 for method in LIMIT_EXECUTION_METHODS}
    second_prices = {method: 100.5 for method in LIMIT_EXECUTION_METHODS}
    first = evaluate_execution_research_intent(
        _intent(prices=first_prices), _ledger(), config
    )
    second = evaluate_execution_research_intent(
        _intent(prices=second_prices), _ledger(), config
    )
    assert first.research_intent_id != second.research_intent_id
    assert first.source_trade_intent_id == second.source_trade_intent_id
    with pytest.raises(ExecutionResearchError, match="duplicate-source"):
        summarize_execution_research((first, second), config)


def test_intent_input_contract_mismatch_fails_pairing_instead_of_falling_back() -> None:
    config = load_execution_research_config(CONFIG)
    with pytest.raises(ExecutionResearchError, match="intent-parallel"):
        evaluate_execution_research_intent(_intent(), (_input(0, symbol="ESM4"),), config)
