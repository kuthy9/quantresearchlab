from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.execution import TopOfBook
from smc_trader.execution_research import (
    ORDERED_EXECUTION_METHODS,
    ExecutionMethod,
    MinuteExecutionInput,
    load_execution_research_config,
)
from smc_trader.execution_research_runner import (
    PAIRED_ROWS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    Phase8RunManifestContract,
    Phase8RunnerError,
    SourceMode,
    evaluate_phase8_research_cases,
    load_execution_research_runner_protocol,
    minute_execution_input_from_payload,
    minute_execution_input_to_payload,
    paired_bootstrap_interval,
    summarize_phase8_paired_rows,
    validate_phase8_run_manifest,
    write_phase8_research_outputs,
)
from smc_trader.execution_research_v2 import (
    IntentLedgerRecord,
    MethodAvailability,
    MethodPriceProvenance,
    MethodPriceSet,
    Phase8AppendOnlyLedger,
    build_research_case_record,
    load_execution_research_v2_config,
    load_risk_admission_protocol,
)
from smc_trader.model import Bar, Direction, LiquidityLevel, StructuralLevel, Timeframe
from smc_trader.signal_policy import CancelCondition, CancelConditionKind, SetupFamily
from smc_trader.trade_intent import EntryMethod, TimeInForce, TradeIntent


pytestmark = pytest.mark.research_runner


ROOT = Path(__file__).resolve().parents[1]
RUNNER_CONFIG = ROOT / "configs/execution_research_runner_v1.json"
V1_CONFIG = ROOT / "configs/execution_research_v1.json"
V2_CONFIG = ROOT / "configs/execution_research_v2.json"
RISK_CONFIG = ROOT / "configs/risk_admission_v1.json"
RUN_TEMPLATE = (
    ROOT / "configs/research/execution_research_phase8_v2_run_template.yaml"
)
T0 = pd.Timestamp("2024-06-03T13:30:00Z")
EXPIRY = T0 + pd.Timedelta(minutes=6)
SEMANTIC_SHA = "a" * 64


METHOD_RULES = {
    ExecutionMethod.MARKET: (
        "causal_arrival_bbo",
        "first_causal_valid_best_quote_ticks_v1",
        "reject_off_tick_no_rounding",
    ),
    ExecutionMethod.FVG_50_LIMIT: (
        "fvg_generation",
        "fvg_midpoint_ticks_v1",
        "nearest_tick_half_even_v1",
    ),
    ExecutionMethod.OB_50_LIMIT: (
        "qualified_order_block_generation",
        "qualified_ob_midpoint_ticks_v1",
        "nearest_tick_half_even_v1",
    ),
    ExecutionMethod.RECLAIM_LIMIT: (
        "liquidity_interaction_generation",
        "first_outside_close_reclaim_ticks_v1",
        "reject_off_tick_no_rounding",
    ),
    ExecutionMethod.BREAKOUT_LIMIT: (
        "structure_transition_generation",
        "first_confirming_outside_close_ticks_v1",
        "reject_off_tick_no_rounding",
    ),
    ExecutionMethod.AGGRESSIVE_LIMIT: (
        "causal_arrival_bbo",
        "near_quote_one_tick_improvement_v1",
        "exact_integer_tick_arithmetic",
    ),
    ExecutionMethod.PASSIVE_LIMIT: (
        "causal_arrival_bbo",
        "far_quote_one_tick_improvement_v1",
        "exact_integer_tick_arithmetic",
    ),
}


def _intent() -> TradeIntent:
    invalidation = StructuralLevel(
        price=95.0,
        side="below",
        source_level_id="structure:invalidation",
        observed_at=T0 - pd.Timedelta(minutes=1),
        rationale="frozen structural invalidation",
    )
    target = LiquidityLevel(
        level_id="liquidity:target",
        timeframe=Timeframe.H1,
        side="above",
        price=110.0,
        formed_at=T0 - pd.Timedelta(hours=2),
        confirmed_at=T0 - pd.Timedelta(hours=1),
        touches=1,
    )
    cancel = CancelCondition(
        kind=CancelConditionKind.SIGNAL_EXPIRY_REACHED,
        reference_id="signal:test",
        operator=">=",
        source_ids=("event:signal",),
        trigger_at=EXPIRY,
    )
    return TradeIntent(
        created_at=T0,
        expires_at=EXPIRY,
        signal_id="signal:test",
        candidate_id="candidate:test",
        episode_id="episode:test",
        setup_id="setup:test",
        setup_family=SetupFamily.DFP,
        competition_set_id="competition:test",
        path_hypothesis_id="path:test",
        dol_ranking_id="dol-ranking:test",
        dol_candidate_id="dol:test",
        source_event_ids=("event:path", "event:setup"),
        source_identity_ids=("identity:path", "identity:setup"),
        policy_protocol_id="signal-policy:test",
        policy_protocol_version="signal-policy-test-v1",
        policy_protocol_fingerprint="f" * 64,
        path_likelihood_artifact_id="path-artifact:test",
        path_model_id="path-model:test",
        path_model_version="path-model-test-v1",
        path_calibration_id="path-calibration:test",
        dol_calibration_artifact_id="dol-artifact:test",
        dol_model_id="dol-model:test",
        dol_model_version="dol-model-test-v1",
        dol_calibration_id="dol-calibration:test",
        outcome_model_artifact_id="outcome-artifact:test",
        outcome_model_id="outcome-model:test",
        outcome_model_version="outcome-model-test-v1",
        outcome_calibration_id="outcome-calibration:test",
        symbol="NQM4",
        instrument_id="NQ:front",
        side=Direction.LONG,
        account_snapshot_id="account:test",
        risk_budget_id="risk-budget:test",
        quantity=1,
        point_value=20.0,
        risk_budget_fraction=0.005,
        risk_budget_amount=500.0,
        position_risk_amount=105.0,
        entry_method_preferences=(
            EntryMethod.MARKET_ENTRY,
            EntryMethod.FVG_50_LIMIT,
        ),
        planned_entry=100.25,
        invalidation=invalidation,
        targets=(target,),
        trade_plan_id="trade-plan:test",
        max_wait_seconds=360.0,
        time_in_force=TimeInForce.GOOD_TIL_TIME,
        cancel_conditions=(cancel,),
    )


def _minute(
    minute: int,
    *,
    passive_bid: float = 0.0,
    low: float = 99.75,
) -> MinuteExecutionInput:
    clock = T0 + pd.Timedelta(minutes=minute)
    bar = Bar(
        start=clock - pd.Timedelta(minutes=1),
        open=100.0,
        high=100.5,
        low=low,
        close=100.0,
        volume=100.0,
        symbol="NQM4",
        instrument_id=13743,
    )
    return MinuteExecutionInput(
        decision_time=clock,
        symbol="NQM4",
        instrument_id="NQ:front",
        vendor_instrument_id=13743,
        instrument_mapping_id="instrument-map:NQ-front-to-NQM4-13743:v1",
        instrument_mapping_sha256=(
            "f820a6f4f329e1df667a3b876e2e0e3738f5e7b122638e9846453a4fe5b362bc"
        ),
        bar=bar,
        book=TopOfBook(
            observed_at=clock,
            bid=100.0,
            ask=100.25,
            bid_size=10.0,
            ask_size=10.0,
        ),
        book_valid=True,
        invalid_reason=None,
        passive_bid_fill_volume=passive_bid,
        passive_ask_fill_volume=0.0,
        displayed_bid_add_volume=0.0,
        displayed_ask_add_volume=0.0,
        displayed_bid_cancel_volume=0.0,
        displayed_ask_cancel_volume=0.0,
        aggressor_buy_volume=0.0,
        aggressor_sell_volume=0.0,
        source_reset=False,
        synthetic_source=False,
        source_artifact_id="phase6-mbo:synthetic-test",
        source_artifact_sha256=(
            "9acbee73784cd260b02b30edbcbdd60aad3ec3faa1ae5de9b52a38fbc03d03c7"
        ),
        source_row_sha256=hashlib.sha256(clock.isoformat().encode()).hexdigest(),
    )


def _ledger_and_inputs(
    *,
    resting_fvg: bool = False,
) -> tuple[Phase8AppendOnlyLedger, tuple[MinuteExecutionInput, ...]]:
    runner_protocol = load_execution_research_runner_protocol(RUNNER_CONFIG)
    execution_protocol = load_execution_research_v2_config(V2_CONFIG)
    risk_protocol = load_risk_admission_protocol(RISK_CONFIG)
    inputs = tuple(
        _minute(
            minute,
            passive_bid=(1.0 if resting_fvg and minute == 1 else 0.0),
        )
        for minute in range(127)
    )
    intent_record = IntentLedgerRecord.from_trade_intent(_intent())
    methods: list[MethodPriceProvenance] = []
    for method in ORDERED_EXECUTION_METHODS:
        semantic, derivation, rounding = METHOD_RULES[method]
        ticks = 401 if method is ExecutionMethod.MARKET else 402
        if resting_fvg and method is ExecutionMethod.FVG_50_LIMIT:
            ticks = 400
        inputs_for_derivation = (
            (inputs[0].input_id,)
            if semantic == "causal_arrival_bbo"
            else (f"semantic-input:{method.value}",)
        )
        methods.append(
            MethodPriceProvenance(
                method=method,
                availability=MethodAvailability.AVAILABLE,
                price_ticks=ticks,
                tick_size=0.25,
                source_semantic_type=semantic,
                source_object_id=f"object:{method.value}",
                source_generation_id=f"generation:{method.value}:1",
                source_event_ids=(f"event:{method.value}:1",),
                source_known_at=T0,
                snapshot_asof=T0,
                derivation_id=derivation,
                derivation_input_ids=inputs_for_derivation,
                rounding_rule=rounding,
                source_protocol_sha256=SEMANTIC_SHA,
                availability_reason="available_before_outcome",
            )
        )
    method_prices = MethodPriceSet(
        source_trade_intent_id=intent_record.source_trade_intent_id,
        snapshot_asof=T0,
        tick_size=0.25,
        methods=tuple(methods),
    )
    window = runner_protocol.windows[0]
    case = build_research_case_record(
        intent_record,
        method_prices,
        execution_protocol,
        risk_protocol,
        source_artifact_ids=(
            f"phase6-mbo-sha256:{window.mbo_artifact_sha256}",
            f"ohlcv-sha256:{runner_protocol.ohlcv_sha256}",
            f"instrument-mapping-sha256:{runner_protocol.mapping_sha256}",
        ),
    )
    return Phase8AppendOnlyLedger((intent_record, case)), inputs


def _evaluate(*, resting_fvg: bool = False):
    ledger, inputs = _ledger_and_inputs(resting_fvg=resting_fvg)
    runner_protocol = load_execution_research_runner_protocol(RUNNER_CONFIG)
    execution_protocol = load_execution_research_v2_config(V2_CONFIG)
    risk_protocol = load_risk_admission_protocol(RISK_CONFIG)
    rows = evaluate_phase8_research_cases(
        ledger,
        inputs,
        runner_protocol=runner_protocol,
        execution_protocol=execution_protocol,
        execution_config=load_execution_research_config(V1_CONFIG),
        risk_protocol_id=risk_protocol.protocol_id,
        risk_protocol_sha256=risk_protocol.source_file_sha256,
    )
    return rows, runner_protocol, execution_protocol


def test_run_template_validation_is_inert_and_opens_no_dataset() -> None:
    contract = validate_phase8_run_manifest(
        RUN_TEMPLATE,
        project_root=ROOT,
        validate_input_files=False,
    )
    assert contract.ready is False
    assert contract.blockers == (
        "intent_research_case_ledger_not_bound",
        "minute_source_mode_not_bound",
        "outputs_not_registered",
        "experiment_identity_missing",
        "frozen_clock_missing",
        "manifest_not_frozen",
    )
    assert contract.ledger_path is None
    assert contract.minute_artifact_path is None


def test_minute_execution_payload_round_trip_conserves_identity() -> None:
    value = _minute(0)
    restored = minute_execution_input_from_payload(
        minute_execution_input_to_payload(value)
    )
    assert restored == value
    assert restored.input_id == value.input_id


def test_primary_plus_six_ofat_rows_are_same_case_and_fail_closed() -> None:
    rows, runner_protocol, execution_protocol = _evaluate()
    assert len(rows) == 7
    assert {item["schema_version"] for item in rows} == {PAIRED_ROWS_SCHEMA_VERSION}
    assert len({item["case_record_id"] for item in rows}) == 1
    assert len({tuple(item["source_input_ids"]) for item in rows}) == 1
    assert [item["variant"]["variant_id"] for item in rows] == [
        item.variant_id for item in execution_protocol.variants
    ]
    by_variant = {item["variant"]["variant_id"]: item for item in rows}
    for variant_id in (
        "primary_v1",
        "wait_1m_v1",
        "wait_5m_v1",
        "cancel_price_terminal_only_v1",
        "target_1r_capped_dol_v1",
    ):
        assert {
            item["evaluation_status"]
            for item in by_variant[variant_id]["method_results"]
        } == {"evaluated"}
    assert {
        item["evaluation_status"]
        for item in by_variant["cancel_gtt_only_v1"]["method_results"]
    } == {"censored"}
    assert {
        item["evaluation_status"]
        for item in by_variant["stop_entry_zone_failure_v1"]["method_results"]
    } == {"censored"}

    summary = summarize_phase8_paired_rows(
        rows,
        runner_protocol=runner_protocol,
        execution_protocol=execution_protocol,
    )
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["case_count"] == 1
    assert len(summary["estimates"]) == 3 * 7 * 6


def test_side_aggregate_phase6_f_is_censored_not_guessed_as_price_fill() -> None:
    rows, _, _ = _evaluate(resting_fvg=True)
    primary = next(
        item for item in rows if item["variant"]["variant_id"] == "primary_v1"
    )
    fvg = next(
        item
        for item in primary["method_results"]
        if item["method"] == ExecutionMethod.FVG_50_LIMIT.value
    )
    assert fvg["evaluation_status"] == "censored"
    assert fvg["outcome"] is None
    assert fvg["censor_reasons"] == [
        "phase6_side_aggregate_passive_F_cannot_prove_method_price_fill"
    ]
    contrast = next(
        item
        for item in primary["paired_contrasts"]
        if item["method"] == ExecutionMethod.FVG_50_LIMIT.value
    )
    assert contrast["complete_pair"] is False


def test_paired_bootstrap_is_deterministic_and_requires_minimum_pairs() -> None:
    kwargs = {
        "replicates": 1000,
        "seed": 20240602,
        "confidence": 0.95,
        "minimum_pairs": 2,
    }
    first = paired_bootstrap_interval((1.0, 2.0, 3.0), **kwargs)
    assert first == paired_bootstrap_interval((1.0, 2.0, 3.0), **kwargs)
    assert first[0] is not None and first[1] is not None
    assert paired_bootstrap_interval((1.0,), **kwargs) == (None, None)


def test_output_writer_is_no_clobber_and_manifest_declares_last(tmp_path: Path) -> None:
    rows, runner_protocol, execution_protocol = _evaluate()
    summary = summarize_phase8_paired_rows(
        rows,
        runner_protocol=runner_protocol,
        execution_protocol=execution_protocol,
    )
    source_manifest = tmp_path / "frozen-run.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    results = tmp_path / "outputs/research/phase8-test"
    contract = Phase8RunManifestContract(
        source_path=source_manifest,
        manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        status="frozen_2024_06_development_research_not_oos_not_trading_authority",
        experiment_id="phase8-synthetic-write-test",
        frozen_at=pd.Timestamp("2024-05-31T00:00:00Z"),
        source_mode=SourceMode.REGISTERED_PHASE6,
        ledger_path=tmp_path / "intent-cases.jsonl",
        ledger_sha256="b" * 64,
        minute_artifact_path=None,
        minute_artifact_sha256=None,
        minute_manifest_path=None,
        minute_manifest_sha256=None,
        output_paths={
            "paired_rows": results / "paired.jsonl",
            "summary": results / "summary.json",
            "output_manifest": results / "manifest.json",
        },
        input_bindings={
            "intent_research_case_ledger": {
                "path": "inputs/intent-cases.jsonl",
                "sha256": "b" * 64,
            },
            "source_mode": SourceMode.REGISTERED_PHASE6.value,
            "minute_execution_artifact": {"path": None, "sha256": None},
            "minute_execution_manifest": {"path": None, "sha256": None},
        },
        payload={},
        blockers=(),
    )
    result = write_phase8_research_outputs(
        rows,
        summary,
        contract=contract,
        runner_protocol=runner_protocol,
        project_root=tmp_path,
    )
    manifest = json.loads(result.output_manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_written_last"] is True
    assert manifest["outputs"]["paired_rows"]["rows"] == 7
    with pytest.raises(Phase8RunnerError, match="cannot be overwritten"):
        write_phase8_research_outputs(
            rows,
            summary,
            contract=contract,
            runner_protocol=runner_protocol,
            project_root=tmp_path,
        )


def test_market_provenance_must_bind_exact_arrival_input() -> None:
    ledger, inputs = _ledger_and_inputs()
    intent_record, case = ledger.records
    market = case.method_price_set.methods[0]
    bad_market = replace(market, derivation_input_ids=("wrong-input",))
    bad_prices = replace(
        case.method_price_set,
        methods=(bad_market, *case.method_price_set.methods[1:]),
    )
    bad_case = replace(case, method_price_set=bad_prices)
    bad_ledger = Phase8AppendOnlyLedger((intent_record, bad_case))
    runner_protocol = load_execution_research_runner_protocol(RUNNER_CONFIG)
    execution_protocol = load_execution_research_v2_config(V2_CONFIG)
    risk_protocol = load_risk_admission_protocol(RISK_CONFIG)
    with pytest.raises(Phase8RunnerError, match="exact arrival input"):
        evaluate_phase8_research_cases(
            bad_ledger,
            inputs,
            runner_protocol=runner_protocol,
            execution_protocol=execution_protocol,
            execution_config=load_execution_research_config(V1_CONFIG),
            risk_protocol_id=risk_protocol.protocol_id,
            risk_protocol_sha256=risk_protocol.source_file_sha256,
        )
