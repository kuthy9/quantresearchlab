from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.execution_research import (
    EXECUTION_RESEARCH_PROTOCOL_SHA256,
    ORDERED_EXECUTION_METHODS,
    ExecutionMethod,
    load_execution_research_config,
)
from smc_trader.execution_research_v2 import (
    EXECUTION_RESEARCH_V2_CONFIG_SHA256,
    RISK_ADMISSION_PROTOCOL_SHA256,
    IntentLedgerRecord,
    MethodAvailability,
    MethodPriceProvenance,
    MethodPriceSet,
    Phase8AppendOnlyLedger,
    Phase8ContractError,
    RiskAdmissionProtocol,
    VariantDimension,
    build_executable_trade_instruction,
    build_research_case_record,
    evaluate_risk_admission,
    load_execution_research_v2_config,
    load_risk_admission_protocol,
    risk_admit_executable_trade_instruction,
    validate_phase8_runner_manifest,
)
from smc_trader.model import (
    AccountState,
    Direction,
    LiquidityLevel,
    StructuralLevel,
    Timeframe,
)
from smc_trader.signal_policy import (
    CancelCondition,
    CancelConditionKind,
    SetupFamily,
)
from smc_trader.trade_intent import EntryMethod, TimeInForce, TradeIntent


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG = ROOT / "configs/execution_research_v2.json"
RISK_CONFIG = ROOT / "configs/risk_admission_v1.json"
MANIFEST = ROOT / "experiments/manifests/execution_research_phase8_v2_template.yaml"
T0 = pd.Timestamp("2024-06-03T13:30:00Z")
EXPIRY = T0 + pd.Timedelta(hours=1)
SEMANTIC_SHA = "a" * 64


METHOD_RULES = {
    ExecutionMethod.MARKET: (
        "causal_arrival_bbo",
        "first_causal_valid_best_quote_ticks_v1",
    ),
    ExecutionMethod.FVG_50_LIMIT: (
        "fvg_generation",
        "fvg_midpoint_ticks_v1",
    ),
    ExecutionMethod.OB_50_LIMIT: (
        "qualified_order_block_generation",
        "qualified_ob_midpoint_ticks_v1",
    ),
    ExecutionMethod.RECLAIM_LIMIT: (
        "liquidity_interaction_generation",
        "first_outside_close_reclaim_ticks_v1",
    ),
    ExecutionMethod.BREAKOUT_LIMIT: (
        "structure_transition_generation",
        "first_confirming_outside_close_ticks_v1",
    ),
    ExecutionMethod.AGGRESSIVE_LIMIT: (
        "causal_arrival_bbo",
        "near_quote_one_tick_improvement_v1",
    ),
    ExecutionMethod.PASSIVE_LIMIT: (
        "causal_arrival_bbo",
        "far_quote_one_tick_improvement_v1",
    ),
}

METHOD_ROUNDING = {
    ExecutionMethod.MARKET: "reject_off_tick_no_rounding",
    ExecutionMethod.FVG_50_LIMIT: "nearest_tick_half_even_v1",
    ExecutionMethod.OB_50_LIMIT: "nearest_tick_half_even_v1",
    ExecutionMethod.RECLAIM_LIMIT: "reject_off_tick_no_rounding",
    ExecutionMethod.BREAKOUT_LIMIT: "reject_off_tick_no_rounding",
    ExecutionMethod.AGGRESSIVE_LIMIT: "exact_integer_tick_arithmetic",
    ExecutionMethod.PASSIVE_LIMIT: "exact_integer_tick_arithmetic",
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
        quantity=2,
        point_value=20.0,
        risk_budget_fraction=0.005,
        risk_budget_amount=500.0,
        position_risk_amount=200.0,
        entry_method_preferences=(
            EntryMethod.MARKET_ENTRY,
            EntryMethod.FVG_50_LIMIT,
        ),
        planned_entry=100.0,
        invalidation=invalidation,
        targets=(target,),
        trade_plan_id="trade-plan:test",
        max_wait_seconds=3600.0,
        time_in_force=TimeInForce.GOOD_TIL_TIME,
        cancel_conditions=(cancel,),
    )


def _account(**changes) -> AccountState:
    values = {
        "equity": 100_000.0,
        "open_risk_fraction": 0.0,
        "requested_risk_fraction": 0.005,
        "quantity": 2,
        "point_value": 20.0,
    }
    values.update(changes)
    return AccountState(**values)


def _method_price(
    method: ExecutionMethod,
    *,
    available: bool,
    price_ticks: int = 400,
    known_at: pd.Timestamp = T0,
) -> MethodPriceProvenance:
    semantic_type, derivation_id = METHOD_RULES[method]
    return MethodPriceProvenance(
        method=method,
        availability=(
            MethodAvailability.AVAILABLE
            if available
            else MethodAvailability.NOT_APPLICABLE
        ),
        price_ticks=price_ticks if available else None,
        tick_size=0.25,
        source_semantic_type=semantic_type,
        source_object_id=f"object:{method.value}" if available else None,
        source_generation_id=f"generation:{method.value}:1" if available else None,
        source_event_ids=(f"event:{method.value}:1",) if available else (),
        source_known_at=known_at if available else None,
        snapshot_asof=T0,
        derivation_id=derivation_id,
        derivation_input_ids=(f"input:{method.value}:1",) if available else (),
        rounding_rule=METHOD_ROUNDING[method] if available else "not_applicable",
        source_protocol_sha256=SEMANTIC_SHA,
        availability_reason=(
            "available_before_outcome"
            if available
            else "required_source_object_absent_at_snapshot"
        ),
    )


def _method_price_set(*, market_ticks: int = 400) -> MethodPriceSet:
    return MethodPriceSet(
        source_trade_intent_id=_intent().intent_id,
        snapshot_asof=T0,
        tick_size=0.25,
        methods=tuple(
            _method_price(
                method,
                available=method in {ExecutionMethod.MARKET, ExecutionMethod.FVG_50_LIMIT},
                price_ticks=market_ticks if method is ExecutionMethod.MARKET else 398,
            )
            for method in ORDERED_EXECUTION_METHODS
        ),
    )


def test_preregistered_config_has_one_primary_and_fixed_ofat_variants() -> None:
    protocol = load_execution_research_v2_config(
        EXECUTION_CONFIG,
        expected_sha256=EXECUTION_RESEARCH_V2_CONFIG_SHA256,
    )
    assert protocol.status == "preregistered_infrastructure_only_not_authorized_to_run"
    assert tuple(protocol.ordered_methods) == ORDERED_EXECUTION_METHODS
    assert [item.changed_dimension for item in protocol.variants] == [
        VariantDimension.PRIMARY,
        VariantDimension.WAIT,
        VariantDimension.WAIT,
        VariantDimension.CANCEL,
        VariantDimension.CANCEL,
        VariantDimension.STOP,
        VariantDimension.TARGET,
    ]
    assert protocol.variants[0].executable_instruction_eligible is True
    assert all(item.outcome_tuning_allowed is False for item in protocol.variants)
    assert all(
        item.executable_instruction_eligible is False
        for item in protocol.variants[1:]
    )
    assert all(
        not any(token in key for token in ("realized", "result", "metric_value"))
        for item in protocol.variants
        for key in item.to_payload()
    )


def test_method_prices_use_ticks_full_lineage_and_causal_availability() -> None:
    method_prices = _method_price_set()
    assert tuple(item.method for item in method_prices.methods) == ORDERED_EXECUTION_METHODS
    assert method_prices.for_method(ExecutionMethod.MARKET).price == 100.0
    unavailable = method_prices.for_method(ExecutionMethod.OB_50_LIMIT)
    assert unavailable.availability is MethodAvailability.NOT_APPLICABLE
    assert unavailable.price is None
    assert MethodPriceSet.from_payload(method_prices.to_payload()) == method_prices

    with pytest.raises(Phase8ContractError, match="future-known"):
        _method_price(
            ExecutionMethod.MARKET,
            available=True,
            known_at=T0 + pd.Timedelta(seconds=1),
        )
    with pytest.raises(Phase8ContractError, match="latent price"):
        replace(unavailable, price_ticks=400)

    censored = replace(
        _method_price(ExecutionMethod.OB_50_LIMIT, available=True),
        availability=MethodAvailability.CENSORED,
        price_ticks=None,
        availability_reason="source_zone_present_but_required_boundary_censored",
    )
    assert censored.price is None
    assert censored.source_generation_id == "generation:ob_50_limit:1"


def test_single_method_instruction_and_loaded_risk_protocol_are_fail_closed(
    tmp_path: Path,
) -> None:
    execution_protocol = load_execution_research_v2_config(EXECUTION_CONFIG)
    risk_protocol = load_risk_admission_protocol(
        RISK_CONFIG,
        expected_sha256=RISK_ADMISSION_PROTOCOL_SHA256,
    )
    intent = _intent()
    instruction = build_executable_trade_instruction(
        intent,
        _method_price_set(),
        protocol=execution_protocol,
        selection_event_id="event:method-selection:1",
    )
    assert instruction.method_price.method is ExecutionMethod.MARKET
    assert instruction.entry_price == 100.0
    assert instruction.stop_price == 95.0
    assert instruction.target_price == 110.0
    assert instruction.submission_allowed is False

    decision = evaluate_risk_admission(
        instruction,
        _account(),
        risk_protocol,
        assessed_at=T0,
    )
    assert decision.passed is True
    assert decision.risk_protocol_sha256 == RISK_ADMISSION_PROTOCOL_SHA256
    admitted = risk_admit_executable_trade_instruction(
        instruction,
        _account(),
        risk_protocol,
        assessed_at=T0,
    )
    assert admitted.admission == decision
    assert admitted.submission_allowed is False
    with pytest.raises(Phase8ContractError, match="does not conserve identity"):
        replace(admitted, admission=replace(decision))

    unsealed_copy = replace(instruction)
    with pytest.raises(TypeError, match="formally built"):
        evaluate_risk_admission(
            unsealed_copy,
            _account(),
            risk_protocol,
            assessed_at=T0,
        )
    with pytest.raises(Phase8ContractError, match="contract is invalid"):
        replace(instruction, execution_protocol_sha256="b" * 64)

    # The admission API has no fingerprint parameter and an un-loaded protocol
    # instance cannot authorize anything.
    with pytest.raises(TypeError, match="loader-authenticated"):
        evaluate_risk_admission(
            instruction,
            _account(),
            RiskAdmissionProtocol(),
            assessed_at=T0,
        )

    changed = json.loads(RISK_CONFIG.read_text(encoding="utf-8"))
    changed["limits"]["maximum_single_trade_risk_fraction"] = 0.5
    changed_path = tmp_path / "changed-risk.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(Phase8ContractError, match="bytes differ"):
        load_risk_admission_protocol(changed_path)

    high_entry = build_executable_trade_instruction(
        intent,
        _method_price_set(market_ticks=432),
        protocol=execution_protocol,
        selection_event_id="event:method-selection:2",
    )
    rejected = evaluate_risk_admission(
        high_entry,
        _account(),
        risk_protocol,
        assessed_at=T0,
    )
    assert rejected.passed is False
    assert "method_price_risk_exceeds_intent_budget" in rejected.reasons


def test_append_only_intent_and_research_case_jsonl_round_trip_is_idempotent() -> None:
    intent = _intent()
    intent_record = IntentLedgerRecord.from_trade_intent(intent)
    method_prices = _method_price_set()
    execution_protocol = load_execution_research_v2_config(EXECUTION_CONFIG)
    risk_protocol = load_risk_admission_protocol(RISK_CONFIG)
    research_case = build_research_case_record(
        intent_record,
        method_prices,
        execution_protocol,
        risk_protocol,
        source_artifact_ids=("instrument-map:NQM4", "semantic-registry:v1"),
    )
    ledger = Phase8AppendOnlyLedger()
    assert ledger.append(intent_record) is True
    assert ledger.append(intent_record) is False
    assert ledger.append(research_case) is True
    assert ledger.append(research_case) is False

    encoded = ledger.to_jsonl()
    restored = Phase8AppendOnlyLedger.from_jsonl(encoded)
    assert restored.records == ledger.records
    assert restored.to_jsonl() == encoded

    rows = encoded.splitlines()
    tampered = json.loads(rows[1])
    tampered["record_id"] = "phase8-research-case:tampered"
    with pytest.raises(Phase8ContractError, match="identity conflicts"):
        Phase8AppendOnlyLedger.from_jsonl(
            rows[0] + "\n" + json.dumps(tampered, sort_keys=True) + "\n"
        )


def test_runner_manifest_validation_is_inert_and_v1_1_remains_compatible(
    tmp_path: Path,
) -> None:
    result = validate_phase8_runner_manifest(MANIFEST)
    assert result.ready is False
    assert result.validate_only is True
    assert result.opened_dataset_bindings == ()
    assert result.written_artifacts == ()
    assert "formal_runner_not_implemented" in result.blockers

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["sealed_oos"]["path"] = "/path/that/must/not/be-opened"
    unsafe = tmp_path / "unsafe-manifest.json"
    unsafe.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase8ContractError, match="inert preregistration"):
        validate_phase8_runner_manifest(unsafe)

    legacy = load_execution_research_config(ROOT / "configs/execution_research_v1.json")
    assert legacy.config_id == f"execution-research-config:{EXECUTION_RESEARCH_PROTOCOL_SHA256}"
