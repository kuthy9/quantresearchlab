from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import smc_trader.event_store as event_store_module
import smc_trader.shadow_live as shadow_live_module

from smc_trader.engine import ContinuousSMCEngine
from smc_trader.event_store import EventStore
from smc_trader.foundation_adapter import CanonicalFoundationAdapter
from smc_trader.execution_fsm import (
    ExecutionFact,
    ExecutionFactKind,
    ExecutionFSM,
    OrderRole,
    OrderSide,
    OrderType,
    SubmitOrderCommand,
    account_state_fingerprint,
    make_execution_event,
    risk_approve_trade_intent,
)
from smc_trader.model import (
    AccountState,
    Bar,
    EventKind,
    EventOrigin,
    MarketEvent,
    Timeframe,
)
from smc_trader.foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
)
from smc_trader.observation import ExecutionRealityInput
from smc_trader.trade_intent import EntryMethod
from smc_trader.shadow_live import (
    NullExecutionGateway,
    SHADOW_COMPONENT_DIGEST_VERSION,
    SHADOW_LEGACY_COMPONENT_DIGEST_VERSION,
    ShadowClockInput,
    ShadowInputJournal,
    ShadowFailureRecord,
    ShadowLiveError,
    ShadowLiveRunner,
    ShadowParityAudit,
    audit_shadow_parity,
    load_shadow_live_protocol,
    replay_shadow_journal,
    shadow_runtime_bindings_from_model_config,
)

from .test_execution_fsm import T0, _account, _approved, _book


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/shadow_live_v1.json"
PROTOCOL_SHA256 = "10accaa0db0818e9785b61cd640edcb6f0d26ec9d5f909b728c50e67feca91f0"


def _bindings() -> tuple[tuple[str, str], ...]:
    return shadow_runtime_bindings_from_model_config(ROOT / "configs/model.json")


def _legacy_bindings() -> tuple[tuple[str, str], ...]:
    return tuple(
        item
        for item in _bindings()
        if item[0] != "shadow_component_digest_version"
    )


def _engine() -> ContinuousSMCEngine:
    return ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )


def test_shadow_runtime_bindings_freeze_foundation_registry() -> None:
    bindings = dict(_bindings())
    assert bindings["foundation_version"] == FOUNDATION_VERSION
    assert (
        bindings["foundation_registry_identity"]
        == FOUNDATION_CANONICAL_IDENTITY
    )

    engine = _engine()
    engine._foundation_registry_identity = "0" * 64
    with pytest.raises(ShadowLiveError, match="runtime bindings differ"):
        ShadowLiveRunner(
            engine=engine,
            protocol=load_shadow_live_protocol(PROTOCOL_PATH),
            runtime_bindings=bindings,
        )


def _shadow_approved():
    template = _approved()
    intent = replace(template.intent, symbol="NQM4")
    account = _account()
    approval = template.approval
    return risk_approve_trade_intent(
        intent,
        account,
        approved_at=T0,
        risk_protocol_id=approval.risk_protocol_id,
        risk_protocol_version=approval.risk_protocol_version,
        risk_protocol_fingerprint=approval.risk_protocol_fingerprint,
        account_snapshot_fingerprint=account_state_fingerprint(account),
    )


def _bar(index: int, *, base: pd.Timestamp | None = None) -> Bar:
    start = (base or pd.Timestamp("2024-06-03T13:30:00Z")) + pd.Timedelta(
        index, unit="m"
    )
    price = 18_500.0 + index * 0.25
    return Bar(
        start=start,
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price + 0.25,
        volume=100.0 + index,
        symbol="NQM4",
        instrument_id=13743,
    )


def _execution(bar: Bar) -> ExecutionRealityInput:
    return ExecutionRealityInput(
        spread_points=0.25,
        expected_slippage_points=0.0,
        commission_per_contract_per_side=2.25,
        quantity=1,
        deadline=bar.end + pd.Timedelta(hours=1),
        data_age_seconds=0.0,
        size_available=10.0,
        source="shadow_fixture_bbo",
        bid=bar.close,
        ask=bar.close + 0.25,
        bid_size=10.0,
        ask_size=10.0,
        depth_imbalance=0.0,
        anomalies=(),
    )


def _input(
    index: int,
    *,
    base: pd.Timestamp | None = None,
    feed_event_id: str | None = None,
    account: AccountState | None = None,
    account_snapshot_id: str | None = None,
    account_observed_at: pd.Timestamp | None = None,
    account_known_at: pd.Timestamp | None = None,
) -> ShadowClockInput:
    bar = _bar(index, base=base)
    execution_source_event_id = f"execution-reality:{bar.end.isoformat()}"
    account_snapshot_id = account_snapshot_id or (
        f"account-snapshot:{bar.end.isoformat()}"
    )
    return ShadowClockInput(
        feed_event_id=feed_event_id or f"feed:{bar.start.isoformat()}",
        received_at=bar.end + pd.Timedelta(milliseconds=5),
        bar=bar,
        execution=_execution(bar),
        execution_observed_at=bar.end,
        execution_known_at=bar.end,
        execution_source_event_id=execution_source_event_id,
        account=account or AccountState(equity=100_000.0),
        account_observed_at=account_observed_at or bar.end,
        account_known_at=account_known_at or bar.end,
        account_snapshot_id=account_snapshot_id,
        source_event_ids=(
            f"market-feed:{bar.start.isoformat()}",
            execution_source_event_id,
            account_snapshot_id,
        ),
    )


def test_protocol_null_gateway_and_input_journal_fail_closed() -> None:
    protocol = load_shadow_live_protocol(
        PROTOCOL_PATH,
        expected_sha256=PROTOCOL_SHA256,
    )
    assert protocol.external_submission_allowed is False
    assert protocol.authority == "null_gateway_no_external_submission"
    with pytest.raises(ShadowLiveError, match="preregistration"):
        replace(protocol, status="validated_live")

    gateway = NullExecutionGateway()
    with pytest.raises(ShadowLiveError, match="forbids"):
        gateway.submit({"order": "must-never-leave-process"})
    assert gateway.submission_attempts == 1

    journal = ShadowInputJournal()
    value = _input(0)
    assert journal.append(value)
    assert not journal.append(value)
    assert len(journal) == 1
    conflicting = replace(
        value,
        execution=replace(value.execution, expected_slippage_points=1.0),
    )
    with pytest.raises(ShadowLiveError, match="conflicts"):
        journal.append(conflicting)
    with pytest.raises(ShadowLiveError, match="clock order"):
        journal.append(_input(-1))

    later = _input(1)
    reused_execution = replace(
        later,
        execution_source_event_id=value.execution_source_event_id,
        source_event_ids=(
            f"market-feed:{later.bar.start.isoformat()}",
            value.execution_source_event_id,
            later.account_snapshot_id,
        ),
    )
    with pytest.raises(ShadowLiveError, match="execution evidence identity conflicts"):
        journal.append(reused_execution)
    reused_account = replace(
        later,
        account=replace(later.account, equity=123_456.0),
        account_snapshot_id=value.account_snapshot_id,
        source_event_ids=(
            f"market-feed:{later.bar.start.isoformat()}",
            later.execution_source_event_id,
            value.account_snapshot_id,
        ),
    )
    with pytest.raises(ShadowLiveError, match="account evidence identity conflicts"):
        journal.append(reused_account)


def test_shadow_runner_requires_exact_engine_and_null_gateway_types() -> None:
    class BypassGateway(NullExecutionGateway):
        def submit(self, _: object) -> None:
            return None

    class BypassEngine(ContinuousSMCEngine):
        pass

    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    bindings = _bindings()
    with pytest.raises(ShadowLiveError, match="only NullExecutionGateway"):
        ShadowLiveRunner(
            engine=_engine(),
            protocol=protocol,
            runtime_bindings=bindings,
            gateway=BypassGateway(),
        )
    with pytest.raises(TypeError, match="exact ContinuousSMCEngine"):
        ShadowLiveRunner(
            engine=BypassEngine.from_config(
                ROOT / "configs/model.json",
                runtime_mode="development",
            ),
            protocol=protocol,
            runtime_bindings=bindings,
        )

    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    with pytest.raises(AttributeError):
        runner.gateway.submit = lambda _: None  # type: ignore[method-assign]

    runner.gateway = BypassGateway()
    with pytest.raises(ShadowLiveError, match="indexes or approvals drifted"):
        runner.process(_input(0))

    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    runner.engine = BypassEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    with pytest.raises(ShadowLiveError, match="indexes or approvals drifted"):
        runner.process(_input(0))

    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    runner.engine.runtime_mode = "live"
    with pytest.raises(ShadowLiveError, match="indexes or approvals drifted"):
        runner.process(_input(0))


def test_shadow_contract_prices_must_use_the_frozen_tick_grid() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    bindings = _bindings()
    template = _approved()
    off_tick_intent = replace(
        template.intent,
        symbol="NQM4",
        planned_entry=100.1,
        position_risk_amount=204.0,
    )
    off_tick_approval = risk_approve_trade_intent(
        off_tick_intent,
        _account(),
        approved_at=T0,
        risk_protocol_id=template.approval.risk_protocol_id,
        risk_protocol_version=template.approval.risk_protocol_version,
        risk_protocol_fingerprint=template.approval.risk_protocol_fingerprint,
        account_snapshot_fingerprint=account_state_fingerprint(_account()),
    )
    off_tick_approval_input = replace(
        _input(
            0,
            base=T0,
            account=_account(),
            account_snapshot_id=off_tick_approval.approval.account_snapshot_id,
            account_observed_at=T0,
            account_known_at=T0,
        ),
        approved_intents=(off_tick_approval,),
    )
    with pytest.raises(ShadowLiveError, match="logical/vendor contract mapping"):
        ShadowLiveRunner(
            engine=_engine(), protocol=protocol, runtime_bindings=bindings
        ).process(off_tick_approval_input)

    approved = _shadow_approved()
    source_fsm = ExecutionFSM(approved)
    command = SubmitOrderCommand(
        command_id="command:phase9:off-tick",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("signal:phase9",),
        order_id="entry:phase9:off-tick",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        limit_price=100.0,
        entry_method=EntryMethod.FVG_50_LIMIT,
    )
    submitted = source_fsm.accept_command(
        command,
        known_at=command.created_at,
        vendor_sequence=1,
        top_of_book=_book(1),
    )
    acknowledged = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.ACKNOWLEDGED,
            approved_intent_id=approved.approved_intent_id,
            order_id=command.order_id,
        ),
        event_time=T0 + pd.Timedelta(seconds=2),
        known_at=T0 + pd.Timedelta(seconds=2),
        source_event_ids=("venue-message:phase9:ack",),
        vendor_sequence=2,
    )
    source_fsm.record(acknowledged)
    off_tick_fill = make_execution_event(
        ExecutionFact(
            kind=ExecutionFactKind.FILLED,
            approved_intent_id=approved.approved_intent_id,
            order_id=command.order_id,
            quantity=1,
            price=99.9,
        ),
        event_time=T0 + pd.Timedelta(seconds=3),
        known_at=T0 + pd.Timedelta(seconds=3),
        source_event_ids=("venue-message:phase9:off-tick",),
        vendor_sequence=3,
    )
    source_fsm.record(off_tick_fill)
    off_tick_event_input = replace(
        _input(
            0,
            base=T0,
            account=_account(),
            account_snapshot_id=approved.approval.account_snapshot_id,
            account_observed_at=T0,
            account_known_at=T0,
        ),
        approved_intents=(approved,),
        execution_events=(submitted, acknowledged, off_tick_fill),
    )
    with pytest.raises(ShadowLiveError, match="frozen tick grid"):
        ShadowLiveRunner(
            engine=_engine(), protocol=protocol, runtime_bindings=bindings
        ).process(off_tick_event_input)

def test_real_engine_shadow_stream_cold_replay_and_restart_are_exact() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    values = tuple(_input(index) for index in range(36))
    live = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )
    for value in values[:18]:
        live.process(value)
    resumed = pickle.loads(pickle.dumps(live))
    for value in values[18:]:
        resumed.process(value)

    cold = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )
    for value in values:
        cold.process(value)
    audit = audit_shadow_parity(resumed, cold)
    audit.require_exact()
    assert resumed.record_fingerprint == cold.record_fingerprint
    assert resumed.journal.fingerprint == cold.journal.fingerprint
    assert all(item.external_submission_attempts == 0 for item in resumed.records)

    replay = replay_shadow_journal(
        resumed.journal,
        engine_factory=_engine,
        protocol=protocol,
        runtime_bindings=_bindings(),
    )
    replay_audit = audit_shadow_parity(resumed, replay)
    replay_audit.require_exact()
    assert replay.record_fingerprint == resumed.record_fingerprint

    before = len(resumed.records)
    assert resumed.process(values[-1]) == resumed.records[-1]
    assert len(resumed.records) == before
    tampered = list(replay.records)
    tampered[-1] = replace(tampered[-1], risk_fingerprint="0" * 64)
    mismatch = audit_shadow_parity(resumed.records, tampered)
    assert not mismatch.exact_match
    assert "risk_fingerprint" in mismatch.mismatches[-1].fields
    assert mismatch.terminal_fields == ("runner_terminal_state_unavailable",)


def test_component_digest_versions_are_explicit_and_old_checkpoints_replay() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    value = _input(0)
    current = ShadowLiveRunner(
        engine=_engine(),
        protocol=protocol,
        runtime_bindings=_bindings(),
    )
    legacy = ShadowLiveRunner(
        engine=_engine(),
        protocol=protocol,
        runtime_bindings=_legacy_bindings(),
    )
    current_record = current.process(value)
    legacy_record = legacy.process(value)

    assert (
        current._component_digest_version  # noqa: SLF001 - contract audit
        == SHADOW_COMPONENT_DIGEST_VERSION
    )
    assert (
        legacy._component_digest_version  # noqa: SLF001 - legacy audit
        == SHADOW_LEGACY_COMPONENT_DIGEST_VERSION
    )
    assert current_record.protocol_id == legacy_record.protocol_id
    assert (
        current_record.runtime_bindings_fingerprint
        != legacy_record.runtime_bindings_fingerprint
    )
    assert (
        current_record.market_snapshot_fingerprint
        != legacy_record.market_snapshot_fingerprint
    )
    assert current.engine.last_snapshot == legacy.engine.last_snapshot
    assert current.engine.observer.audit_store.fingerprint() == (
        legacy.engine.observer.audit_store.fingerprint()
    )
    assert legacy_record.observation_fingerprint == shadow_live_module._digest(
        legacy.engine.last_snapshot.observation
    )
    with pytest.raises(ShadowLiveError, match="legacy component digest"):
        legacy.compact_runtime_checkpoint()
    technical_kinds = {
        EventKind.TIMEFRAME_STATE_CHANGED,
        EventKind.RELATION_STATE_CHANGED,
        EventKind.SESSION_STATE_CHANGED,
    }
    current_events = current.engine.observer.audit_store.events()
    technical_events = tuple(
        event for event in current_events if event.kind in technical_kinds
    )
    assert technical_events
    # Technical projections remain the complete canonical replay payload.
    # No plain or compact store can opt into a lossy commitment substitute.
    assert all(
        isinstance(event.details["projection_state"], Mapping)
        and "projection_sha256" not in event.details["projection_state"]
        for event in technical_events
    )
    assert all(event.details is event.evidence for event in current_events)
    assert not hasattr(event_store_module, "RebuildableProjectionCommitment")
    assert not hasattr(
        event_store_module,
        "REBUILDABLE_PROJECTION_COMMITMENT_VERSION",
    )
    with pytest.raises(TypeError):
        EventStore(rebuildable_projection_compaction=False)
    assert EventStore() == EventStore()

    state = dict(current.engine.observer.audit_store.__getstate__())
    assert set(state) == {
        "semantic_version",
        "_definition_identity",
        "_definition_identity_digest",
        "_events",
    }
    # Previous checkpoints serialized these rebuildable indexes alongside the
    # same canonical event sequence.  Current restore ignores and recomputes
    # them, so the protocol remains backward compatible.
    legacy_state = {
        **state,
        "_by_id": dict(current.engine.observer.audit_store._by_id),
        "_digests": dict(current.engine.observer.audit_store._digests),
        "_terminal_crossing_event_ids": dict(
            current.engine.observer.audit_store._terminal_crossing_event_ids
        ),
    }
    legacy_rebuilt = object.__new__(EventStore)
    legacy_rebuilt.__setstate__(legacy_state)
    assert legacy_rebuilt.events() == tuple(current_events)
    assert legacy_rebuilt.fingerprint() == (
        current.engine.observer.audit_store.fingerprint()
    )

    current_checkpoint = current.compact_runtime_checkpoint()
    assert current_checkpoint["schema_version"] == (
        "shadow_compact_runtime_v4"
    )
    previous_checkpoint = copy.deepcopy(current_checkpoint)
    previous_checkpoint["schema_version"] = "shadow_compact_runtime_v3"
    with pytest.raises(ShadowLiveError, match="legacy compact"):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            pickle.loads(pickle.dumps(previous_checkpoint)),
            journal_events=current.journal.events,
            records=current.records,
        )
    restored = ShadowLiveRunner.from_compact_runtime_checkpoint(
        pickle.loads(pickle.dumps(current_checkpoint)),
        journal_events=current.journal.events,
        records=current.records,
    )
    assert restored.process(_input(1)) == current.process(_input(1))

    restored_legacy = pickle.loads(pickle.dumps(legacy))
    assert restored_legacy.records == legacy.records
    assert (
        restored_legacy._component_digest_version  # noqa: SLF001
        == SHADOW_LEGACY_COMPONENT_DIGEST_VERSION
    )


def test_component_digest_recomputes_mutable_hypotheses_without_a_cache() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    runner.process(_input(0))
    belief = runner.engine.last_snapshot.belief
    assert belief.hypotheses
    before = runner._component_digest_bundle(  # noqa: SLF001
        runner.engine.last_snapshot
    ).as_record_fields()

    belief.hypotheses.pop(next(iter(belief.hypotheses)))
    after = runner._component_digest_bundle(  # noqa: SLF001
        runner.engine.last_snapshot
    ).as_record_fields()

    assert after["belief_fingerprint"] != before["belief_fingerprint"]
    assert (
        after["engine_snapshot_fingerprint"]
        != before["engine_snapshot_fingerprint"]
    )


def test_observation_event_bytes_must_match_the_same_audit_identity() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    runner.process(_input(0))
    runner.process(_input(1))
    snapshot = runner.engine.last_snapshot
    recent = snapshot.observation.recent_events
    assert recent
    target = recent[-1]
    tampered_event = replace(
        target,
        strength=1.0 if target.strength != 1.0 else 0.0,
    )
    assert tampered_event.event_id == target.event_id
    assert tampered_event.sequence_no == target.sequence_no
    assert tampered_event.strength != target.strength
    tampered_observation = replace(
        snapshot.observation,
        recent_events=(*recent[:-1], tampered_event),
    )
    tampered_snapshot = replace(
        snapshot,
        observation=tampered_observation,
    )

    original_fields = runner._component_digest_bundle(  # noqa: SLF001
        snapshot
    ).as_record_fields()
    tampered_fields = runner._component_digest_bundle(  # noqa: SLF001
        tampered_snapshot
    ).as_record_fields()
    assert tampered_fields["observation_fingerprint"] != (
        original_fields["observation_fingerprint"]
    )
    assert tampered_fields["engine_snapshot_fingerprint"] != (
        original_fields["engine_snapshot_fingerprint"]
    )

    runner.engine._last_snapshot = tampered_snapshot
    with pytest.raises(
        ShadowLiveError,
        match="event bytes differ from exact audit history",
    ):
        pickle.loads(pickle.dumps(runner))


def test_market_event_evidence_alias_requires_strict_primitive_equality() -> None:
    common = {
        "kind": EventKind.SWING_STATE,
        "observed_at": T0,
        "timeframe": Timeframe.M1,
        "side": None,
        "price": None,
        "strength": 0.0,
    }
    equal = MarketEvent(
        event_id="strict-evidence-equal",
        details={"flag": True, "nested": ("x", 1)},
        evidence={"flag": True, "nested": ("x", 1)},
        **common,
    )
    coercible_but_distinct = MarketEvent(
        event_id="strict-evidence-bool-int",
        details={"value": True},
        evidence={"value": 1},
        **common,
    )

    assert equal.details is equal.evidence
    assert coercible_but_distinct.details is not coercible_but_distinct.evidence
    assert type(coercible_but_distinct.details["value"]) is bool
    assert type(coercible_but_distinct.evidence["value"]) is int


def test_shadow_boundary_restore_requires_exact_real_bar() -> None:
    clock = pd.Timestamp("2024-06-03 10:00", tz="America/New_York")
    boundary = SimpleNamespace(timeframe=Timeframe.M1, known_at=clock)
    clock_only = MarketEvent(
        event_id="shadow-boundary-clock-only-bar",
        kind=EventKind.BAR_COMPLETED,
        observed_at=clock,
        timeframe=Timeframe.M1,
        side=None,
        price=100.0,
        strength=0.0,
        event_time=clock,
        known_at=clock,
        evidence={"real_completed": False, "clock_only": True},
        source_data_ids=("shadow-boundary-clock-only-data",),
        origin=EventOrigin.NORMALIZED_DATA,
    )

    with pytest.raises(ValueError, match="boundary attack real BAR"):
        shadow_live_module._require_shadow_boundary_attack_real_bar(
            clock_only,
            boundary,
        )

    real_evidence = {"real_completed": True, "clock_only": False}
    real = replace(
        clock_only,
        details=real_evidence,
        evidence=real_evidence,
    )
    shadow_live_module._require_shadow_boundary_attack_real_bar(real, boundary)


def test_compact_checkpoint_recomputes_terminal_engine_components() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    runner.process(_input(0))
    checkpoint = copy.deepcopy(runner.compact_runtime_checkpoint())
    belief = checkpoint["engine"].last_snapshot.belief
    assert belief.hypotheses
    belief.hypotheses.pop(next(iter(belief.hypotheses)))

    with pytest.raises(
        ShadowLiveError,
        match="terminal parity components differ",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            checkpoint,
            journal_events=runner.journal.events,
            records=runner.records,
        )


def test_restore_recomputes_every_historical_parity_record_id() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    for index in range(3):
        runner.process(_input(index))
    checkpoint = runner.compact_runtime_checkpoint()
    tampered_records = list(copy.deepcopy(runner.records))
    object.__setattr__(tampered_records[0], "risk_fingerprint", "0" * 64)
    with pytest.raises(
        ShadowLiveError,
        match="record content does not bind record_id",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            checkpoint,
            journal_events=runner.journal.events,
            records=tampered_records,
        )

    whole = pickle.loads(pickle.dumps(runner))
    first = whole._records[0]
    object.__setattr__(first, "risk_fingerprint", "0" * 64)
    whole._by_feed_id[first.feed_event_id] = first
    with pytest.raises(
        ShadowLiveError,
        match="record content does not bind record_id",
    ):
        pickle.loads(pickle.dumps(whole))


def test_restore_recomputes_every_historical_shadow_input_digest() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    for index in range(3):
        runner.process(_input(index))
    checkpoint = runner.compact_runtime_checkpoint()
    tampered_journal = list(copy.deepcopy(runner.journal.events))
    first = tampered_journal[0]
    object.__setattr__(
        first,
        "bar",
        replace(first.bar, volume=first.bar.volume + 999.0),
    )
    with pytest.raises(
        ShadowLiveError,
        match="input content does not bind input_digest",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            checkpoint,
            journal_events=tampered_journal,
            records=runner.records,
        )

    whole = pickle.loads(pickle.dumps(runner))
    first = whole.journal._events[0]
    object.__setattr__(
        first,
        "bar",
        replace(first.bar, volume=first.bar.volume + 999.0),
    )
    whole.journal._attempts[0] = first
    whole.journal._by_feed_id[first.feed_event_id] = first
    with pytest.raises(
        ShadowLiveError,
        match="input content does not bind input_digest",
    ):
        pickle.loads(pickle.dumps(whole))


def test_compact_restore_rehydrates_all_engine_derived_indexes() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    for index in range(36):
        runner.process(_input(index))
    checkpoint = runner.compact_runtime_checkpoint()
    checkpoint_journal = runner.journal.events
    checkpoint_records = runner.records

    event_tamper = copy.deepcopy(checkpoint)
    snapshot = event_tamper["engine"].last_snapshot
    retained_ids = {
        event.event_id
        for sequence in (
            snapshot.observation.recent_events,
            snapshot.observation.semantic_events_this_update,
            *snapshot.observation.retained_entity_timelines.values(),
            snapshot.market_snapshot.events_this_update,
        )
        for event in sequence
    }
    store = event_tamper["engine"].observer.audit_store
    target_index, target = next(
        (index, event)
        for index, event in enumerate(store._events)  # noqa: SLF001
        if event.event_id not in retained_ids
    )
    store._events[target_index] = replace(  # noqa: SLF001
        target,
        strength=1.0 if target.strength != 1.0 else 0.0,
    )
    with pytest.raises(
        ShadowLiveError,
        match="foundation source registry differs from audit history",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            event_tamper,
            journal_events=checkpoint_journal,
            records=checkpoint_records,
        )

    derived_tamper = copy.deepcopy(checkpoint)
    adapter = derived_tamper["engine"].observer._foundation_adapter
    assert adapter._real_bars  # noqa: SLF001
    adapter._real_bar_by_id.clear()  # noqa: SLF001
    adapter._lifecycle_indexes.clear()  # noqa: SLF001
    restored = ShadowLiveRunner.from_compact_runtime_checkpoint(
        derived_tamper,
        journal_events=checkpoint_journal,
        records=checkpoint_records,
    )
    restored_adapter = restored.engine.observer._foundation_adapter
    assert len(restored_adapter._real_bar_by_id) == len(  # noqa: SLF001
        restored_adapter._real_bars  # noqa: SLF001
    )
    assert restored_adapter._lifecycle_indexes  # noqa: SLF001
    assert restored.process(_input(36)) == runner.process(_input(36))

    empty_authority = copy.deepcopy(checkpoint)
    old_adapter = empty_authority["engine"].observer._foundation_adapter
    empty_authority["engine"].observer._foundation_adapter = type(old_adapter)(
        tick_size=old_adapter.tick_size
    )
    with pytest.raises(
        ShadowLiveError,
        match="foundation authority differs from published projection",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            empty_authority,
            journal_events=checkpoint_journal,
            records=checkpoint_records,
        )

    canonical_bar_tamper = copy.deepcopy(checkpoint)
    bar_adapter = canonical_bar_tamper["engine"].observer._foundation_adapter
    assert bar_adapter._real_bars  # noqa: SLF001
    bar_adapter._real_bars.clear()  # noqa: SLF001
    bar_adapter._real_bar_by_id.clear()  # noqa: SLF001
    canonical_bar_tamper["foundation_authority_digest"] = (
        bar_adapter.checkpoint().checkpoint_digest
    )
    with pytest.raises(
        ShadowLiveError,
        match="foundation .* differs",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            canonical_bar_tamper,
            journal_events=checkpoint_journal,
            records=checkpoint_records,
        )

    whole_runner_tamper = pickle.loads(pickle.dumps(runner))
    whole_adapter = whole_runner_tamper.engine.observer._foundation_adapter
    whole_adapter.lifecycle = type(whole_adapter.lifecycle)()
    with pytest.raises(
        ValueError,
        match="foundation adapter pickle lifecycle owner differs",
    ):
        pickle.loads(pickle.dumps(whole_runner_tamper))


def test_foundation_restore_rejects_self_consistent_runtime_authority_tamper() -> None:
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    previous_close = 18_500.0
    for index in range(120):
        value = _input(index)
        close = round(
            (18_500.0 + 20.0 * math.sin(index * 0.45)) * 4.0
        ) / 4.0
        bar = Bar(
            start=value.bar.start,
            open=previous_close,
            high=max(previous_close, close) + 2.0,
            low=min(previous_close, close) - 2.0,
            close=close,
            volume=100.0 + index,
            symbol="NQM4",
            instrument_id=13_743,
        )
        runner.process(
            replace(value, bar=bar, execution=_execution(bar))
        )
        previous_close = close
    checkpoint = runner.compact_runtime_checkpoint()
    journal = runner.journal.events
    records = runner.records

    applied_tamper = copy.deepcopy(checkpoint)
    adapter = applied_tamper["engine"].observer._foundation_adapter
    manual_id = next(
        fact_id
        for fact_id in adapter.lifecycle_fact_fingerprints
        if fact_id.startswith("observer-boundary-attack:")
    )
    adapter._lifecycle_owner._fact_fingerprints[manual_id] = "0" * 64
    with pytest.raises(ValueError, match="lifecycle owner fact chain differs"):
        adapter.checkpoint()

    asof_tamper = copy.deepcopy(checkpoint)
    adapter = asof_tamper["engine"].observer._foundation_adapter
    adapter.lifecycle = replace(
        adapter.lifecycle,
        asof=pd.Timestamp("2099-01-01T00:00:00Z"),
    )
    adapter._lifecycle_owner._state = adapter.lifecycle
    asof_tamper["foundation_authority_digest"] = (
        adapter.checkpoint().checkpoint_digest
    )
    with pytest.raises(
        ShadowLiveError,
        match="foundation authority differs from audit replay",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            asof_tamper,
            journal_events=journal,
            records=records,
        )

    tick_tamper = copy.deepcopy(checkpoint)
    observer = tick_tamper["engine"].observer
    adapter = observer._foundation_adapter
    adapter.tick_size = 0.125
    authoritative = tuple(
        sorted(
            (
                event
                for event in observer.audit_store.events()
                if event.origin
                in {
                    EventOrigin.NORMALIZED_DATA,
                    EventOrigin.SEMANTIC_ATOMIC,
                }
            ),
            key=lambda event: (
                event.known_at,
                event.sequence_no,
                event.event_id,
            ),
        )
    )
    false_replay = CanonicalFoundationAdapter(tick_size=0.125)
    false_replay.consume_batch(authoritative)
    adapter._real_bars = false_replay._real_bars
    adapter._real_bar_by_id = false_replay._real_bar_by_id
    adapter._crossings = false_replay._crossings
    adapter._structure_bindings = false_replay._structure_bindings
    tick_tamper["foundation_authority_digest"] = (
        adapter.checkpoint().checkpoint_digest
    )
    with pytest.raises(
        ShadowLiveError,
        match="tick size differs from runtime authority",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            tick_tamper,
            journal_events=journal,
            records=records,
        )

    whole_tick_tamper = pickle.loads(pickle.dumps(runner))
    whole_tick_tamper.engine.observer._foundation_adapter.tick_size = 0.125
    with pytest.raises(
        ShadowLiveError,
        match="tick size differs from runtime authority",
    ):
        pickle.loads(pickle.dumps(whole_tick_tamper))

    duplicate_tamper = copy.deepcopy(checkpoint)
    adapter = duplicate_tamper["engine"].observer._foundation_adapter
    assert adapter.lifecycle.levels
    object.__setattr__(
        adapter.lifecycle,
        "levels",
        (*adapter.lifecycle.levels, adapter.lifecycle.levels[-1]),
    )
    adapter._rebuild_derived_indexes()
    with pytest.raises(ValueError, match="lifecycle owner differs"):
        adapter.checkpoint()

    order_tamper = copy.deepcopy(checkpoint)
    adapter = order_tamper["engine"].observer._foundation_adapter
    assert len(adapter.lifecycle.levels) > 1
    adapter.lifecycle = replace(
        adapter.lifecycle,
        levels=tuple(reversed(adapter.lifecycle.levels)),
    )
    adapter._lifecycle_owner._state = adapter.lifecycle
    adapter._rebuild_derived_indexes()
    order_tamper["foundation_authority_digest"] = (
        adapter.checkpoint().checkpoint_digest
    )
    with pytest.raises(
        ShadowLiveError,
        match="foundation lifecycle differs from audit replay",
    ):
        ShadowLiveRunner.from_compact_runtime_checkpoint(
            order_tamper,
            journal_events=journal,
            records=records,
        )

    whole_duplicate_tamper = pickle.loads(pickle.dumps(runner))
    adapter = whole_duplicate_tamper.engine.observer._foundation_adapter
    object.__setattr__(
        adapter.lifecycle,
        "levels",
        (*adapter.lifecycle.levels, adapter.lifecycle.levels[-1]),
    )
    adapter._rebuild_derived_indexes()
    with pytest.raises(
        ValueError,
        match="foundation adapter pickle lifecycle owner differs",
    ):
        pickle.loads(pickle.dumps(whole_duplicate_tamper))


def test_component_digest_final_audit_replays_full_market_payload() -> None:
    valid = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    for index in range(36):
        valid.process(_input(index))
    tampered = pickle.loads(pickle.dumps(valid))
    projection = tampered.engine.last_snapshot.market_snapshot.foundation
    assert projection is not None
    object.__setattr__(
        projection.current_records[-1],
        "record_id",
        "foundation-record:" + "0" * 64,
    )

    audit = audit_shadow_parity(valid, tampered)
    assert not audit.gate_pass
    assert any(
        value.endswith("market_full_replay_payload")
        for value in audit.terminal_fields
    )


@pytest.mark.historical_frozen
def test_real_w1_manual_foundation_transitions_restore_and_continue_exactly() -> None:
    from itertools import islice

    from scripts.run_shadow_file_pilot import iter_shadow_clock_file

    values = tuple(
        islice(
            iter_shadow_clock_file(
                ROOT / "inputs/phase9_w1_foundation_v3_7465a04.jsonl"
            ),
            101,
        )
    )
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    for value in values[:100]:
        runner.process(value)

    whole = pickle.loads(pickle.dumps(runner))
    assert whole.engine.last_snapshot == runner.engine.last_snapshot
    journal = runner.journal.events
    records = runner.records
    checkpoint = runner.compact_runtime_checkpoint()
    restored = ShadowLiveRunner.from_compact_runtime_checkpoint(
        pickle.loads(pickle.dumps(checkpoint)),
        journal_events=journal,
        records=records,
    )
    assert restored.process(values[100]) == runner.process(values[100])
    assert restored.record_fingerprint == runner.record_fingerprint
    assert restored.journal.fingerprint == runner.journal.fingerprint
    assert restored.gateway.submission_attempts == 0

    for prefix in ("relation-observation:", "delivery-observation:"):
        tampered = copy.deepcopy(checkpoint)
        adapter = tampered["engine"].observer._foundation_adapter
        target_id = next(
            fact_id
            for fact_id in adapter.lifecycle_fact_fingerprints
            if fact_id.startswith(prefix)
        )
        adapter._lifecycle_owner._fact_fingerprints[target_id] = "0" * 64
        adapter._rebuild_derived_indexes()
        tampered["foundation_authority_digest"] = (
            adapter.checkpoint().checkpoint_digest
        )
        with pytest.raises(
            ShadowLiveError,
            match="foundation .* differs",
        ):
            ShadowLiveRunner.from_compact_runtime_checkpoint(
                tampered,
                journal_events=journal,
                records=records,
            )


def test_execution_fsm_events_are_part_of_the_same_clock_parity_record() -> None:
    approved = _shadow_approved()
    source_fsm = ExecutionFSM(approved)
    command = SubmitOrderCommand(
        command_id="command:phase9:submit",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("signal:phase9",),
        order_id="entry:phase9",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        limit_price=100.0,
        entry_method=EntryMethod.FVG_50_LIMIT,
    )
    submitted = source_fsm.accept_command(
        command,
        known_at=command.created_at,
        vendor_sequence=1,
        top_of_book=_book(1),
    )
    bar = _bar(0, base=T0)
    execution_source_event_id = "execution-reality:phase9"
    account_snapshot_id = approved.approval.account_snapshot_id
    value = ShadowClockInput(
        feed_event_id="feed:phase9:execution",
        received_at=bar.end,
        bar=bar,
        execution=_execution(bar),
        execution_observed_at=bar.end,
        execution_known_at=bar.end,
        execution_source_event_id=execution_source_event_id,
        account=_account(),
        account_observed_at=T0,
        account_known_at=T0,
        account_snapshot_id=account_snapshot_id,
        source_event_ids=(
            "market-feed:phase9",
            execution_source_event_id,
            account_snapshot_id,
        ),
        approved_intents=(approved,),
        execution_events=(submitted,),
    )
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    live = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )
    record = live.process(value)
    assert approved.approved_intent_id in live.execution_fsms
    assert len(live.execution_fsms[approved.approved_intent_id].store) == 1
    assert record.execution_event_fingerprint != record.execution_state_fingerprint

    replay = replay_shadow_journal(
        live.journal,
        engine_factory=_engine,
        protocol=protocol,
        runtime_bindings=_bindings(),
    )
    audit_shadow_parity(live, replay).require_exact()


def test_shadow_input_rejects_future_execution_fact_and_wrong_account_approval() -> None:
    approved = _shadow_approved()
    source_fsm = ExecutionFSM(approved)
    command = SubmitOrderCommand(
        command_id="command:phase9:future",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("signal:phase9",),
        order_id="entry:phase9:future",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        limit_price=100.0,
        entry_method=EntryMethod.FVG_50_LIMIT,
    )
    submitted = source_fsm.accept_command(
        command,
        known_at=command.created_at,
        vendor_sequence=1,
        top_of_book=_book(1),
    )
    early_bar = _bar(0, base=T0 - pd.Timedelta(minutes=1))
    with pytest.raises(ShadowLiveError, match="future-known"):
        ShadowClockInput(
            feed_event_id="feed:phase9:future",
            received_at=early_bar.end,
            bar=early_bar,
            execution=_execution(early_bar),
            execution_observed_at=early_bar.end,
            execution_known_at=early_bar.end,
            execution_source_event_id="execution-reality:future",
            account=_account(),
            account_observed_at=early_bar.end,
            account_known_at=early_bar.end,
            account_snapshot_id=approved.approval.account_snapshot_id,
            source_event_ids=(
                "market-feed:phase9",
                "execution-reality:future",
                approved.approval.account_snapshot_id,
            ),
            approved_intents=(approved,),
            execution_events=(submitted,),
        )

    bar = _bar(0, base=T0)
    execution_source_event_id = "execution-reality:wrong-account"
    account_snapshot_id = approved.approval.account_snapshot_id
    value = ShadowClockInput(
        feed_event_id="feed:phase9:wrong-account",
        received_at=bar.end,
        bar=bar,
        execution=_execution(bar),
        execution_observed_at=bar.end,
        execution_known_at=bar.end,
        execution_source_event_id=execution_source_event_id,
        account=_account(equity=90_000.0),
        account_observed_at=T0,
        account_known_at=T0,
        account_snapshot_id=account_snapshot_id,
        source_event_ids=(
            "market-feed:phase9",
            execution_source_event_id,
            account_snapshot_id,
        ),
        approved_intents=(approved,),
    )
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(PROTOCOL_PATH),
        runtime_bindings=_bindings(),
    )
    with pytest.raises(ShadowLiveError, match="another account"):
        runner.process(value)
    assert runner.failure is not None
    assert len(runner.journal) == 1
    assert not runner.records
    with pytest.raises(ShadowLiveError, match="terminal"):
        runner.process(_input(1, base=T0))
    replay = replay_shadow_journal(
        runner.journal,
        engine_factory=_engine,
        protocol=runner.protocol,
        runtime_bindings=_bindings(),
    )
    assert replay.failure is not None
    assert replay.failure.failure_id == runner.failure.failure_id
    failed_audit = audit_shadow_parity(runner, replay)
    assert failed_audit.exact_match
    assert not failed_audit.coverage_complete
    assert not failed_audit.gate_pass
    with pytest.raises(ShadowLiveError, match="healthy non-empty"):
        failed_audit.require_exact()


def test_runtime_action_policy_and_model_bytes_are_part_of_parity_identity() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    bindings = _bindings()
    standard = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    filtered_engine = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
        action_disabled_playbooks=("liquidity_sweep_reversal",),
    )
    filtered = ShadowLiveRunner(
        engine=filtered_engine,
        protocol=protocol,
        runtime_bindings=bindings,
    )
    value = _input(0)
    standard_record = standard.process(value)
    filtered_record = filtered.process(value)
    assert (
        standard_record.runtime_action_policy_fingerprint
        != filtered_record.runtime_action_policy_fingerprint
    )
    mismatch = audit_shadow_parity(standard, filtered)
    assert not mismatch.exact_match
    assert "runtime_action_policy_fingerprint" in mismatch.mismatches[0].fields

    bad_bindings = dict(bindings)
    bad_bindings["model_config_sha256"] = "0" * 64
    with pytest.raises(ShadowLiveError, match="runtime bindings differ"):
        ShadowLiveRunner(
            engine=_engine(),
            protocol=protocol,
            runtime_bindings=bad_bindings,
        )


def test_shadow_evidence_clocks_age_and_whole_contracts_fail_closed() -> None:
    value = _input(0)
    with pytest.raises(ShadowLiveError, match="clock-inverted"):
        replace(
            value,
            execution_observed_at=value.bar.end + pd.Timedelta(seconds=1),
        )
    with pytest.raises(ShadowLiveError, match="data age"):
        replace(
            value,
            execution_observed_at=value.bar.end - pd.Timedelta(seconds=1),
        )
    with pytest.raises(ShadowLiveError, match="whole contracts"):
        replace(value, execution=replace(value.execution, quantity=0.5))
    with pytest.raises(ShadowLiveError, match="whole contracts"):
        replace(value, account=replace(value.account, quantity=0.5))
    with pytest.raises(ShadowLiveError, match="displayed execution capacity"):
        replace(value, execution=replace(value.execution, ask_size=0.5))
    with pytest.raises(ShadowLiveError, match="redundant fields"):
        replace(value, execution=replace(value.execution, spread_points=10.0))
    with pytest.raises(ShadowLiveError, match="redundant fields"):
        replace(value, execution=replace(value.execution, size_available=9.0))
    with pytest.raises(ShadowLiveError, match="omit exact"):
        replace(
            value,
            source_event_ids=(
                value.execution_source_event_id,
                "market-feed:missing-account",
            ),
        )


def test_failure_and_gateway_terminal_state_are_part_of_parity() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    bindings = _bindings()
    healthy = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    failed = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    approved = _shadow_approved()
    value = _input(
        0,
        account=_account(equity=90_000.0),
        base=T0,
        account_snapshot_id=approved.approval.account_snapshot_id,
        account_observed_at=T0,
        account_known_at=T0,
    )
    approved_value = replace(value, approved_intents=(approved,))
    with pytest.raises(ShadowLiveError, match="another account"):
        failed.process(approved_value)
    audit = audit_shadow_parity(healthy, failed)
    assert not audit.exact_match
    assert "journal_fingerprint" in audit.terminal_fields
    assert "failure_id" in audit.terminal_fields

    live = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    accepted = _input(0)
    live.process(accepted)
    with pytest.raises(ShadowLiveError, match="forbids"):
        live.gateway.submit({"must": "never leave process"})
    with pytest.raises(ShadowLiveError, match="attempted"):
        live.process(accepted)
    assert live.failure is not None
    replay = replay_shadow_journal(
        live.journal,
        engine_factory=_engine,
        protocol=protocol,
        runtime_bindings=bindings,
    )
    terminal_audit = audit_shadow_parity(live, replay)
    assert not terminal_audit.exact_match
    assert "failure_id" in terminal_audit.terminal_fields
    assert "external_submission_attempts" in terminal_audit.terminal_fields

    first_empty = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    second_empty = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    empty_audit = audit_shadow_parity(first_empty, second_empty)
    assert empty_audit.exact_match
    assert not empty_audit.coverage_complete
    assert not empty_audit.gate_pass


def test_all_processing_errors_produce_stable_terminal_failure_records() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )
    runner.engine.on_bar = lambda *args, **kwargs: None  # type: ignore[method-assign]
    with pytest.raises(ShadowLiveError, match="did not publish"):
        runner.process(_input(0))
    assert runner.failure is not None

    empty = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )

    def _raise_empty(*args, **kwargs):
        raise RuntimeError()

    empty.engine.on_bar = _raise_empty  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        empty.process(_input(0))
    assert empty.failure is not None
    assert empty.failure.error_message == "exception_without_message"

    conflict = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )
    accepted = _input(0)
    conflict.process(accepted)
    conflicting = replace(
        accepted,
        execution=replace(accepted.execution, expected_slippage_points=1.0),
    )
    with pytest.raises(ShadowLiveError, match="conflicts"):
        conflict.process(conflicting)
    replayed_conflict = replay_shadow_journal(
        conflict.journal,
        engine_factory=_engine,
        protocol=protocol,
        runtime_bindings=_bindings(),
    )
    assert replayed_conflict.failure is not None
    assert replayed_conflict.failure.failure_id == conflict.failure.failure_id
    conflict_audit = audit_shadow_parity(conflict, replayed_conflict)
    assert conflict_audit.exact_match
    assert not conflict_audit.gate_pass


def test_execution_facts_must_arrive_in_their_exact_completed_clock_batch() -> None:
    approved = _shadow_approved()
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=_bindings()
    )
    first = replace(
        _input(
            0,
            base=T0,
            account=_account(),
            account_snapshot_id=approved.approval.account_snapshot_id,
            account_observed_at=T0,
            account_known_at=T0,
        ),
        approved_intents=(approved,),
    )
    runner.process(first)

    source_fsm = ExecutionFSM(approved)
    command = SubmitOrderCommand(
        command_id="command:phase9:late-delivery",
        created_at=T0 + pd.Timedelta(seconds=1),
        approved_intent_id=approved.approved_intent_id,
        source_event_ids=("signal:phase9",),
        order_id="entry:phase9:late-delivery",
        role=OrderRole.ENTRY,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1,
        limit_price=100.0,
        entry_method=EntryMethod.FVG_50_LIMIT,
    )
    submitted = source_fsm.accept_command(
        command,
        known_at=command.created_at,
        vendor_sequence=1,
        top_of_book=_book(1),
    )
    second = replace(
        _input(1, base=T0, account=_account()),
        execution_events=(submitted,),
    )
    with pytest.raises(ShadowLiveError, match="causal completed-clock batch"):
        runner.process(second)
    assert runner.failure is not None


def test_runtime_binding_duplicates_and_checkpoint_code_drift_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    bindings = _bindings()
    with pytest.raises(ShadowLiveError, match="duplicated"):
        ShadowLiveRunner(
            engine=_engine(),
            protocol=protocol,
            runtime_bindings=(*bindings, bindings[0]),
        )

    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    runner.process(_input(0))
    checkpoint = pickle.dumps(runner)
    monkeypatch.setattr(
        shadow_live_module,
        "SMC_SEMANTIC_VERSION",
        "simulated-semantic-v999",
    )
    with pytest.raises(ShadowLiveError, match="drifted after restore"):
        pickle.loads(checkpoint)


def test_contract_mapping_account_identity_and_audit_tamper_fail_closed() -> None:
    protocol = load_shadow_live_protocol(PROTOCOL_PATH)
    bindings = _bindings()
    wrong_contract = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    wrong_bar = replace(_input(0).bar, symbol="ESM4", instrument_id=999)
    wrong_input = replace(
        _input(0),
        bar=wrong_bar,
        received_at=wrong_bar.end,
        execution=_execution(wrong_bar),
        execution_observed_at=wrong_bar.end,
        execution_known_at=wrong_bar.end,
        account_observed_at=wrong_bar.end,
        account_known_at=wrong_bar.end,
    )
    with pytest.raises(ShadowLiveError, match="vendor instrument mapping"):
        wrong_contract.process(wrong_input)

    approved = _shadow_approved()
    wrong_account_identity = replace(
        _input(
            0,
            base=T0,
            account=_account(),
            account_observed_at=T0,
            account_known_at=T0,
        ),
        approved_intents=(approved,),
    )
    runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    with pytest.raises(ShadowLiveError, match="another account"):
        runner.process(wrong_account_identity)

    future_known_account = replace(
        _input(
            0,
            base=T0,
            account=_account(),
            account_snapshot_id=approved.approval.account_snapshot_id,
        ),
        approved_intents=(approved,),
    )
    future_account_runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    with pytest.raises(ShadowLiveError, match="another account"):
        future_account_runner.process(future_known_account)

    wrong_point_value = replace(
        _input(0),
        account=replace(_input(0).account, point_value=10.0),
    )
    point_value_runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    with pytest.raises(ShadowLiveError, match="vendor instrument mapping"):
        point_value_runner.process(wrong_point_value)

    legacy_symbol = _approved()
    wrong_mapping = replace(
        _input(
            0,
            base=T0,
            account=_account(),
            account_snapshot_id=legacy_symbol.approval.account_snapshot_id,
        ),
        approved_intents=(legacy_symbol,),
    )
    mapping_runner = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    with pytest.raises(ShadowLiveError, match="logical/vendor contract mapping"):
        mapping_runner.process(wrong_mapping)

    valid = ShadowLiveRunner(
        engine=_engine(), protocol=protocol, runtime_bindings=bindings
    )
    record = valid.process(_input(0))
    alias_audit = audit_shadow_parity(valid, valid)
    assert alias_audit.exact_match is False
    assert alias_audit.gate_pass is False
    assert "non_independent_runner_alias" in alias_audit.terminal_fields
    with pytest.raises(ShadowLiveError, match="lowercase SHA"):
        replace(record, risk_fingerprint="z" * 64)
    with pytest.raises(ShadowLiveError, match="identity namespace"):
        replace(record, protocol_id="bogus")
    contextual_tamper = pickle.loads(pickle.dumps(valid))
    contextual_tamper._records[0] = replace(  # noqa: SLF001 - adversarial audit
        contextual_tamper.records[0],
        sequence=99,
    )
    contextual_audit = audit_shadow_parity(contextual_tamper, contextual_tamper)
    assert not contextual_audit.exact_match
    assert any(
        value.endswith("record_sequence")
        for value in contextual_audit.terminal_fields
    )
    journal_tamper = pickle.loads(pickle.dumps(valid))
    journal_tamper.journal._by_feed_id.clear()  # noqa: SLF001 - adversarial audit
    journal_audit = audit_shadow_parity(valid, journal_tamper)
    assert not journal_audit.gate_pass
    assert any(
        value.endswith("journal_or_runtime_consistency")
        for value in journal_audit.terminal_fields
    )
    with pytest.raises(ShadowLiveError, match="indexes"):
        pickle.loads(pickle.dumps(journal_tamper))

    runner_index_tamper = pickle.loads(pickle.dumps(valid))
    runner_index_tamper._by_feed_id.clear()  # noqa: SLF001 - adversarial audit
    runner_index_audit = audit_shadow_parity(valid, runner_index_tamper)
    assert not runner_index_audit.gate_pass
    assert any(
        value.endswith("journal_or_runtime_consistency")
        for value in runner_index_audit.terminal_fields
    )
    with pytest.raises(ShadowLiveError, match="runner indexes"):
        pickle.loads(pickle.dumps(runner_index_tamper))

    engine_tamper = pickle.loads(pickle.dumps(valid))
    engine_tamper.engine._last_snapshot = None  # noqa: SLF001 - adversarial audit
    engine_audit = audit_shadow_parity(valid, engine_tamper)
    assert not engine_audit.gate_pass
    assert any(
        value.endswith("engine_last_snapshot_missing")
        for value in engine_audit.terminal_fields
    )
    with pytest.raises(ShadowLiveError, match="terminal state drifted"):
        pickle.loads(pickle.dumps(engine_tamper))
    with pytest.raises(ShadowLiveError, match="status is inconsistent"):
        ShadowParityAudit(
            expected_records=-1,
            actual_records=-1,
            mismatches=(),
            terminal_fields=(),
            exact_match=True,
            coverage_complete=True,
            gate_pass=True,
        )
    with pytest.raises(ShadowLiveError, match="lowercase SHA"):
        ShadowFailureRecord(
            sequence=1,
            feed_event_id="feed:tampered",
            input_digest="z" * 64,
            journal_fingerprint="0" * 64,
            attempt_fingerprint="0" * 64,
            last_record_id=None,
            error_type="RuntimeError",
            error_message="tampered",
        )
