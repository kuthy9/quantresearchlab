from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_shadow_file_pilot import (
    shadow_clock_input_from_payload,
    shadow_clock_input_payload,
)
from scripts.run_shadow_file_pilot_v3 import _checkpoint, _restore
from smc_trader.calibration_replay import ReplayCheckpointStore
from smc_trader.shadow_operational import (
    CAPACITY_AUTHORIZATION_SCHEMA_VERSION,
    PHASE9_WEEK1_BUNDLE_SCHEMA_VERSION,
    CapacityAuthorizationError,
    DurableShadowWAL,
    InstrumentMappingBinding,
    OperationalShadowSession,
    Phase9OperationalError,
    ShadowBackfillRequired,
    build_phase9_bundle_v3_payload,
    build_week1_bundle_v3_payload,
    capacity_authorization_id,
    load_capacity_authorization,
    load_instrument_mapping,
    load_shadow_operational_protocol,
    phase9_historical_window,
    runtime_code_environment_identity,
    validate_phase9_bundle_v3_payload,
    validate_week1_bundle_v3_payload,
    expected_missing_completed_clocks,
)
from smc_trader.shadow_live import (
    ShadowInputJournal,
    ShadowLiveRunner,
    load_shadow_live_protocol,
    shadow_runtime_bindings_from_model_config,
)

from .test_shadow_live_parity import _engine, _input


pytestmark = pytest.mark.research_runner


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_PROTOCOL = ROOT / "configs/phase9_shadow_operational_v1.json"
MONDAY = pd.Timestamp("2024-06-03T13:30:00Z")


def _mapping(*, mode: str = "historical_simulation") -> InstrumentMappingBinding:
    return InstrumentMappingBinding(
        mapping_id="shadow-instrument-mapping:test-v1",
        mapping_version="test-v1",
        mode=mode,
        logical_instrument_id="NQ:front",
        vendor_symbol="NQM4",
        vendor_instrument_id=13743,
        tick_size=0.25,
        point_value=20.0,
        effective_from=pd.Timestamp("2024-06-01T00:00:00Z"),
        effective_until=pd.Timestamp("2024-07-01T00:00:00Z"),
        source_identity="0" * 64,
    )


def _wal(path: Path) -> DurableShadowWAL:
    return DurableShadowWAL(
        path,
        encode_input=shadow_clock_input_payload,
        decode_input=shadow_clock_input_from_payload,
    )


def test_operational_protocol_is_engineering_only_and_mapping_is_versioned() -> None:
    protocol = load_shadow_operational_protocol(OPERATIONAL_PROTOCOL)
    assert protocol.historical_simulation_only is True
    assert protocol.real_time_live_claim_allowed is False
    assert protocol.external_capacity_authorization_required is True
    assert protocol.external_submission_allowed is False
    assert protocol.recovery_point_objective_accepted_clocks == 0

    value = _input(0, base=MONDAY)
    _mapping().require_input(value, mode="historical_simulation")
    with pytest.raises(Phase9OperationalError, match="mapping mode"):
        _mapping().require_input(value, mode="current_live_shadow")
    with pytest.raises(Phase9OperationalError, match="effective interval"):
        _mapping().require_input(
            _input(0, base=pd.Timestamp("2026-08-20T13:30:00Z")),
            mode="historical_simulation",
        )
    template = ROOT / "configs/phase9_current_contract_mapping_v1.template.json"
    with pytest.raises(Phase9OperationalError, match="not frozen"):
        load_instrument_mapping(
            template,
            expected_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
            required_mode="current_live_shadow",
        )


def test_incremental_v2_fingerprints_remain_byte_exact_and_pickle_safe() -> None:
    journal = ShadowInputJournal()
    values = tuple(_input(index, base=MONDAY) for index in range(4))
    for value in values:
        journal.record_attempt(value)
        assert journal.append(value)
    canonical = json.dumps(
        [value.input_digest for value in values],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert journal.fingerprint == hashlib.sha256(canonical).hexdigest()
    assert journal.attempt_fingerprint == journal.fingerprint
    restored = pickle.loads(pickle.dumps(journal))
    assert restored.fingerprint == journal.fingerprint
    assert restored.attempt_fingerprint == journal.attempt_fingerprint


def test_runner_hot_path_skips_deep_rebuild_and_compact_checkpoint_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_shadow_live_protocol(ROOT / "configs/shadow_live_v1.json")
    bindings = shadow_runtime_bindings_from_model_config(ROOT / "configs/model.json")
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=protocol,
        runtime_bindings=bindings,
    )
    runner.process(_input(0, base=MONDAY))

    original = runner.journal.require_consistent
    monkeypatch.setattr(
        runner.journal,
        "require_consistent",
        lambda: pytest.fail("per-clock path must not rebuild full journal"),
    )
    runner.process(_input(1, base=MONDAY))
    monkeypatch.setattr(runner.journal, "require_consistent", original)

    compact = runner.compact_runtime_checkpoint()
    assert "journal" not in compact
    assert "record_payloads" not in compact
    assert len(pickle.dumps(compact)) < len(pickle.dumps(runner))
    restored = ShadowLiveRunner.from_compact_runtime_checkpoint(
        pickle.loads(pickle.dumps(compact)),
        journal_events=runner.journal.events,
        records=runner.records,
    )
    assert restored.record_fingerprint == runner.record_fingerprint
    assert restored.journal.fingerprint == runner.journal.fingerprint
    restored.process(_input(2, base=MONDAY))
    assert len(restored.records) == 3


def test_wal_is_durable_before_processing_and_recovers_pending_attempt(
    tmp_path: Path,
) -> None:
    wal_path = tmp_path / "shadow.wal.jsonl"
    wal = _wal(wal_path)
    session = OperationalShadowSession(
        protocol=load_shadow_operational_protocol(OPERATIONAL_PROTOCOL),
        mapping=_mapping(),
        wal=wal,
        mode="historical_simulation",
    )
    value = _input(0, base=MONDAY)

    def crash_after_durable_attempt(_value):
        reopened = _wal(wal_path)
        assert reopened.pending_input is not None
        assert reopened.pending_input.input_digest == value.input_digest
        raise RuntimeError("simulated process loss")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        session.ingest(value, process=crash_after_durable_attempt)
    assert session.status == "failed_recovery_required"

    reopened = _wal(wal_path)
    recovered = OperationalShadowSession(
        protocol=load_shadow_operational_protocol(OPERATIONAL_PROTOCOL),
        mapping=_mapping(),
        wal=reopened,
        mode="historical_simulation",
    )
    recovered.reconnect()
    result = recovered.recover_pending(
        process=lambda item: {"record_id": f"record:{item.input_digest}"},
    )
    assert result == {"record_id": f"record:{value.input_digest}"}
    assert recovered.status == "connected"
    assert reopened.committed_count == 1
    assert reopened.pending_input is None

    duplicate = recovered.ingest(
        value,
        process=lambda _: pytest.fail("exact duplicate must not be reprocessed"),
    )
    assert duplicate.duplicate is True
    assert reopened.committed_count == 1


def test_reconnect_requires_ordered_backfill_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    wal = _wal(tmp_path / "shadow.wal.jsonl")
    session = OperationalShadowSession(
        protocol=load_shadow_operational_protocol(OPERATIONAL_PROTOCOL),
        mapping=_mapping(),
        wal=wal,
        mode="historical_simulation",
    )

    def process(item):
        return {"record_id": f"record:{item.input_digest}"}

    first = _input(0, base=MONDAY)
    second = _input(1, base=MONDAY)
    third = _input(2, base=MONDAY)
    session.ingest(first, process=process)

    session.disconnect(reason="simulated_network_loss")
    with pytest.raises(Phase9OperationalError, match="disconnected"):
        session.ingest(second, process=process)
    session.reconnect()
    with pytest.raises(ShadowBackfillRequired) as missing:
        session.ingest(third, process=process)
    assert missing.value.missing_completed_clocks == (second.bar.end,)

    session.ingest(second, process=process, backfill=True)
    session.ingest(third, process=process)
    assert wal.committed_count == 3

    conflicting = replace(first, received_at=first.received_at + pd.Timedelta(1, "ms"))
    with pytest.raises(Phase9OperationalError, match="conflicts"):
        session.ingest(conflicting, process=process)
    with pytest.raises(Phase9OperationalError, match="out of order"):
        session.ingest(
            _input(0, base=MONDAY + pd.Timedelta(seconds=30)),
            process=process,
        )


def test_registered_weekend_closure_does_not_create_false_backfill() -> None:
    friday_close = pd.Timestamp("2024-06-07T17:00:00-04:00")
    sunday_first_end = pd.Timestamp("2024-06-09T18:01:00-04:00")
    assert expected_missing_completed_clocks(friday_close, sunday_first_end) == ()


def test_compact_cursor_checkpoint_contains_no_input_or_result_payloads(
    tmp_path: Path,
) -> None:
    wal = _wal(tmp_path / "shadow.wal.jsonl")
    session = OperationalShadowSession(
        protocol=load_shadow_operational_protocol(OPERATIONAL_PROTOCOL),
        mapping=_mapping(),
        wal=wal,
        mode="historical_simulation",
    )
    for index in range(3):
        value = _input(index, base=MONDAY)
        session.ingest(
            value,
            process=lambda item: {"record_id": f"record:{item.input_digest}"},
        )
    checkpoint = session.compact_checkpoint_payload(
        engine_checkpoint_sha256="1" * 64,
    )
    encoded = json.dumps(checkpoint, sort_keys=True)
    assert checkpoint["committed_clocks"] == 3
    assert (
        checkpoint["last_completed_clock"] == _input(2, base=MONDAY).bar.end.isoformat()
    )
    assert "approved_intents" not in encoded
    assert "execution_events" not in encoded
    assert "result_payload" not in encoded
    assert len(encoded) < 4_096


def test_compact_runtime_restore_replays_fsynced_pending_attempt_once(
    tmp_path: Path,
) -> None:
    protocol = load_shadow_operational_protocol(OPERATIONAL_PROTOCOL)
    mapping = _mapping()
    wal_path = tmp_path / "shadow.wal.jsonl"
    wal = _wal(wal_path)
    session = OperationalShadowSession(
        protocol=protocol,
        mapping=mapping,
        wal=wal,
        mode="historical_simulation",
    )
    shadow_protocol = load_shadow_live_protocol(ROOT / "configs/shadow_live_v1.json")
    runner = ShadowLiveRunner(
        engine=_engine(),
        protocol=shadow_protocol,
        runtime_bindings=shadow_runtime_bindings_from_model_config(
            ROOT / "configs/model.json"
        ),
    )
    first = _input(0, base=MONDAY)
    second = _input(1, base=MONDAY)
    session.ingest(first, process=runner.process)
    checkpoint_store = ReplayCheckpointStore(tmp_path / "checkpoint")
    cursor_path = tmp_path / "cursor.json"
    bindings = {"run_manifest_sha256": "9" * 64}
    _checkpoint(
        checkpoint_store,
        runner,
        session,
        bindings=bindings,
        cursor_path=cursor_path,
    )

    # Crash window: the second attempt is durable but neither processed in the
    # checkpointed Engine nor committed in the WAL.
    wal.append_attempt(second)
    reopened = _wal(wal_path)
    recovered_session = OperationalShadowSession(
        protocol=protocol,
        mapping=mapping,
        wal=reopened,
        mode="historical_simulation",
    )
    recovered = _restore(
        checkpoint_store,
        recovered_session,
        bindings=bindings,
        cursor_path=cursor_path,
    )
    assert reopened.pending_input is None
    assert reopened.committed_count == 2
    assert len(recovered.records) == 2
    assert recovered.records[-1].input_digest == second.input_digest


def _authorization_payload(
    *,
    bundle_sha256: str,
    sidecar_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CAPACITY_AUTHORIZATION_SCHEMA_VERSION,
        "status": "externally_frozen_historical_simulation_capacity_authorized",
        "authority": "independent_capacity_review",
        "input_bundle_sha256": bundle_sha256,
        "input_sidecar_sha256": sidecar_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "authorized_rows": 6900,
        "full_6900_file_replay_authorized": True,
        "real_time_live_authorized": False,
        "sealed_oos_authorized": False,
        "maximum_checkpoint_bytes": 500_000_000,
        "maximum_peak_working_set_bytes": 2_000_000_000,
        "minimum_available_memory_bytes": 1,
        "minimum_available_disk_bytes": 1,
        "frozen_at": "2026-08-23T00:00:00Z",
    }
    payload["authorization_id"] = capacity_authorization_id(payload)
    return payload


def test_capacity_authorization_is_external_hash_bound_and_never_self_issued(
    tmp_path: Path,
) -> None:
    bundle_sha = "2" * 64
    sidecar_sha = "5" * 64
    runtime_sha = "3" * 64
    payload = _authorization_payload(
        bundle_sha256=bundle_sha,
        sidecar_sha256=sidecar_sha,
        runtime_identity_sha256=runtime_sha,
    )
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    authorization = load_capacity_authorization(
        path,
        expected_sha256=artifact_sha,
        input_bundle_sha256=bundle_sha,
        input_sidecar_sha256=sidecar_sha,
        runtime_identity_sha256=runtime_sha,
        required_rows=6900,
    )
    assert authorization.full_6900_file_replay_authorized is True
    assert authorization.real_time_live_authorized is False

    with pytest.raises(CapacityAuthorizationError, match="artifact SHA"):
        load_capacity_authorization(
            path,
            expected_sha256="4" * 64,
            input_bundle_sha256=bundle_sha,
            input_sidecar_sha256=sidecar_sha,
            runtime_identity_sha256=runtime_sha,
            required_rows=6900,
        )
    tampered = dict(payload)
    tampered["real_time_live_authorized"] = True
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tampered_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CapacityAuthorizationError, match="cannot authorize live"):
        load_capacity_authorization(
            path,
            expected_sha256=tampered_sha,
            input_bundle_sha256=bundle_sha,
            input_sidecar_sha256=sidecar_sha,
            runtime_identity_sha256=runtime_sha,
            required_rows=6900,
        )


def _base_v1_sidecar(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "week1.jsonl"
    return {
        "format_version": 1,
        "schema_version": "phase9_shadow_week1_materialization_v1",
        "input_schema_version": "phase9_shadow_file_input_v2",
        "status": "complete_historical_cold_start_file_input",
        "authority": "historical_engineering_file_input_only",
        "contract": {
            "symbol": "NQM4",
            "instrument_id": 13743,
            "tick_size": 0.25,
            "point_value": 20.0,
        },
        "window": {
            "id": "2024-06-week-1",
            "start": "2024-06-02T22:00:00+00:00",
            "end_exclusive": "2024-06-07T21:01:00+00:00",
        },
        "census": {
            "rows": 6900,
            "real_completed": 6899,
            "synthetic_no_trade": 1,
            "synthetic_decision_clocks": ["2024-06-07T03:10:00+00:00"],
            "first_decision_clock": "2024-06-02T22:01:00+00:00",
            "last_decision_clock": "2024-06-07T21:00:00+00:00",
            "data_gap_resets": 0,
            "contract_changes": 0,
        },
        "source_bindings": {
            key: {"path": f"fixtures/{key}", "sha256": str(index) * 64}
            for index, key in enumerate(
                (
                    "week1_v5_manifest",
                    "mbo_feature_artifact",
                    "mbo_feature_manifest",
                    "ohlcv_artifact",
                    "ohlcv_manifest",
                ),
                start=1,
            )
        },
        "projection_contract": {
            "execution_provider": "TopOfBookExecutionProvider",
            "source_depth_imbalance": "validated_top5_not_serialized",
            "serialized_depth_imbalance": "recomputed_best_level",
            "data_age_seconds": "scalar_decision_clock_minus_book_observed_at",
            "received_delay_milliseconds": 5,
            "execution_deadline_minutes": 60,
            "account": {
                "equity": 100000.0,
                "flat": True,
                "open_risk_fraction": 0.0,
                "requested_risk_fraction": 0.0,
            },
            "approved_intents": 0,
            "execution_events": 0,
        },
        "limits": {
            "cold_start": True,
            "warm_state_restored": False,
            "warmup_history_included": False,
            "real_time_live": False,
            "multi_day_live_pilot": False,
            "broker_submission": False,
            "live_account_state": False,
            "raw_mbo_read": False,
            "sealed_holdout_read": False,
            "phase9_gate_closed": True,
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": "a" * 64,
            "rows": 6900,
            "bytes": 123456,
            "input_digest_sequence_sha256": "b" * 64,
        },
    }


def test_v3_bundle_payload_binds_exact_census_sources_runtime_code_and_environment(
    tmp_path: Path,
) -> None:
    runtime_identity = runtime_code_environment_identity(ROOT)
    protocol = load_shadow_operational_protocol(OPERATIONAL_PROTOCOL)
    payload = build_week1_bundle_v3_payload(
        _base_v1_sidecar(tmp_path),
        runtime_identity=runtime_identity,
        operational_protocol=protocol,
    )
    assert payload["schema_version"] == PHASE9_WEEK1_BUNDLE_SCHEMA_VERSION
    validate_week1_bundle_v3_payload(
        payload,
        output_path=tmp_path / "week1.jsonl",
        output_sha256="a" * 64,
        output_bytes=123456,
        output_rows=6900,
        runtime_identity=runtime_identity,
        operational_protocol=protocol,
        verify_source_files=False,
    )

    wrong_census = json.loads(json.dumps(payload))
    wrong_census["census"]["rows"] = 6899
    with pytest.raises(Phase9OperationalError, match="census"):
        validate_week1_bundle_v3_payload(
            wrong_census,
            output_path=tmp_path / "week1.jsonl",
            output_sha256="a" * 64,
            output_bytes=123456,
            output_rows=6900,
            runtime_identity=runtime_identity,
            operational_protocol=protocol,
            verify_source_files=False,
        )

    wrong_runtime = json.loads(json.dumps(payload))
    wrong_runtime["runtime_code_environment"]["uv_lock_sha256"] = "f" * 64
    with pytest.raises(Phase9OperationalError, match="runtime/code/environment"):
        validate_week1_bundle_v3_payload(
            wrong_runtime,
            output_path=tmp_path / "week1.jsonl",
            output_sha256="a" * 64,
            output_bytes=123456,
            output_rows=6900,
            runtime_identity=runtime_identity,
            operational_protocol=protocol,
            verify_source_files=False,
        )


def test_v3_bundle_accepts_only_the_exact_registered_week2_census(
    tmp_path: Path,
) -> None:
    runtime_identity = runtime_code_environment_identity(ROOT)
    protocol = load_shadow_operational_protocol(OPERATIONAL_PROTOCOL)
    window = phase9_historical_window("W2")
    base = _base_v1_sidecar(tmp_path)
    base["schema_version"] = "phase9_shadow_historical_window_materialization_v1"
    base["window"] = {
        "id": window.window_id,
        "start": window.start.isoformat(),
        "end_exclusive": window.end_exclusive.isoformat(),
    }
    base["census"] = {
        "rows": window.rows,
        "real_completed": window.real_rows,
        "synthetic_no_trade": len(window.synthetic_clocks),
        "synthetic_decision_clocks": [
            value.isoformat() for value in window.synthetic_clocks
        ],
        "first_decision_clock": window.first_decision_clock.isoformat(),
        "last_decision_clock": window.last_decision_clock.isoformat(),
        "data_gap_resets": 0,
        "contract_changes": 0,
    }
    base["source_bindings"].pop("week1_v5_manifest")
    base["source_bindings"]["week2_extension_v2_manifest"] = {
        "path": "fixtures/week2_extension_v2_manifest",
        "sha256": "6" * 64,
    }
    payload = build_phase9_bundle_v3_payload(
        base,
        runtime_identity=runtime_identity,
        operational_protocol=protocol,
    )
    assert payload["admission_contract"]["exact_window_id"] == window.window_id
    assert payload["admission_contract"]["exact_rows"] == 6900
    validate_phase9_bundle_v3_payload(
        payload,
        output_path=tmp_path / "week1.jsonl",
        output_sha256="a" * 64,
        output_bytes=123456,
        output_rows=6900,
        runtime_identity=runtime_identity,
        operational_protocol=protocol,
        verify_source_files=False,
    )

    wrong_synthetic = json.loads(json.dumps(payload))
    wrong_synthetic["census"]["synthetic_decision_clocks"] = [
        "2024-06-07T03:10:00+00:00"
    ]
    with pytest.raises(Phase9OperationalError, match="census"):
        validate_phase9_bundle_v3_payload(
            wrong_synthetic,
            output_path=tmp_path / "week1.jsonl",
            output_sha256="a" * 64,
            output_bytes=123456,
            output_rows=6900,
            runtime_identity=runtime_identity,
            operational_protocol=protocol,
            verify_source_files=False,
        )
