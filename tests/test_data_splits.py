from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_continuous_replay import _load_mbo_execution
from smc_trader.validation import ValidationProtocolError, load_validation_protocol


ROOT = Path(__file__).resolve().parents[1]
DATA_SPLITS = ROOT / "configs/data_splits.json"


def test_current_data_splits_preserve_causal_and_mbo_identities() -> None:
    protocol = load_validation_protocol(DATA_SPLITS)

    assert protocol.schema_version == 1
    assert protocol.causal_source.path.endswith(
        "nq_1m_previous_session_front_v2_3_2017_2026.parquet"
    )
    assert protocol.causal_source.sha256 == (
        "84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b"
    )
    assert protocol.causal_source.manifest_sha256 == (
        "391415fa23abfde5281c60ad4d882b32ff284b1aedb6298e398ed72036a9042f"
    )
    assert protocol.mbo_identity.development_partition_manifest_sha256 == (
        "9fbaef324de51cdbc60e13ea07acb7015cce6c409545761d7148ec5f7e8045cd"
    )
    assert len(protocol.mbo_identity.development_execution_artifacts) == 2
    assert protocol.mbo_identity.sealed_sha256 == (
        "8927a904876d3a22af8dce7f3fe05d2ba574f2083435ba9bb477faafc45477b8"
    )
    assert protocol.mbo_identity.sealed_manifest_sha256 == (
        "cb04c2706a21bc4c293e6900ef7d34b0fecf13bd304a13dfbebf162278e17c6f"
    )


def test_current_data_splits_keep_stage_boundaries_and_group4_windows() -> None:
    protocol = load_validation_protocol()

    assert tuple(protocol.ohlcv_windows) == (
        "development",
        "calibration",
        "rolling_oof",
        "sealed_holdout",
    )
    assert (
        protocol.classify_ohlcv(
            pd.Timestamp("2022-02-01", tz="America/New_York"),
            pd.Timestamp("2022-03-01", tz="America/New_York"),
        ).role
        == "calibration"
    )
    assert (
        protocol.classify_mbo(
            pd.Timestamp("2024-06-02", tz="UTC"),
            pd.Timestamp("2024-06-03", tz="UTC"),
        ).role
        == "development"
    )
    assert tuple(
        window.role
        for window in protocol.fixed_development_windows["mature_range_coverage"]
    ) == (
        "2017-june",
        "2018-june",
        "2019-june",
        "2020-june",
        "2021-june",
    )
    with pytest.raises(ValidationProtocolError):
        protocol.classify_ohlcv(
            pd.Timestamp("2026-03-31", tz="America/New_York"),
            pd.Timestamp("2026-04-02", tz="America/New_York"),
        )


def test_current_schema_rejects_changed_mbo_manifest_binding(tmp_path: Path) -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    payload["sources"]["mbo"]["development"]["partition_manifest_sha256"] = "0" * 64
    candidate = tmp_path / "data_splits.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationProtocolError, match="SHA-256 mismatch"):
        load_validation_protocol(candidate)


def test_archived_protocol_shape_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "archived_protocol.json"
    candidate.write_text(json.dumps({"protocol_version": "archived-test"}))

    with pytest.raises(ValidationProtocolError, match="schema_version must be 1"):
        load_validation_protocol(candidate)


def test_registered_development_mbo_artifacts_load_by_exact_identity() -> None:
    protocol = load_validation_protocol(DATA_SPLITS)
    for artifact in protocol.mbo_identity.development_execution_artifacts:
        source = ROOT / artifact.path
        manifest_path = ROOT / artifact.manifest_path
        if not source.is_file() or not manifest_path.is_file():
            pytest.skip("registered development MBO artifact is not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store, loaded_manifest = _load_mbo_execution(
            source,
            validation=protocol,
            start=pd.Timestamp(manifest["start"]),
            end=pd.Timestamp(manifest["end_exclusive"]),
            reveal_sealed_holdout=False,
        )
        assert len(store.frame) > 0
        assert loaded_manifest["output_sha256"] == artifact.sha256
