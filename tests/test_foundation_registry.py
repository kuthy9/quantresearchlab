from __future__ import annotations

import json
from pathlib import Path

import pytest

from smc_trader.foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
    FoundationRegistryError,
    load_foundation_registry,
)


def test_foundation_registry_loads_without_granting_authority() -> None:
    registry = load_foundation_registry()
    assert registry.foundation_version == FOUNDATION_VERSION
    assert registry.identity == FOUNDATION_CANONICAL_IDENTITY
    assert len(registry.canonical_objects) == 24
    assert set(registry.payload["authority"].values()) == {False}
    assert registry.payload["fvg_expiry"]["bar_ttl"] is None
    assert "semantic_version" in registry.payload[
        "common_generation_contract"
    ]["required_fields"]
    assert "foundation_version" not in registry.payload[
        "common_generation_contract"
    ]["required_fields"]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("authority", "trading_authority"), True),
        (("fvg_expiry", "bar_ttl"), 20),
        (("liquidity_cluster", "merge_tolerance_ticks"), 2),
        (("liquidity_cluster", "linkage"), "single_link"),
        (
            ("structure_generation", "mss_confirms_opposite_generation"),
            True,
        ),
        (("multi_bar_ancestry", "temporal_relation_is_ancestry"), True),
    ),
)
def test_foundation_registry_rejects_material_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
) -> None:
    payload = json.loads(
        Path("semantics/foundation_v2_1.yaml").read_text(encoding="utf-8")
    )
    payload[path[0]][path[1]] = value
    target = tmp_path / "foundation.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FoundationRegistryError):
        load_foundation_registry(target)


def test_foundation_registry_identity_is_bound() -> None:
    registry = load_foundation_registry()
    with pytest.raises(FoundationRegistryError, match="identity mismatch"):
        load_foundation_registry(expected_identity="0" * 64)


def test_foundation_registry_rejects_unenumerated_nested_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        Path("semantics/foundation_v2_1.yaml").read_text(encoding="utf-8")
    )
    payload["first_reinteraction"]["future_selected_override"] = True
    target = tmp_path / "foundation.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FoundationRegistryError, match="identity mismatch"):
        load_foundation_registry(target)
