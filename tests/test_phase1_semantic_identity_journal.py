from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.event_store import (
    ImmutableEventStore,
    read_event_journal,
    write_event_journal,
)
from smc_trader.model import (
    Direction,
    EventKind,
    MarketEvent,
    SMC_SEMANTIC_VERSION,
    Timeframe,
)
from smc_trader.semantics import SemanticRegistry, SemanticRegistryError


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2026-08-19 10:00", tz="America/New_York") + pd.Timedelta(
        minutes * 60,
        unit="s",
    )


def _events() -> tuple[MarketEvent, ...]:
    first = MarketEvent(
        event_id="event-001",
        kind=EventKind.SWING_CONFIRMED,
        observed_at=_clock(10),
        timeframe=Timeframe.M5,
        side="above",
        price=21_500.0,
        strength=0.8,
        source_ids=("pivot-candle",),
        details={"features": ["prominence", "duration"]},
        direction=Direction.LONG,
        sequence_no=0,
        event_time=_clock(0),
        known_at=_clock(10),
    )
    second = MarketEvent(
        event_id="event-002",
        kind=EventKind.FVG_CREATED,
        observed_at=_clock(15),
        timeframe=Timeframe.M5,
        side="below",
        price=21_495.0,
        strength=0.6,
        source_ids=(first.event_id, "candle-2", "candle-3"),
        details={"lower_bound": 21_494.0, "upper_bound": 21_496.0},
        entity_id="fvg:example",
        lifecycle="open",
        formed_at=_clock(15),
        direction=Direction.SHORT,
        sequence_no=1,
        event_time=_clock(15),
        known_at=_clock(15),
    )
    third = MarketEvent(
        event_id="event-003",
        kind=EventKind.LEVEL_TOUCHED,
        observed_at=_clock(20),
        timeframe=Timeframe.M5,
        side="below",
        price=21_496.0,
        strength=0.4,
        source_ids=(second.event_id,),
        details={"touch_ordinal": 1},
        sequence_no=2,
        event_time=_clock(20),
        known_at=_clock(20),
    )
    return first, second, third


def _bound_store() -> tuple[SemanticRegistry, ImmutableEventStore]:
    registry = SemanticRegistry.from_file()
    store = ImmutableEventStore(
        semantic_version=registry.semantic_version,
        definition_identity=registry.definition_identity,
    )
    store.append_batch(_events())
    return registry, store


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_definition_identity_binds_registry_parameters_protocols_and_split() -> None:
    registry = SemanticRegistry.from_file()
    identity = registry.definition_identity

    assert registry.identity == identity.identity
    assert identity.registry_sha256 == _sha256(registry.source_path)
    assert identity.parameters_sha256 == _sha256(
        registry.parameters.source_path
    )
    assert identity.data_split_sha256 == _sha256(
        Path(identity.data_split_registry)
    )
    assert set(identity.primitive_protocol_sha256) == {
        "configs/primitives_displacement.json",
        "configs/primitives_range.json",
        "configs/primitives_structure_liquidity.json",
        "configs/primitives_zones.json",
    }
    for path, digest in identity.primitive_protocol_sha256.items():
        assert digest == _sha256(Path(path))
    assert type(identity).from_metadata(identity.to_metadata()) == identity


def test_same_version_registry_byte_drift_fails_closed(tmp_path: Path) -> None:
    registry = SemanticRegistry.from_file()
    changed = tmp_path / "registry.yaml"
    changed.write_bytes(registry.source_path.read_bytes() + b"\n")

    with pytest.raises(
        SemanticRegistryError,
        match="identity drifted without a version change",
    ):
        SemanticRegistry.from_file(
            changed,
            expected_definition_identity=registry.definition_identity,
        )


@pytest.mark.parametrize(
    "changed_reference",
    (
        "configs/primitives_displacement.json",
        "configs/data_splits.json",
    ),
)
def test_referenced_protocol_or_split_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_reference: str,
) -> None:
    source_registry = SemanticRegistry.from_file()
    references = {
        "semantics/registry.yaml": source_registry.source_path,
        "semantics/parameters.yaml": source_registry.parameters.source_path,
        source_registry.definition_identity.data_split_registry: Path(
            source_registry.definition_identity.data_split_registry
        ),
        **{
            reference: Path(reference)
            for reference in source_registry.definition_identity.primitive_protocol_sha256
        },
    }
    for relative, source in references.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.chdir(tmp_path)
    baseline = SemanticRegistry.from_file()
    changed = tmp_path / changed_reference
    changed.write_bytes(changed.read_bytes() + b"\n")

    with pytest.raises(
        SemanticRegistryError,
        match="identity drifted without a version change",
    ):
        SemanticRegistry.from_file(
            expected_definition_identity=baseline.definition_identity,
        )


def test_event_store_fingerprint_and_checkpoint_bind_definition_identity() -> None:
    registry, store = _bound_store()
    drifted_identity = replace(
        registry.definition_identity,
        registry_sha256="0" * 64,
    )
    drifted = ImmutableEventStore.from_events(
        _events(),
        definition_identity=drifted_identity,
    )

    assert store.fingerprint() != drifted.fingerprint()
    assert store.metadata()["semantic_definition_identity"] == registry.identity
    with pytest.raises(ValueError, match="identity drifted"):
        store.require_definition_identity(drifted_identity)

    checkpoint = store.checkpoint_metadata()
    restored = ImmutableEventStore.from_checkpoint(
        store.events(),
        checkpoint,
        expected_definition_identity=registry.definition_identity,
    )
    assert restored == store
    assert restored.fingerprint() == store.fingerprint()

    changed_checkpoint = dict(checkpoint)
    changed_checkpoint["event_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="content binding"):
        ImmutableEventStore.from_checkpoint(store.events(), changed_checkpoint)


def test_parquet_event_journal_roundtrip_is_logically_deterministic(
    tmp_path: Path,
) -> None:
    registry, store = _bound_store()
    left = write_event_journal(
        tmp_path / "left",
        store,
        maximum_rows_per_shard=1,
    )
    right = write_event_journal(
        tmp_path / "right",
        store,
        maximum_rows_per_shard=1,
    )
    differently_sharded = write_event_journal(
        tmp_path / "differently-sharded",
        store,
        maximum_rows_per_shard=2,
    )

    assert left.logical_fingerprint == store.fingerprint()
    assert left.logical_fingerprint == right.logical_fingerprint
    assert left.logical_fingerprint == differently_sharded.logical_fingerprint
    assert left.manifest_sha256 == right.manifest_sha256
    assert left.shard_sha256 == right.shard_sha256
    assert left.manifest_sha256 != differently_sharded.manifest_sha256

    replayed = read_event_journal(
        tmp_path / "left",
        expected_definition_identity=registry.definition_identity,
    )
    assert replayed.store == store
    assert replayed.store.events() == store.events()
    assert replayed.logical_fingerprint == store.fingerprint()
    assert replayed.manifest.rows == len(store)

    with pytest.raises(FileExistsError, match="immutable"):
        write_event_journal(tmp_path / "left", store)


def test_event_journal_rejects_parquet_tampering(tmp_path: Path) -> None:
    _, store = _bound_store()
    manifest = write_event_journal(
        tmp_path / "journal",
        store,
        maximum_rows_per_shard=1,
    )
    shard = tmp_path / "journal" / "events" / "part-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="shard hash"):
        read_event_journal(tmp_path / "journal")
    assert manifest.rows == 3


def test_event_journal_rejects_rehashed_manifest_binding_tamper(
    tmp_path: Path,
) -> None:
    _, store = _bound_store()
    write_event_journal(tmp_path / "journal", store)
    manifest_path = tmp_path / "journal" / "events.manifest.json"
    sidecar_path = tmp_path / "journal" / "events.manifest.sha256"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["bindings"]["semantic_definition_identity"] = "0" * 64
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    sidecar_path.write_text(f"{_sha256(manifest_path)}\n", encoding="ascii")

    with pytest.raises(ValueError, match="definition binding"):
        read_event_journal(tmp_path / "journal")


def test_event_journal_rejects_wrong_expected_definition(tmp_path: Path) -> None:
    registry, store = _bound_store()
    write_event_journal(tmp_path / "journal", store)
    drifted_identity = replace(
        registry.definition_identity,
        data_split_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="identity drifted"):
        read_event_journal(
            tmp_path / "journal",
            expected_definition_identity=drifted_identity,
        )


def test_legacy_unbound_store_api_and_fingerprint_remain_compatible(
    tmp_path: Path,
) -> None:
    store = ImmutableEventStore.from_events(
        _events(),
        semantic_version=SMC_SEMANTIC_VERSION,
    )
    digest = hashlib.sha256(SMC_SEMANTIC_VERSION.encode("utf-8"))
    for event in store.events():
        # Verify compatibility through a second legacy store instead of
        # relying on the private event-digest representation.
        assert event.semantic_version == SMC_SEMANTIC_VERSION
    assert store.semantic_definition_identity is None
    assert store.fingerprint() == ImmutableEventStore.from_events(
        _events()
    ).fingerprint()
    assert digest.hexdigest() != store.fingerprint()
    with pytest.raises(ValueError, match="complete semantic definition"):
        write_event_journal(tmp_path / "unused", store)
