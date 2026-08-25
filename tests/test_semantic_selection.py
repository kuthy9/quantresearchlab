from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smc_trader import foundation_registry as foundation_module
from smc_trader.engine import ContinuousSMCEngine
from smc_trader.semantics import (
    SemanticRegistry,
    SemanticRegistryError,
    load_semantic_selection,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection() -> dict[str, object]:
    model = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))
    return model["semantic_selection"]


def test_selection_keeps_two_identities_and_parent_binding() -> None:
    selection = load_semantic_selection(_selection(), root=ROOT)

    assert selection.atomic_semantics_version == "smc_semantics_v1.2"
    assert selection.atomic_definition_identity == (
        "83f6f7dda806271c9dadfb78cbeb40ac14c2a0fda71463e65bd07a963e3040c7"
    )
    assert selection.foundation_projection_version == (
        "smc_semantic_foundation_v2.0"
    )
    assert selection.foundation_registry_identity == (
        "ac04636919931d774309a0c306764fdf8eb53aee41df0f31d4d94e5b9125732b"
    )
    assert selection.parent_atomic_semantics_version == (
        selection.atomic_semantics_version
    )
    assert "stack_version" not in selection.to_metadata()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("atomic_semantics_version", "smc_semantics_v2.0", "atomic semantics"),
        (
            "foundation_projection_version",
            "smc_semantic_foundation_v3.0",
            "Foundation projection",
        ),
        ("atomic_definition_identity", "0" * 64, "definition identity drifted"),
        (
            "foundation_registry_identity",
            "0" * 64,
            "canonical identity mismatch",
        ),
    ),
)
def test_selection_rejects_version_or_identity_drift(
    field: str,
    value: str,
    message: str,
) -> None:
    payload = deepcopy(_selection())
    payload[field] = value
    with pytest.raises(SemanticRegistryError, match=message):
        load_semantic_selection(payload, root=ROOT)


def test_selection_rejects_shape_and_wrong_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = {**_selection(), "combined_semantic_version": "forbidden"}
    with pytest.raises(SemanticRegistryError, match="fields differ"):
        load_semantic_selection(extra, root=ROOT)

    loaded = foundation_module.load_foundation_registry()
    wrong_parent = SimpleNamespace(
        foundation_version=loaded.foundation_version,
        parent_atomic_semantic_version="smc_semantics_v1.1",
        identity=loaded.identity,
    )
    monkeypatch.setattr(
        foundation_module,
        "load_foundation_registry",
        lambda *_args, **_kwargs: wrong_parent,
    )
    with pytest.raises(SemanticRegistryError, match="parent atomic semantics"):
        load_semantic_selection(_selection(), root=ROOT)


def test_model_has_no_duplicate_observer_selection_aliases() -> None:
    model = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))
    observer = model["observer"]
    assert set(observer).isdisjoint(
        {
            "semantic_registry",
            "canonical_foundation_enabled",
            "canonical_foundation_registry",
            "canonical_foundation_identity",
        }
    )


def test_engine_injects_the_single_loaded_atomic_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SemanticRegistry.from_file.__func__
    calls: list[object] = []

    def tracked(cls, source, **kwargs):
        calls.append(source)
        return original(cls, source, **kwargs)

    monkeypatch.setattr(SemanticRegistry, "from_file", classmethod(tracked))
    engine = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )

    assert len(calls) == 1
    assert engine.observer.semantic_registry.identity == (
        "83f6f7dda806271c9dadfb78cbeb40ac14c2a0fda71463e65bd07a963e3040c7"
    )
