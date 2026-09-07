"""Every detector protocol the Eye executes must be inside its identity.

The atomic definition identity is a SHA-256 over the registry, the parameters,
the data-split registry and the primitive protocol files.  Those protocol files
are *discovered* by scanning the registered parameters' ``source`` strings, so a
protocol that no parameter happens to mention is silently excluded: its
thresholds can then change while the semantic identity — and therefore the
freeze policy — reports that nothing changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eyes.core.semantics import SemanticRegistry


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_KEYS = (
    "structure_protocol",
    "liquidity_protocol",
    "displacement_protocol",
    "zone_protocol",
    "range_auction_protocol",
    "interaction_protocol",
)


def _loaded_protocols() -> set[str]:
    model = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))
    observer = model["observer"]
    return {
        observer[key]
        for key in PROTOCOL_KEYS
        if observer.get(key) is not None
    }


def test_every_protocol_the_eye_loads_is_bound_into_the_atomic_identity() -> None:
    registry = SemanticRegistry.from_file()
    bound = set(registry.definition_identity.primitive_protocol_sha256)

    missing = sorted(_loaded_protocols() - bound)

    assert not missing, (
        f"the Eye loads {missing} but the atomic identity does not hash them; "
        "changing their thresholds would leave the semantic version unchanged"
    )


@pytest.mark.parametrize("protocol", sorted(_loaded_protocols()))
def test_a_bound_protocol_hash_matches_the_file_on_disk(protocol: str) -> None:
    import hashlib

    registry = SemanticRegistry.from_file()
    expected = registry.definition_identity.primitive_protocol_sha256[protocol]
    digest = hashlib.sha256((ROOT / protocol).read_bytes()).hexdigest()

    assert digest == expected
