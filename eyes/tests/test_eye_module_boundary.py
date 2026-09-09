"""The Trading Eye may not import the layers that consume it.

The Eye publishes deterministic market facts.  Interpretation, orchestration and
execution all sit downstream of it, so an import pointing that way is a layering
defect however harmless the imported name looks.

Since the four-way split the Eye's modules live in ``eyes/core/`` and its shared
foundations in ``shares/core/``, so this scan resolves both the intra-package
relative imports and the cross-package absolute ones.

The type contracts moved out of ``shares/core/model.py`` into ``contract/``, so
this file also pins the contract layering itself: a contract package may only
import from packages earlier in the chain.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SUBSYSTEMS = ("eyes", "brain", "execution", "shares")

# Everything the Eye is built from: normalization, clock, registries, the six
# detectors, emission, storage and snapshot reduction.  The value is the
# subsystem package that now owns the module.
EYE_MODULES = {
    "causal": "eyes",
    "displacement": "eyes",
    "displacement_observer": "eyes",
    "event_memory": "eyes",
    "event_store": "eyes",
    "foundation_registry": "eyes",
    "interaction": "eyes",
    "io": "shares",
    "liquidity": "eyes",
    "market_clock": "shares",
    "market_state": "eyes",
    "observation": "eyes",
    "range_auction": "eyes",
    "scale_registry": "shares",
    "semantic_event_emitter": "eyes",
    "semantics": "eyes",
    "structure": "eyes",
    "zone": "eyes",
}

# Layers that consume the Eye's output and must never be imported by it.
DOWNSTREAM_MODULES = frozenset(
    {
        "decision",
        "engine",
        "execution",
        "risk",
        "scene_graph",
        "simulation",
    }
)


def _local_imports(path: pathlib.Path) -> set[str]:
    """Every first-party module name this file imports, however it spells it."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if node.level:
                # ``from .zone import ...`` inside one subsystem package.
                found.add(parts[0])
            elif len(parts) >= 3 and parts[0] in SUBSYSTEMS and parts[1] == "core":
                # ``from shares.core.model import ...`` across subsystems.
                found.add(parts[2])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 3 and parts[0] in SUBSYSTEMS and parts[1] == "core":
                    found.add(parts[2])
    return found


@pytest.mark.parametrize("module", sorted(EYE_MODULES))
def test_eye_module_does_not_import_a_downstream_layer(module: str) -> None:
    path = ROOT / EYE_MODULES[module] / "core" / f"{module}.py"
    assert path.exists(), f"{module}.py is missing from the Eye"

    offending = sorted(_local_imports(path) & DOWNSTREAM_MODULES)

    assert not offending, (
        f"{module}.py imports downstream layer(s) {offending}; the Eye "
        "publishes facts and must not depend on what consumes them"
    )


# The contract layering, most fundamental first.  ``contract/`` replaced
# ``shares/core/model.py``: one package per boundary the payload crosses.
CONTRACT_ORDER = (
    "market",
    "execution",
    "eye",
    "brain",
    "decision",
    "risk",
    "research",
)


def _contract_imports(path: pathlib.Path) -> set[str]:
    """Every ``contract.<package>`` this file imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "contract" and len(parts) >= 2:
                found.add(parts[1])
            elif node.level and node.module in CONTRACT_ORDER:
                found.add(node.module)
    return found


@pytest.mark.parametrize("package", CONTRACT_ORDER)
def test_contract_package_only_imports_more_fundamental_packages(
    package: str,
) -> None:
    directory = ROOT / "contract" / package
    assert directory.is_dir(), f"contract/{package} is missing"

    allowed = set(CONTRACT_ORDER[: CONTRACT_ORDER.index(package) + 1])
    for path in sorted(directory.glob("*.py")):
        offending = sorted(_contract_imports(path) - allowed)
        assert not offending, (
            f"contract/{package}/{path.name} imports {offending}, which sit "
            f"later in the contract chain {CONTRACT_ORDER}"
        )


def test_the_retired_model_module_has_no_replacement_in_a_subsystem() -> None:
    """``contract/`` is the only home for a cross-boundary payload type."""

    for subsystem in SUBSYSTEMS:
        stray = ROOT / subsystem / "core" / "model.py"
        assert not stray.exists(), (
            f"{subsystem}/core/model.py exists again; cross-boundary contracts "
            "belong in contract/, not inside one subsystem"
        )
