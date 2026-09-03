"""The Trading Eye may not import the layers that consume it.

The Eye publishes deterministic market facts.  Interpretation, orchestration and
execution all sit downstream of it, so an import pointing that way is a layering
defect however harmless the imported name looks.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parent.parent / "smc_trader"

# Everything the Eye is built from: normalization, clock, registries, the six
# detectors, emission, storage and snapshot reduction.
EYE_MODULES = (
    "causal",
    "displacement",
    "displacement_observer",
    "event_memory",
    "event_store",
    "foundation_registry",
    "interaction",
    "io",
    "liquidity",
    "market_clock",
    "market_state",
    "model",
    "observation",
    "range_auction",
    "scale_registry",
    "semantic_event_emitter",
    "semantics",
    "structure",
    "zone",
)

# Layers that consume the Eye's output and must never be imported by it.
DOWNSTREAM_MODULES = frozenset(
    {
        "decision",
        "engine",
        "execution",
        "execution_fsm",
        "playbooks",
        "risk",
        "scene_graph",
        "simulation",
    }
)


def _local_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("smc_trader."):
                    found.add(alias.name.split(".")[1])
    return found


@pytest.mark.parametrize("module", EYE_MODULES)
def test_eye_module_does_not_import_a_downstream_layer(module: str) -> None:
    path = ROOT / f"{module}.py"
    assert path.exists(), f"{module}.py is missing from the Eye"

    offending = sorted(_local_imports(path) & DOWNSTREAM_MODULES)

    assert not offending, (
        f"{module}.py imports downstream layer(s) {offending}; the Eye "
        "publishes facts and must not depend on what consumes them"
    )
