from __future__ import annotations

import importlib.util
from pathlib import Path
import platform

import pytest


HARNESS = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_v3_observer_engineering.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "benchmark_v3_observer_engineering",
        HARNESS,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_fingerprint_reports_current_process_translation() -> None:
    harness = _load_harness()
    fingerprint = harness.runtime_fingerprint()
    assert fingerprint["platform_machine"] == platform.machine()
    translated = fingerprint["rosetta_translated"]
    assert translated is None or isinstance(translated, bool)

    if platform.system() != "Darwin":
        assert fingerprint["host_arm64_capable"] is None
        assert translated is None
        return
    host_arm64 = fingerprint["host_arm64_capable"]
    assert isinstance(host_arm64, bool)
    if host_arm64 and platform.machine() == "x86_64":
        assert translated is True
    elif not host_arm64:
        assert translated is False
    else:
        pytest.skip("native arm64 process does not exercise Rosetta")
