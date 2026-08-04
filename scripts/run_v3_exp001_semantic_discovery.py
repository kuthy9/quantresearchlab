#!/usr/bin/env python3
"""Validate or, after a later governance release, run v3 semantic discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import sha256_file
from smc_trader.semantic_audit import (
    DEFAULT_AUDIT_CONTRACT,
    load_audit_contract,
)
from smc_trader.semantic_discovery_runner import (
    RUNNER_CONTRACT,
    load_runner_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("validate-only", "pass1", "pass2"),
    )
    parser.add_argument(
        "--audit-contract",
        default=str(DEFAULT_AUDIT_CONTRACT),
    )
    parser.add_argument(
        "--runner-contract",
        default=str(RUNNER_CONTRACT),
    )
    parser.add_argument(
        "--model-config",
        default=(
            "configs/model_v3_0_1_exp001_structure_bos_identity_r3.json"
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-source-rows", type=int, default=25_000)
    parser.add_argument(
        "--diagnostic-stop-after-source-rows",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_contract)
    runner_path = Path(args.runner_contract)
    runner_contract = load_runner_contract(runner_path)
    # Validation and the current governance rejection must not open the
    # registered discovery source merely to recompute its file hash.
    audit = load_audit_contract(
        audit_path,
        verify_bound_files=False,
    )
    implementation_hashes = {
        "semantic_discovery_runner_sha256": sha256_file(
            ROOT / "smc_trader/semantic_discovery_runner.py"
        ),
        "semantic_audit_sha256": sha256_file(
            ROOT / "smc_trader/semantic_audit.py"
        ),
        "io_sha256": sha256_file(ROOT / "smc_trader/io.py"),
        "causal_sha256": sha256_file(ROOT / "smc_trader/causal.py"),
        "market_clock_sha256": sha256_file(
            ROOT / "smc_trader/market_clock.py"
        ),
        "observation_sha256": sha256_file(
            ROOT / "smc_trader/observation.py"
        ),
        "structure_sha256": sha256_file(
            ROOT / "smc_trader/structure.py"
        ),
    }
    validation = {
        "status": "implementation_contract_valid",
        "authorization": runner_contract["authorization"],
        "audit_contract_sha256": sha256_file(audit_path),
        "runner_contract_sha256": sha256_file(runner_path),
        "source_sha256": audit["bindings"]["causal_source_sha256"],
        "source_start": audit["source"]["start"],
        "source_end_exclusive": audit["source"]["end_exclusive"],
        "implementation_hashes": implementation_hashes,
    }
    if args.phase == "validate-only":
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    raise PermissionError(
        "this engineering CLI is permanently execution-locked; use the "
        "phase-scoped Pass1 release CLI after independent governance approval"
    )


if __name__ == "__main__":
    main()
