#!/usr/bin/env python3
"""Validate or execute only a governance-released v3 EXP001 Pass1 scan."""
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
    PASS1_EXECUTION_RELEASE,
    RUNNER_CONTRACT,
    SemanticDiscoveryRunner,
    load_pass1_execution_release,
    load_runner_contract,
    runtime_environment,
    source_for_pass1_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate-only", "run"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact release-bound output and checkpoint",
    )
    return parser.parse_args()


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _implementation_hashes() -> dict[str, str]:
    return {
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


def main() -> None:
    args = parse_args()
    audit_path = ROOT / DEFAULT_AUDIT_CONTRACT
    runner_path = ROOT / RUNNER_CONTRACT
    release_path = ROOT / PASS1_EXECUTION_RELEASE

    # These validations deliberately do not open or hash the market source.
    audit = load_audit_contract(
        audit_path,
        verify_bound_files=False,
    )
    runner_contract = load_runner_contract(
        runner_path,
        verify_bound_files=True,
    )
    release = load_pass1_execution_release(
        release_path,
        verify_bound_files=True,
    )
    release_sha256 = sha256_file(release_path)
    validation = {
        "status": release["status"],
        "execution_authorized": release["execution_authorized"],
        "authorized_phases": release["authorized_phases"],
        "release_id": release["release_id"],
        "release_sha256": release_sha256,
        "audit_contract_sha256": sha256_file(audit_path),
        "engineering_runner_contract_sha256": sha256_file(
            runner_path
        ),
        "runtime": runtime_environment(),
        "source_hash_recomputed": False,
        "source_rows_read": 0,
        "output_path": release["output"]["path"],
    }
    if args.action == "validate-only":
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if release.get("execution_authorized") is not True:
        raise PermissionError(
            "Pass1 execution release is a draft; source access remains locked"
        )

    # No caller-supplied source, model, output, batch, history, checkpoint, or
    # interruption option exists.  Every production value comes from the
    # independently frozen release.
    run = release["run"]
    source = source_for_pass1_release(
        release,
        resume=bool(args.resume),
    )
    implementation_hashes = _implementation_hashes()
    discovery = SemanticDiscoveryRunner(
        audit_contract=audit,
        runner_contract=runner_contract,
        source=source,
        model_config_path=_absolute(
            release["bound_files"]["model_config_sha256"]
        ),
        output_root=_absolute(release["output"]["path"]),
        audit_contract_sha256=sha256_file(audit_path),
        runner_contract_sha256=sha256_file(runner_path),
        implementation_hashes=implementation_hashes,
        maximum_history=int(run["maximum_history"]),
        checkpoint_source_rows=int(
            run["checkpoint_source_rows"]
        ),
        execution_release=release,
        execution_release_sha256=release_sha256,
    )
    print(discovery.run_pass1(resume=bool(args.resume)))


if __name__ == "__main__":
    main()
