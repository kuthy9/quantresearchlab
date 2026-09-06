#!/usr/bin/env python3
"""Validate or explicitly run the frozen Phase 8 v2 development study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smc_trader.execution_research_runner import (
    Phase8RunnerError,
    run_phase8_execution_research,
    validate_phase8_run_manifest,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs/research/execution_research_phase8_v2_run_template.yaml"
)


def _validation_payload(contract) -> dict[str, object]:
    return {
        "manifest_sha256": contract.manifest_sha256,
        "status": contract.status,
        "ready": contract.ready,
        "blockers": list(contract.blockers),
        "source_mode": (
            None if contract.source_mode is None else contract.source_mode.value
        ),
        "opened_dataset_bindings": [],
        "written_artifacts": [],
        "authority": "research_only_never_submit",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an inert Phase 8 manifest by default. A development run "
            "requires the explicit --execute-development flag."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="repository-local Phase 8 run manifest",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate identities/readiness without opening datasets or writing outputs (default)",
    )
    mode.add_argument(
        "--execute-development",
        action="store_true",
        help="run only a fully frozen open-development W1/W2 manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    try:
        if not args.execute_development:
            contract = validate_phase8_run_manifest(
                manifest,
                project_root=root,
                validate_input_files=False,
            )
            print(json.dumps(_validation_payload(contract), sort_keys=True))
            return 0
        result = run_phase8_execution_research(manifest, project_root=root)
        payload = asdict(result)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["authority"] = "research_only_never_submit"
        print(json.dumps(payload, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, Phase8RunnerError) as exc:
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "written_artifacts": (
                        []
                        if not args.execute_development
                        else "audit_registered_output_paths_after_failed_execution"
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
