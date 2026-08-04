#!/usr/bin/env python3
"""Create-once R20 runtime/native content materialization.

This file is stdlib-only.  It is not authorized to run until its exact Stage-A
hash and tool manifest receive the required independent approvals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Any

from v3_structure_bos_runtime_tool_r20 import (
    MATERIALIZATION_ATTEMPT_ID,
    PROTOCOL_ID,
    R20ToolError,
    canonical_json_bytes,
    fsync_directory,
    lexical_relative,
    load_canonical_json,
    materialize_runtime_closure,
    sha256_file,
    validate_tool_freeze_environment,
    write_atomic_once,
    write_exclusive,
)


def _aware_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_no_symlink_components(path: Path, *, anchor: Path) -> None:
    lexical_relative(path, anchor)
    current = anchor
    if stat.S_ISLNK(current.lstat().st_mode):
        raise R20ToolError(f"workspace anchor is a symlink: {current}")
    for part in path.relative_to(anchor).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if stat.S_ISLNK(current.lstat().st_mode):
                raise R20ToolError(f"output path component is a symlink: {current}")


def _create_attempt_root(
    *,
    output_root: Path,
    workspace_root: Path,
    manifest_sha256: str,
    run_id: str,
) -> None:
    _require_no_symlink_components(output_root.parent, anchor=workspace_root)
    if output_root.exists() or output_root.is_symlink():
        raise R20ToolError(f"materialization output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(output_root.parent, anchor=workspace_root)
    output_root.mkdir(mode=0o700, exist_ok=False)
    fsync_directory(output_root.parent)
    attempt = {
        "format_version": 1,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": run_id,
        "mode": "runtime_content_materialization",
        "tool_manifest_sha256": manifest_sha256,
        "created_at": _aware_now(),
    }
    write_exclusive(
        output_root / "ATTEMPT.json",
        canonical_json_bytes(attempt, final_lf=True, sort_keys=False),
    )


def _write_artifact(output_root: Path, name: str, payload: Any) -> dict[str, Any]:
    path = output_root / name
    raw = canonical_json_bytes(payload, final_lf=True, sort_keys=False)
    write_atomic_once(path, raw)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise R20ToolError(f"materialized artifact is not regular: {path}")
    return {
        "filename": name,
        "length": int(info.st_size),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def execute_materialization(
    *,
    tool_manifest_path: Path,
    expected_tool_manifest_sha256: str,
) -> int:
    manifest = load_canonical_json(
        tool_manifest_path,
        expected_sha256=expected_tool_manifest_sha256,
        final_lf=True,
        sort_keys=True,
    )
    if not isinstance(manifest, dict):
        raise R20ToolError("tool manifest must be a JSON object")
    actual_manifest_sha256 = sha256_file(tool_manifest_path)
    runner = validate_tool_freeze_environment(
        manifest,
        verify_command_identities=True,
    )
    materialization = manifest.get("materialization")
    if not isinstance(materialization, dict):
        raise R20ToolError("materialization identity is absent")
    if materialization.get("attempt_id") != MATERIALIZATION_ATTEMPT_ID:
        raise R20ToolError("materialization attempt identity changed")
    workspace_root = Path(str(manifest["paths"]["workspace_root"]))
    if not workspace_root.is_absolute():
        raise R20ToolError("workspace root must be absolute")
    output_root = Path(str(materialization["output_root"]))
    if not output_root.is_absolute():
        output_root = workspace_root / output_root
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    expected_suffix = Path(
        "artifacts/synthetic_semantic_reachability/"
        "r20-content-materialization-attempt-001"
    )
    if output_root != workspace_root / expected_suffix:
        raise R20ToolError("materialization output root changed")

    # The create-once marker is the first materialization mutation and precedes
    # distribution scanning, content records, Mach-O roots and graph nodes.
    _create_attempt_root(
        output_root=output_root,
        workspace_root=workspace_root,
        manifest_sha256=actual_manifest_sha256,
        run_id=MATERIALIZATION_ATTEMPT_ID,
    )
    started_at = _aware_now()
    try:
        closure = materialize_runtime_closure(manifest, runner=runner)
        support_rows = manifest.get("support_rows")
        if not isinstance(support_rows, dict):
            raise R20ToolError("frozen support_rows payload is absent")
        support_root = hashlib.sha256(
            canonical_json_bytes(support_rows.get("rows"))
        ).hexdigest()
        if support_root != support_rows.get("support_root_sha256"):
            raise R20ToolError("frozen support-row root does not reconcile")

        runtime_payload = {
            "format_version": 1,
            "protocol_id": PROTOCOL_ID,
            "python_identity": closure["python_identity"],
            "project_component": closure["project_component"],
            "stdlib_component": closure["stdlib_component"],
            "runtime_distribution_set": closure["runtime_distribution_set"],
            "shared_cache_identity_sha256": closure["shared_cache"][
                "shared_cache_identity_sha256"
            ],
            "native_loader_graph_sha256": closure["native_loader_graph"][
                "native_loader_graph_sha256"
            ],
            "runtime_content_root_sha256": closure[
                "runtime_content_root_sha256"
            ],
        }
        test_payload = {
            "format_version": 1,
            "protocol_id": PROTOCOL_ID,
            "test_distribution_set": closure["test_distribution_set"],
            "pytest_contract": manifest["pytest_contract"],
        }
        native_payload = {
            "format_version": 1,
            "protocol_id": PROTOCOL_ID,
            "shared_cache": closure["shared_cache"],
            "native_loader_graph": closure["native_loader_graph"],
        }
        artifact_rows = [
            _write_artifact(output_root, "support_rows.json", support_rows),
            _write_artifact(
                output_root,
                "runtime_content_records.json",
                runtime_payload,
            ),
            _write_artifact(
                output_root,
                "test_content_records.json",
                test_payload,
            ),
            _write_artifact(
                output_root,
                "native_loader_graph.json",
                native_payload,
            ),
        ]
        summary = {
            "format_version": 1,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": MATERIALIZATION_ATTEMPT_ID,
            "tool_manifest_sha256": actual_manifest_sha256,
            "support_root_sha256": support_root,
            "runtime_content_root_sha256": closure[
                "runtime_content_root_sha256"
            ],
            "native_loader_graph_sha256": closure["native_loader_graph"][
                "native_loader_graph_sha256"
            ],
            "artifacts": artifact_rows,
            "status": "success",
        }
        artifact_rows.append(
            _write_artifact(
                output_root,
                "materialization_summary.json",
                summary,
            )
        )
        result = {
            "format_version": 1,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": MATERIALIZATION_ATTEMPT_ID,
            "mode": "runtime_content_materialization",
            "status": "success",
            "tool_manifest_sha256": actual_manifest_sha256,
            "started_at": started_at,
            "completed_at": _aware_now(),
            "outputs": artifact_rows,
        }
        write_atomic_once(
            output_root / "RESULT.json",
            canonical_json_bytes(result, final_lf=True, sort_keys=False),
        )
        return 0
    except BaseException as exc:
        failure = {
            "format_version": 1,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": MATERIALIZATION_ATTEMPT_ID,
            "mode": "runtime_content_materialization",
            "status": "failure",
            "tool_manifest_sha256": actual_manifest_sha256,
            "started_at": started_at,
            "completed_at": _aware_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            write_atomic_once(
                output_root / "RESULT.json",
                canonical_json_bytes(failure, final_lf=True, sort_keys=False),
            )
        except BaseException:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-manifest", required=True)
    parser.add_argument("--expected-tool-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    return execute_materialization(
        tool_manifest_path=Path(arguments.tool_manifest),
        expected_tool_manifest_sha256=arguments.expected_tool_manifest_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
