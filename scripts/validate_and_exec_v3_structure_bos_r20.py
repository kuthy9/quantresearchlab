#!/usr/bin/env python3
"""R20 stdlib-only validate-then-exec launcher.

Required invocation prefix:
    <frozen-python> -I -S -B validate_and_exec_v3_structure_bos_r20.py ...

The sibling runtime tool is hashed before it is imported.  No arbitrary child
arguments are accepted: the selected command and environment come from the
frozen manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


def _fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _regular_bytes(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        _fail(f"launcher input must be regular non-symlink: {path}")
    return path.read_bytes()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        _fail(f"launcher input must be regular non-symlink: {path}")
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = _regular_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail("launcher manifest SHA-256 changed")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=lambda value: _fail(f"invalid JSON constant: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("launcher manifest is malformed") from exc
    if not isinstance(payload, dict) or raw != _canonical(payload):
        _fail("launcher manifest is not canonical")
    return payload


def _load_verified_runtime_tool(
    *,
    path: Path,
    expected_sha256: str,
    manifest: dict[str, Any],
):
    if _sha256(path) != expected_sha256:
        _fail("runtime-tool SHA-256 changed")
    bindings = [
        value
        for value in manifest.get("tool_files", [])
        if value.get("path") == os.fspath(path)
    ]
    if len(bindings) != 1:
        _fail("runtime tool is not uniquely bound by the manifest")
    binding = bindings[0]
    if (
        binding.get("sha256") != expected_sha256
        or int(binding.get("length", -1)) != path.lstat().st_size
    ):
        _fail("runtime-tool manifest binding changed")
    spec = importlib.util.spec_from_file_location(
        "v3_structure_bos_runtime_tool_r20_verified",
        path,
    )
    if spec is None or spec.loader is None:
        _fail("cannot create verified runtime-tool import specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_launcher_flags() -> None:
    if not sys.flags.isolated:
        _fail("R20 launcher requires Python -I")
    if not sys.flags.no_site:
        _fail("R20 launcher requires Python -S")
    if not sys.flags.dont_write_bytecode:
        _fail("R20 launcher requires Python -B")


def _validated_command(manifest: dict[str, Any], command_id: str) -> tuple[str, list[str], str, dict[str, str]]:
    commands = manifest.get("commands")
    if not isinstance(commands, dict) or command_id not in commands:
        _fail("frozen child command is absent")
    command = commands[command_id]
    if not isinstance(command, dict):
        _fail("frozen child command is malformed")
    executable = str(command.get("executable", ""))
    argv = command.get("argv")
    cwd = str(command.get("cwd", ""))
    environment = command.get("environment")
    if not os.path.isabs(executable) or not isinstance(argv, list) or not argv:
        _fail("frozen child executable/argv is invalid")
    if argv[0] != executable or not all(isinstance(value, str) for value in argv):
        _fail("frozen child argv is invalid")
    if not os.path.isabs(cwd):
        _fail("frozen child working directory is invalid")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        _fail("frozen child environment is invalid")
    required = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONSTARTUP": "",
    }
    for key, value in required.items():
        if environment.get(key) != value:
            _fail(f"frozen child environment changed: {key}")
    if any(key.startswith("DYLD_") for key in environment):
        _fail("unregistered DYLD environment is forbidden")
    if environment.get("PATH", "") != "":
        _fail("child PATH must be empty")
    cache_prefix = environment.get("PYTHONPYCACHEPREFIX")
    if not cache_prefix or not os.path.isabs(cache_prefix):
        _fail("isolated bytecode-cache prefix is absent")
    cache_path = Path(cache_prefix)
    if cache_path.exists() or cache_path.is_symlink():
        _fail("isolated bytecode-cache prefix must remain absent")
    if executable not in {
        "~/miniconda3/bin/python",
        "~/miniconda3/bin/python3.12",
    }:
        _fail("child executable is not the frozen Python")
    if "-s" not in argv or "-B" not in argv:
        _fail("child Python command must use -s -B")
    return executable, list(argv), cwd, dict(environment)


def execute(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    runtime_tool_path: Path,
    expected_runtime_tool_sha256: str,
    command_id: str,
) -> None:
    _require_launcher_flags()
    manifest = _load_manifest(manifest_path, expected_manifest_sha256)
    kind = manifest.get("manifest_kind")
    if kind == "r20_tool_freeze":
        tool_manifest = manifest
    elif kind == "r20_final_freeze":
        tool_path = Path(str(manifest.get("tool_manifest_path", "")))
        tool_sha = str(manifest.get("tool_manifest_sha256", ""))
        tool_manifest = _load_manifest(tool_path, tool_sha)
    else:
        _fail("unsupported R20 manifest kind")
    runtime = _load_verified_runtime_tool(
        path=runtime_tool_path,
        expected_sha256=expected_runtime_tool_sha256,
        manifest=tool_manifest,
    )
    runtime.validate_tool_freeze_environment(
        tool_manifest,
        verify_command_identities=True,
    )
    if kind == "r20_final_freeze":
        runtime.validate_materialization_bundle(
            output_root=str(manifest["materialization_output_root"]),
            tool_manifest=tool_manifest,
            expected_result_sha256=str(
                manifest["materialization_result_sha256"]
            ),
        )
    executable, argv, cwd, environment = _validated_command(manifest, command_id)
    runtime.validate_python_identity()
    os.chdir(cwd)
    os.execve(executable, argv, environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--runtime-tool", required=True)
    parser.add_argument("--expected-runtime-tool-sha256", required=True)
    parser.add_argument("--command-id", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    execute(
        manifest_path=Path(arguments.manifest),
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        runtime_tool_path=Path(arguments.runtime_tool),
        expected_runtime_tool_sha256=arguments.expected_runtime_tool_sha256,
        command_id=arguments.command_id,
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
