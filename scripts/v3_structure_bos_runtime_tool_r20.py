#!/usr/bin/env python3
"""Fail-closed, stdlib-only runtime/native closure tooling for EXP001 R20.

This module deliberately does not import project or third-party packages.  It
is a Stage-A candidate only: execution requires a separately frozen tool
manifest and authority record.
"""
from __future__ import annotations

import csv
import email.parser
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import stat
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = (
    "EXP-SMC-3.0.1-001-REGISTERED-SUPPORT-LIFECYCLE-REACHABILITY-R20"
)
MATERIALIZATION_ATTEMPT_ID = (
    "EXP-SMC-3.0.1-001-R20-CONTENT-MATERIALIZATION-ATTEMPT-001"
)
PYTHON_ENTRY = Path("~/miniconda3/bin/python")
PYTHON_LINK_TARGET = "python3.12"
PYTHON_FINAL = Path("~/miniconda3/bin/python3.12")
PYTHON_FINAL_LENGTH = 6_887_816
PYTHON_FINAL_SHA256 = (
    "3a80e322390c8df16971c26240781f2f881f25b09fb30a88c90908fe89df413b"
)
X86_CACHE_BASE = Path(
    "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/"
    "dyld_shared_cache_x86_64"
)

INSPECTION_TOOL_PATHS = (
    "/usr/bin/arch",
    "/usr/bin/otool",
    "/usr/bin/dyld_info",
    "/usr/bin/codesign",
    "/usr/sbin/diskutil",
    "/usr/bin/sw_vers",
    "/usr/bin/uname",
)
ARCH_SHA256 = (
    "c6d529462664161cc34751f592c961ef1f0990775cb1d3af8a98b9c5f28a0bce"
)
ARCH_FULL_CDHASH = (
    "31fd0f1885eb34464866260d2b52c5e59e14ee714b694a0f9c4c9add4fc53d3f"
)

RUNTIME_DISTRIBUTIONS = (
    ("exchange-calendars", "4.11"),
    ("korean-lunar-calendar", "0.3.1"),
    ("numpy", "1.26.4"),
    ("pandas", "2.1.3"),
    ("pyluach", "2.2.0"),
    ("python-dateutil", "2.9.0.post0"),
    ("pytz", "2024.2"),
    ("six", "1.17.0"),
    ("toolz", "1.0.0"),
    ("tzdata", "2025.1"),
)
TEST_DISTRIBUTIONS = (
    ("iniconfig", "2.1.0"),
    ("packaging", "24.1"),
    ("pluggy", "1.6.0"),
    ("pygments", "2.19.1"),
    ("pytest", "8.4.1"),
)

MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce": (">", False, False),
    b"\xce\xfa\xed\xfe": ("<", False, False),
    b"\xfe\xed\xfa\xcf": (">", True, False),
    b"\xcf\xfa\xed\xfe": ("<", True, False),
    b"\xca\xfe\xba\xbe": (">", False, True),
    b"\xbe\xba\xfe\xca": ("<", False, True),
    b"\xca\xfe\xba\xbf": (">", True, True),
    b"\xbf\xba\xfe\xca": ("<", True, True),
}
DYLIB_LOAD_COMMANDS = frozenset(
    {
        "LC_LOAD_DYLIB",
        "LC_LOAD_WEAK_DYLIB",
        "LC_REEXPORT_DYLIB",
        "LC_LOAD_UPWARD_DYLIB",
        "LC_LAZY_LOAD_DYLIB",
    }
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"(?i)\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})\b"
)


class R20ToolError(RuntimeError):
    """Any ambiguity or unsupported input fails the frozen candidate closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise R20ToolError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def canonical_json_bytes(
    payload: Any,
    *,
    final_lf: bool = False,
    sort_keys: bool = True,
) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def load_canonical_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    final_lf: bool = True,
    sort_keys: bool = True,
) -> Any:
    source = Path(path)
    raw = read_regular_bytes(source)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise R20ToolError(f"JSON SHA-256 changed: {source}")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                R20ToolError(f"invalid JSON constant: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise R20ToolError(f"non-UTF-8 JSON: {source}") from exc
    except json.JSONDecodeError as exc:
        raise R20ToolError(f"malformed JSON: {source}") from exc
    if raw != canonical_json_bytes(payload, final_lf=final_lf, sort_keys=sort_keys):
        raise R20ToolError(f"noncanonical JSON: {source}")
    return payload


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise R20ToolError(f"expected regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_regular_bytes(path: str | Path) -> bytes:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise R20ToolError(f"expected regular non-symlink file: {source}")
    return source.read_bytes()


def regular_file_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    source = Path(path)
    anchor = Path(relative_to)
    relative = lexical_relative(source, anchor)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise R20ToolError(f"expected regular file: {source}")
    return {
        "path": relative,
        "kind": "regular",
        "length": int(info.st_size),
        "sha256": sha256_file(source),
    }


def lexical_relative(path: str | Path, anchor: str | Path) -> str:
    source = Path(os.path.abspath(os.fspath(path)))
    root = Path(os.path.abspath(os.fspath(anchor)))
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise R20ToolError(f"path escapes anchor: {source} not under {root}") from exc
    value = relative.as_posix()
    if value in {"", "."} or value.startswith("../") or "/../" in value:
        raise R20ToolError(f"invalid relative path: {value}")
    return value


def resolve_symlink_chain(
    path: str | Path,
    *,
    anchor: str | Path | None,
    require_regular: bool = True,
) -> tuple[list[dict[str, str]], Path]:
    current = Path(os.path.abspath(os.fspath(path)))
    root = None if anchor is None else Path(os.path.abspath(os.fspath(anchor)))
    seen: set[str] = set()
    chain: list[dict[str, str]] = []
    for _ in range(64):
        key = os.fspath(current)
        if key in seen:
            raise R20ToolError(f"symlink cycle: {path}")
        seen.add(key)
        if root is not None:
            lexical_relative(current, root)
        info = current.lstat()
        if not stat.S_ISLNK(info.st_mode):
            if require_regular and not stat.S_ISREG(info.st_mode):
                raise R20ToolError(f"symlink did not resolve to regular file: {current}")
            return chain, current
        literal = os.readlink(current)
        target = Path(literal)
        if not target.is_absolute():
            target = current.parent / target
        target = Path(os.path.abspath(os.path.normpath(os.fspath(target))))
        if root is not None:
            lexical_relative(target, root)
        chain.append(
            {
                "path": os.fspath(current),
                "literal_target": literal,
                "next_path": os.fspath(target),
            }
        )
        current = target
    raise R20ToolError(f"symlink chain exceeds 64 hops: {path}")


def validate_python_identity() -> dict[str, Any]:
    entry_info = PYTHON_ENTRY.lstat()
    if not stat.S_ISLNK(entry_info.st_mode):
        raise R20ToolError("registered Python entry is not the frozen symlink")
    if os.readlink(PYTHON_ENTRY) != PYTHON_LINK_TARGET:
        raise R20ToolError("registered Python literal symlink target changed")
    chain, final = resolve_symlink_chain(
        PYTHON_ENTRY,
        anchor=PYTHON_ENTRY.parent,
        require_regular=True,
    )
    if len(chain) != 1 or final != PYTHON_FINAL:
        raise R20ToolError("registered Python symlink chain changed")
    info = final.lstat()
    if info.st_size != PYTHON_FINAL_LENGTH or sha256_file(final) != PYTHON_FINAL_SHA256:
        raise R20ToolError("registered Python final executable identity changed")
    slices = macho_slices(final)
    if [value["architecture"] for value in slices] != ["x86_64"]:
        raise R20ToolError("registered Python must be x86_64-only")
    if Path(os.path.realpath(sys.executable)) != PYTHON_FINAL:
        raise R20ToolError("current interpreter is not the registered executable")
    if platform.machine() != "x86_64":
        raise R20ToolError("registered Python runtime machine is not x86_64")
    if sys.implementation.name != "cpython" or sys.implementation.cache_tag != "cpython-312":
        raise R20ToolError("registered Python implementation/cache tag changed")
    return {
        "entry_path": os.fspath(PYTHON_ENTRY),
        "entry_kind": "symlink",
        "entry_literal_target": PYTHON_LINK_TARGET,
        "final_path": os.fspath(PYTHON_FINAL),
        "final_kind": "regular",
        "final_length": PYTHON_FINAL_LENGTH,
        "final_sha256": PYTHON_FINAL_SHA256,
        "mach_o_architectures": ["x86_64"],
        "runtime_machine": "x86_64",
        "implementation": "cpython",
        "cache_tag": "cpython-312",
    }


def _excluded(
    path: Path,
    *,
    excluded_prefixes: Sequence[Path],
    exclude_cache: bool,
) -> bool:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for prefix in excluded_prefixes:
        root = Path(os.path.abspath(os.fspath(prefix)))
        try:
            absolute.relative_to(root)
            return True
        except ValueError:
            pass
    if exclude_cache:
        if "__pycache__" in absolute.parts or absolute.suffix in {".pyc", ".pyo"}:
            return True
    return False


def _walk_paths(root: Path) -> Iterable[Path]:
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode):
        yield root
        return
    if not stat.S_ISDIR(info.st_mode):
        raise R20ToolError(f"unsupported filesystem kind: {root}")
    entries = sorted(os.scandir(root), key=lambda item: os.fsencode(item.name))
    for entry in entries:
        path = Path(entry.path)
        child = path.lstat()
        if stat.S_ISDIR(child.st_mode) and not stat.S_ISLNK(child.st_mode):
            yield from _walk_paths(path)
        else:
            yield path


def build_content_component(
    *,
    name: str,
    anchor: str | Path,
    roots: Sequence[str | Path],
    excluded_prefixes: Sequence[str | Path] = (),
    exclude_cache: bool = True,
    allow_file_symlinks: bool = False,
    absent_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root_path = Path(os.path.abspath(os.fspath(anchor)))
    if not stat.S_ISDIR(root_path.lstat().st_mode):
        raise R20ToolError(f"component anchor is not a directory: {root_path}")
    normalized_roots = sorted(
        {lexical_relative(value, root_path) for value in roots},
        key=lambda value: value.encode("utf-8"),
    )
    exclusions = tuple(Path(os.path.abspath(os.fspath(value))) for value in excluded_prefixes)
    records_by_path: dict[str, dict[str, Any]] = {}
    pending_regular_targets: list[Path] = []
    for relative_root in normalized_roots:
        selected = root_path / relative_root
        for source in _walk_paths(selected):
            if _excluded(source, excluded_prefixes=exclusions, exclude_cache=exclude_cache):
                continue
            relative = lexical_relative(source, root_path)
            info = source.lstat()
            if stat.S_ISREG(info.st_mode):
                records_by_path[relative] = regular_file_record(source, relative_to=root_path)
            elif stat.S_ISLNK(info.st_mode):
                if not allow_file_symlinks:
                    raise R20ToolError(f"symlink forbidden in component {name}: {source}")
                chain, final = resolve_symlink_chain(source, anchor=root_path)
                if len(chain) != 1:
                    raise R20ToolError(
                        f"each content symlink must be explicitly recorded: {source}"
                    )
                records_by_path[relative] = {
                    "path": relative,
                    "kind": "symlink",
                    "literal_target": chain[0]["literal_target"],
                    "resolved_path": lexical_relative(final, root_path),
                }
                pending_regular_targets.append(final)
            else:
                raise R20ToolError(f"nonregular content forbidden: {source}")
    for target in pending_regular_targets:
        relative = lexical_relative(target, root_path)
        records_by_path.setdefault(
            relative,
            regular_file_record(target, relative_to=root_path),
        )
    absent = sorted(
        {lexical_relative(value, root_path) for value in absent_paths},
        key=lambda value: value.encode("utf-8"),
    )
    for relative in absent:
        if (root_path / relative).exists() or (root_path / relative).is_symlink():
            raise R20ToolError(f"path registered absent is present: {relative}")
    records = [
        records_by_path[key]
        for key in sorted(records_by_path, key=lambda value: value.encode("utf-8"))
    ]
    record_root = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    return {
        "name": name,
        "anchor": os.fspath(root_path),
        "roots": normalized_roots,
        "excluded_prefixes": sorted(
            {lexical_relative(value, root_path) for value in exclusions},
            key=lambda value: value.encode("utf-8"),
        ),
        "exclude_cache": bool(exclude_cache),
        "allow_file_symlinks": bool(allow_file_symlinks),
        "absent_paths": absent,
        "records": records,
        "content_root_sha256": record_root,
    }


def validate_content_component(expected: Mapping[str, Any]) -> None:
    rebuilt = build_content_component(
        name=str(expected["name"]),
        anchor=str(expected["anchor"]),
        roots=[Path(str(expected["anchor"])) / str(value) for value in expected["roots"]],
        excluded_prefixes=[
            Path(str(expected["anchor"])) / str(value)
            for value in expected["excluded_prefixes"]
        ],
        exclude_cache=bool(expected["exclude_cache"]),
        allow_file_symlinks=bool(expected["allow_file_symlinks"]),
        absent_paths=[
            Path(str(expected["anchor"])) / str(value)
            for value in expected["absent_paths"]
        ],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(expected)):
        raise R20ToolError(f"content component changed: {expected.get('name')}")


def normalize_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized):
        raise R20ToolError(f"invalid distribution name: {value}")
    return normalized


def _metadata_identity(raw: bytes) -> tuple[str, str, list[str]]:
    try:
        message = email.parser.BytesParser().parsebytes(raw)
    except Exception as exc:
        raise R20ToolError("malformed distribution METADATA") from exc
    name = normalize_distribution_name(str(message.get("Name", "")))
    version = str(message.get("Version", ""))
    if not version:
        raise R20ToolError(f"distribution version absent: {name}")
    requires = [str(value) for value in message.get_all("Requires-Dist", [])]
    return name, version, requires


def _find_distribution(
    site_packages: Sequence[Path],
    *,
    expected_name: str,
    expected_version: str,
) -> tuple[Path, bytes, list[str]]:
    matches: list[tuple[Path, bytes, list[str]]] = []
    for site in site_packages:
        if not stat.S_ISDIR(site.lstat().st_mode):
            raise R20ToolError(f"site-packages is not a directory: {site}")
        for entry in sorted(os.scandir(site), key=lambda item: os.fsencode(item.name)):
            if not (entry.name.endswith(".dist-info") or entry.name.endswith(".egg-info")):
                continue
            info_path = Path(entry.path)
            if stat.S_ISLNK(info_path.lstat().st_mode):
                raise R20ToolError(f"symlinked distribution metadata: {info_path}")
            metadata = info_path / "METADATA"
            if not metadata.exists():
                continue
            raw = read_regular_bytes(metadata)
            name, version, requires = _metadata_identity(raw)
            if name == expected_name and version == expected_version:
                matches.append((info_path, raw, requires))
    if len(matches) != 1:
        raise R20ToolError(
            f"expected exactly one {expected_name}=={expected_version}, got {len(matches)}"
        )
    return matches[0]


def _record_paths(info_path: Path, *, prefix: Path) -> tuple[list[Path], list[Path]]:
    record_file = info_path / "RECORD"
    if not record_file.exists():
        raise R20ToolError(f"distribution RECORD absent: {info_path}")
    declared: list[Path] = []
    with record_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3 or not row[0]:
                raise R20ToolError(f"malformed RECORD row: {record_file}")
            candidate = Path(
                os.path.abspath(os.path.normpath(os.fspath(info_path.parent / row[0])))
            )
            lexical_relative(candidate, prefix)
            if not candidate.exists() and not candidate.is_symlink():
                raise R20ToolError(f"declared distribution file missing: {candidate}")
            declared.append(candidate)
    roots: set[Path] = {info_path}
    explicit: set[Path] = set()
    site = info_path.parent
    for candidate in declared:
        try:
            relative = candidate.relative_to(site)
        except ValueError:
            explicit.add(candidate)
            continue
        if not relative.parts or relative.parts[0] == info_path.name:
            continue
        top = site / relative.parts[0]
        roots.add(top)
    return sorted(roots, key=lambda value: os.fsencode(value)), sorted(
        explicit, key=lambda value: os.fsencode(value)
    )


def materialize_distribution_set(
    *,
    set_name: str,
    expected: Sequence[Mapping[str, Any]],
    registered: Sequence[tuple[str, str]],
    prefix: str | Path,
    site_packages: Sequence[str | Path],
) -> dict[str, Any]:
    exact = [(str(value["name"]), str(value["version"])) for value in expected]
    if exact != list(registered):
        raise R20ToolError(f"{set_name} distribution set differs from preregistration")
    prefix_path = Path(os.path.abspath(os.fspath(prefix)))
    sites = [Path(os.path.abspath(os.fspath(value))) for value in site_packages]
    components: list[dict[str, Any]] = []
    owners: dict[str, str] = {}
    metadata_rows: list[dict[str, Any]] = []
    for expected_row in expected:
        name = str(expected_row["name"])
        version = str(expected_row["version"])
        info_path, metadata_raw, requires = _find_distribution(
            sites,
            expected_name=name,
            expected_version=version,
        )
        metadata_hash = hashlib.sha256(metadata_raw).hexdigest()
        if metadata_hash != str(expected_row["metadata_sha256"]):
            raise R20ToolError(f"raw METADATA changed: {name}")
        roots, explicit = _record_paths(info_path, prefix=prefix_path)
        component = build_content_component(
            name=f"{set_name}:{name}",
            anchor=prefix_path,
            roots=[*roots, *explicit],
            exclude_cache=True,
            allow_file_symlinks=False,
        )
        for record in component["records"]:
            prior = owners.get(record["path"])
            if prior is not None and prior != name:
                raise R20ToolError(
                    f"overlapping distribution ownership without shared-root record: "
                    f"{record['path']} ({prior}, {name})"
                )
            owners[record["path"]] = name
        components.append(component)
        metadata_rows.append(
            {
                "name": name,
                "version": version,
                "metadata_path": lexical_relative(info_path / "METADATA", prefix_path),
                "metadata_sha256": metadata_hash,
                "requires_dist": requires,
            }
        )
    aggregate = [
        {"name": value["name"], "content_root_sha256": value["content_root_sha256"]}
        for value in components
    ]
    return {
        "set_name": set_name,
        "distributions": metadata_rows,
        "components": components,
        "aggregate_root_sha256": hashlib.sha256(
            canonical_json_bytes(aggregate)
        ).hexdigest(),
    }


def build_stdlib_component(
    *,
    stdlib_root: str | Path,
    site_packages: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(stdlib_root)))
    return build_content_component(
        name="python-stdlib",
        anchor=root,
        roots=[root],
        excluded_prefixes=site_packages,
        exclude_cache=True,
        allow_file_symlinks=False,
    )


def _architecture_name(cpu_type: int, cpu_subtype: int) -> str:
    unsigned_type = cpu_type & 0xFFFFFFFF
    subtype = cpu_subtype & 0x00FFFFFF
    if unsigned_type == 0x01000007:
        return "x86_64h" if subtype == 8 else "x86_64"
    if unsigned_type == 0x0100000C:
        return "arm64e" if subtype == 2 else "arm64"
    if unsigned_type == 7:
        return "i386"
    if unsigned_type == 12:
        return "arm"
    return f"cpu-{unsigned_type:08x}-subtype-{subtype:08x}"


def macho_slices(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise R20ToolError(f"Mach-O input is not regular: {source}")
    with source.open("rb") as handle:
        prefix = handle.read(8)
    if len(prefix) < 8 or prefix[:4] not in MACHO_MAGICS:
        raise R20ToolError(f"not a supported Mach-O file: {source}")
    endian, is_64, is_fat = MACHO_MAGICS[prefix[:4]]
    output: list[dict[str, Any]] = []
    if not is_fat:
        with source.open("rb") as handle:
            header = handle.read(12)
        if len(header) < 12:
            raise R20ToolError(f"truncated Mach-O header: {source}")
        cpu_type, cpu_subtype = struct.unpack(f"{endian}ii", header[4:12])
        output.append(
            {
                "architecture": _architecture_name(cpu_type, cpu_subtype),
                "cpu_type": int(cpu_type & 0xFFFFFFFF),
                "cpu_subtype": int(cpu_subtype & 0xFFFFFFFF),
                "offset": 0,
                "size": int(info.st_size),
            }
        )
        return output
    count = struct.unpack(f"{endian}I", prefix[4:8])[0]
    if count < 1 or count > 64:
        raise R20ToolError(f"invalid fat Mach-O slice count: {source}")
    entry_size = 32 if is_64 else 20
    required = 8 + count * entry_size
    with source.open("rb") as handle:
        header = handle.read(required)
    if len(header) < required:
        raise R20ToolError(f"truncated fat Mach-O header: {source}")
    for index in range(count):
        offset = 8 + index * entry_size
        if is_64:
            cpu_type, cpu_subtype, slice_offset, size, _align, _reserved = struct.unpack(
                f"{endian}iiQQII", header[offset : offset + entry_size]
            )
        else:
            cpu_type, cpu_subtype, slice_offset, size, _align = struct.unpack(
                f"{endian}iiIII", header[offset : offset + entry_size]
            )
        if size < 1 or slice_offset + size > info.st_size:
            raise R20ToolError(f"fat Mach-O slice escapes file: {source}")
        output.append(
            {
                "architecture": _architecture_name(cpu_type, cpu_subtype),
                "cpu_type": int(cpu_type & 0xFFFFFFFF),
                "cpu_subtype": int(cpu_subtype & 0xFFFFFFFF),
                "offset": int(slice_offset),
                "size": int(size),
            }
        )
    names = [value["architecture"] for value in output]
    if len(names) != len(set(names)):
        raise R20ToolError(f"duplicate Mach-O architecture slices: {source}")
    return output


def is_macho(path: str | Path) -> bool:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size < 4:
        return False
    with source.open("rb") as handle:
        return handle.read(4) in MACHO_MAGICS


def parse_codesign_identity(text: str) -> dict[str, Any]:
    cdhashes: list[str] = []
    authorities: list[str] = []
    identifier: str | None = None
    team_identifier: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("CDHash="):
            value = line.split("=", 1)[1].lower()
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
                raise R20ToolError("unsupported codesign CDHash")
            cdhashes.append(value)
        elif line.startswith("CandidateCDHashFull ") and "=" in line:
            value = line.rsplit("=", 1)[1].lower()
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise R20ToolError("unsupported codesign full CDHash")
            cdhashes.append(value)
        elif line.startswith("Authority="):
            authorities.append(line.split("=", 1)[1])
        elif line.startswith("Identifier="):
            identifier = line.split("=", 1)[1]
        elif line.startswith("TeamIdentifier="):
            team_identifier = line.split("=", 1)[1]
        elif any(
            line.startswith(prefix)
            for prefix in (
                "Executable=",
                "Executable Segment ",
                "Format=",
                "CodeDirectory ",
                "Hash choices=",
                "Hash type=",
                "Hash size=",
                "CMSDigest=",
                "CMSDigestType=",
                "CandidateCDHash ",
                "Signature=",
                "Signature size=",
                "Timestamp=",
                "Info.plist=",
                "Info.plist entries=",
                "Sealed Resources=",
                "Internal requirements ",
                "Page size=",
                "Platform identifier=",
                "Runtime Version=",
                "Notarization Ticket=",
            )
        ):
            continue
        else:
            raise R20ToolError(f"unknown codesign output line: {line}")
    if not cdhashes or not authorities or identifier is None:
        raise R20ToolError("incomplete codesign identity")
    return {
        "cdhashes": cdhashes,
        "authorities": authorities,
        "identifier": identifier,
        "team_identifier": team_identifier,
    }


class FrozenCommandRunner:
    """Run only absolute, hash-bound Apple inspection commands."""

    def __init__(self, bindings: Mapping[str, Mapping[str, Any]]) -> None:
        if set(bindings) != set(INSPECTION_TOOL_PATHS):
            raise R20ToolError("inspection-tool path set changed")
        self.bindings = {str(key): dict(value) for key, value in bindings.items()}
        self._files_verified = False
        self._identities_verified = False

    def verify_files(self) -> None:
        for path, expected in self.bindings.items():
            source = Path(path)
            info = source.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise R20ToolError(f"inspection tool is not regular: {source}")
            if int(expected["length"]) != info.st_size:
                raise R20ToolError(f"inspection tool length changed: {source}")
            if str(expected["sha256"]) != sha256_file(source):
                raise R20ToolError(f"inspection tool hash changed: {source}")
        arch = self.bindings["/usr/bin/arch"]
        if arch["sha256"] != ARCH_SHA256:
            raise R20ToolError("registered /usr/bin/arch hash changed")
        if ARCH_FULL_CDHASH not in arch["cdhashes"]:
            raise R20ToolError("registered /usr/bin/arch CDHash changed")
        self._files_verified = True

    def _run_verified_file(
        self,
        executable: str,
        arguments: Sequence[str],
    ) -> tuple[str, str]:
        if not self._files_verified or executable not in self.bindings:
            raise R20ToolError(f"unverified inspection command: {executable}")
        environment = {
            "PATH": "",
            "LANG": "C",
            "LC_ALL": "C",
            "TMPDIR": "/tmp",
        }
        completed = subprocess.run(
            [executable, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            close_fds=True,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
        if completed.returncode != 0:
            raise R20ToolError(
                f"inspection command failed ({completed.returncode}): "
                f"{executable} {' '.join(arguments)}"
            )
        return completed.stdout, completed.stderr

    def verify_identities(self) -> None:
        if not self._files_verified:
            self.verify_files()
        for path, expected in self.bindings.items():
            stdout, stderr = self._run_verified_file(
                "/usr/bin/codesign",
                ["-d", "--verbose=6", path],
            )
            if stdout.strip():
                raise R20ToolError("codesign unexpectedly wrote stdout")
            actual = parse_codesign_identity(stderr)
            if actual["cdhashes"] != list(expected["cdhashes"]):
                raise R20ToolError(f"inspection-tool CDHash changed: {path}")
            if actual["authorities"] != list(expected["authorities"]):
                raise R20ToolError(f"inspection-tool authority changed: {path}")
            if actual["identifier"] != str(expected["identifier"]):
                raise R20ToolError(f"inspection-tool identifier changed: {path}")
        self._identities_verified = True

    def run(self, executable: str, arguments: Sequence[str]) -> tuple[str, str]:
        if not self._identities_verified:
            raise R20ToolError("inspection-tool identities not fully verified")
        return self._run_verified_file(executable, arguments)

    def run_x86_dyld_info(self, arguments: Sequence[str]) -> tuple[str, str]:
        return self.run(
            "/usr/bin/arch",
            ["-x86_64", "/usr/bin/dyld_info", *arguments],
        )

    def inspect_codesign_target(self, path: str | Path) -> dict[str, Any]:
        stdout, stderr = self.run(
            "/usr/bin/codesign",
            ["-d", "--verbose=6", os.fspath(path)],
        )
        if stdout.strip():
            raise R20ToolError("codesign target inspection unexpectedly wrote stdout")
        return parse_codesign_identity(stderr)


def verify_host_and_translation(runner: FrozenCommandRunner) -> dict[str, str]:
    native_stdout, native_stderr = runner.run(
        "/usr/bin/arch",
        ["-arm64", "/usr/bin/uname", "-m"],
    )
    x86_stdout, x86_stderr = runner.run(
        "/usr/bin/arch",
        ["-x86_64", "/usr/bin/uname", "-m"],
    )
    if native_stderr.strip() or x86_stderr.strip():
        raise R20ToolError("architecture probe wrote unexpected stderr")
    if native_stdout.strip() != "arm64":
        raise R20ToolError("physical host arm64 probe failed")
    if x86_stdout.strip() != "x86_64":
        raise R20ToolError("x86_64/Rosetta probe failed")
    return {
        "physical_host_architecture": "arm64",
        "registered_runtime_architecture": "x86_64",
        "execution_mode": "x86_64-under-rosetta",
    }


def parse_otool_load_commands(text: str) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if re.fullmatch(r"Load command [0-9]+", line.strip()):
            if current is not None:
                commands.append(_parse_otool_command(current))
            current = []
            continue
        if current is None:
            if line.endswith(":") or "(architecture " in line:
                continue
            raise R20ToolError(f"unknown otool preamble line: {line}")
        current.append(line.strip())
    if current is not None:
        commands.append(_parse_otool_command(current))
    if not commands:
        raise R20ToolError("otool emitted no load commands")
    uuids = [value["uuid"] for value in commands if value["cmd"] == "LC_UUID"]
    if len(uuids) != 1:
        raise R20ToolError("Mach-O must expose exactly one LC_UUID")
    return {
        "uuid": uuids[0],
        "rpaths": [
            value["path"] for value in commands if value["cmd"] == "LC_RPATH"
        ],
        "dependencies": [
            {"kind": value["cmd"], "install_name": value["name"]}
            for value in commands
            if value["cmd"] in DYLIB_LOAD_COMMANDS
        ],
        "load_commands": commands,
    }


def _parse_otool_command(lines: Sequence[str]) -> dict[str, Any]:
    field_rows: list[list[str]] = []
    values_by_key: dict[str, list[str]] = {}
    for line in lines:
        if line == "Section":
            field_rows.append(["_marker", "Section"])
            continue
        if " " not in line:
            raise R20ToolError(f"unknown otool load-command line: {line}")
        key, value = line.split(None, 1)
        field_rows.append([key, value])
        values_by_key.setdefault(key, []).append(value)
    commands = values_by_key.get("cmd", [])
    command = commands[0] if len(commands) == 1 else None
    if not command or not command.startswith("LC_"):
        raise R20ToolError("otool load command kind absent")
    output: dict[str, Any] = {
        "cmd": command,
        "fields": field_rows,
    }
    if command == "LC_UUID":
        uuid_values = values_by_key.get("uuid", [])
        match = UUID_RE.search(uuid_values[0]) if len(uuid_values) == 1 else None
        if match is None:
            raise R20ToolError("LC_UUID value missing")
        output["uuid"] = match.group(1).lower()
    if command == "LC_RPATH":
        path_values = values_by_key.get("path", [])
        path = (
            path_values[0].split(" (offset ", 1)[0]
            if len(path_values) == 1
            else ""
        )
        if not path:
            raise R20ToolError("LC_RPATH value missing")
        output["path"] = path
    if command in DYLIB_LOAD_COMMANDS:
        name_values = values_by_key.get("name", [])
        name = (
            name_values[0].split(" (offset ", 1)[0]
            if len(name_values) == 1
            else ""
        )
        if not name:
            raise R20ToolError(f"{command} install name missing")
        output["name"] = name
    return output


def parse_dyld_uuid(text: str) -> str:
    matches = {match.group(1).lower() for match in UUID_RE.finditer(text)}
    if len(matches) != 1:
        raise R20ToolError("dyld_info UUID output is unsupported or ambiguous")
    return next(iter(matches))


def inspect_macho_x86(path: str | Path, runner: FrozenCommandRunner) -> dict[str, Any]:
    source = Path(os.path.abspath(os.fspath(path)))
    slices = macho_slices(source)
    names = [value["architecture"] for value in slices]
    if "x86_64" not in names or "x86_64h" in names:
        raise R20ToolError(f"Mach-O lacks the exact registered x86_64 slice: {source}")
    stdout, stderr = runner.run(
        "/usr/bin/otool",
        ["-arch", "x86_64", "-l", os.fspath(source)],
    )
    if stderr.strip():
        raise R20ToolError("otool emitted unexpected stderr")
    parsed = parse_otool_load_commands(stdout)
    uuid_out, uuid_err = runner.run_x86_dyld_info(
        ["-arch", "x86_64", "-uuid", os.fspath(source)]
    )
    if uuid_err.strip():
        raise R20ToolError("x86_64 dyld_info UUID emitted unexpected stderr")
    if parse_dyld_uuid(uuid_out) != parsed["uuid"]:
        raise R20ToolError("otool and x86_64 dyld_info UUID mismatch")
    return {
        "path": os.fspath(source),
        "length": int(source.lstat().st_size),
        "sha256": sha256_file(source),
        "architecture": "x86_64",
        "uuid": parsed["uuid"],
        "rpaths": parsed["rpaths"],
        "dependencies": parsed["dependencies"],
        "load_commands": parsed["load_commands"],
    }


def _expand_rpath_token(
    value: str,
    *,
    loader: Path,
    executable: Path,
) -> Path:
    if value.startswith("@loader_path"):
        suffix = value[len("@loader_path") :].lstrip("/")
        return Path(os.path.abspath(os.path.normpath(os.fspath(loader.parent / suffix))))
    if value.startswith("@executable_path"):
        suffix = value[len("@executable_path") :].lstrip("/")
        return Path(
            os.path.abspath(os.path.normpath(os.fspath(executable.parent / suffix)))
        )
    if value.startswith("@"):
        raise R20ToolError(f"unsupported recursive rpath token: {value}")
    if not os.path.isabs(value):
        raise R20ToolError(f"relative LC_RPATH is unsupported: {value}")
    return Path(os.path.abspath(os.path.normpath(value)))


def resolve_install_name(
    install_name: str,
    *,
    loader: Path,
    executable: Path,
    image_rpaths: Sequence[str],
    inherited_rpaths: Sequence[str],
    shared_cache_images: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[Path] = []
    if install_name.startswith("@loader_path"):
        suffix = install_name[len("@loader_path") :].lstrip("/")
        candidates.append(loader.parent / suffix)
    elif install_name.startswith("@executable_path"):
        suffix = install_name[len("@executable_path") :].lstrip("/")
        candidates.append(executable.parent / suffix)
    elif install_name.startswith("@rpath/"):
        suffix = install_name[len("@rpath/") :]
        for raw_rpath in [*image_rpaths, *inherited_rpaths]:
            candidates.append(
                _expand_rpath_token(
                    raw_rpath,
                    loader=loader,
                    executable=executable,
                )
                / suffix
            )
    elif os.path.isabs(install_name):
        candidates.append(Path(install_name))
    else:
        raise R20ToolError(f"unsupported Mach-O install name: {install_name}")
    normalized: list[Path] = []
    for candidate in candidates:
        value = Path(os.path.abspath(os.path.normpath(os.fspath(candidate))))
        if value not in normalized:
            normalized.append(value)
    resolved: list[dict[str, Any]] = []
    for candidate in normalized:
        if candidate.exists() or candidate.is_symlink():
            chain, final = resolve_symlink_chain(candidate, anchor=None)
            resolved.append(
                {
                    "kind": "on_disk",
                    "candidate": os.fspath(candidate),
                    "symlink_chain": chain,
                    "target": os.fspath(final),
                }
            )
    if not resolved and os.path.isabs(install_name) and install_name in shared_cache_images:
        resolved.append(
            {
                "kind": "shared_cache",
                "candidate": install_name,
                "symlink_chain": [],
                "target": install_name,
                "cache_image_uuid": shared_cache_images[install_name]["uuid"],
            }
        )
    if len(resolved) != 1:
        raise R20ToolError(
            f"Mach-O dependency must resolve uniquely: {install_name} "
            f"({len(resolved)} matches)"
        )
    return {
        "literal_install_name": install_name,
        "ordered_candidates": [os.fspath(value) for value in normalized],
        "resolution": resolved[0],
    }


def build_native_graph(
    *,
    root_paths: Sequence[str | Path],
    python_executable: str | Path,
    runner: FrozenCommandRunner,
    shared_cache: Mapping[str, Any],
) -> dict[str, Any]:
    executable = Path(os.path.abspath(os.fspath(python_executable)))
    image_map = {
        str(value["install_name"]): dict(value)
        for value in shared_cache["images"]
    }
    roots: list[dict[str, Any]] = []
    queue: list[tuple[Path, tuple[str, ...]]] = []
    for raw_root in root_paths:
        entry = Path(os.path.abspath(os.fspath(raw_root)))
        chain, final = resolve_symlink_chain(entry, anchor=None)
        roots.append(
            {
                "entry_path": os.fspath(entry),
                "symlink_chain": chain,
                "final_path": os.fspath(final),
                "architecture": "x86_64",
            }
        )
        queue.append((final, ()))
    node_by_path: dict[str, dict[str, Any]] = {}
    state_seen: set[tuple[str, tuple[str, ...]]] = set()
    edge_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    resolved_target_by_source_edge: dict[tuple[str, str, str], str] = {}
    while queue:
        path, inherited = queue.pop(0)
        state_key = (os.fspath(path), inherited)
        if state_key in state_seen:
            continue
        state_seen.add(state_key)
        if len(state_seen) > 20_000:
            raise R20ToolError("native graph state bound exceeded")
        node = node_by_path.get(os.fspath(path))
        if node is None:
            node = inspect_macho_x86(path, runner)
            node_by_path[os.fspath(path)] = node
        image_rpaths = tuple(
            os.fspath(
                _expand_rpath_token(
                    value,
                    loader=path,
                    executable=executable,
                )
            )
            for value in node["rpaths"]
        )
        effective_rpaths = tuple(dict.fromkeys([*image_rpaths, *inherited]))
        for dependency in node["dependencies"]:
            resolved = resolve_install_name(
                dependency["install_name"],
                loader=path,
                executable=executable,
                image_rpaths=image_rpaths,
                inherited_rpaths=inherited,
                shared_cache_images=image_map,
            )
            result = resolved["resolution"]
            edge = {
                "source": os.fspath(path),
                "architecture": "x86_64",
                "load_command_kind": dependency["kind"],
                **resolved,
            }
            edge_key = (
                edge["source"],
                dependency["kind"],
                dependency["install_name"],
                str(result["target"]),
            )
            source_edge = edge_key[:3]
            prior_target = resolved_target_by_source_edge.get(source_edge)
            if prior_target is not None and prior_target != str(result["target"]):
                raise R20ToolError(
                    "loader-chain context changed a frozen dependency resolution"
                )
            resolved_target_by_source_edge[source_edge] = str(result["target"])
            edge_by_key[edge_key] = edge
            if result["kind"] == "on_disk":
                target = Path(result["target"])
                slices = [value["architecture"] for value in macho_slices(target)]
                if "x86_64" not in slices:
                    raise R20ToolError(
                        f"resolved dependency lacks x86_64 slice: {target}"
                    )
                queue.append((target, effective_rpaths))
    direct_python_edges = [
        value
        for value in edge_by_key.values()
        if value["source"] == os.fspath(executable)
    ]
    if [
        value["literal_install_name"]
        for value in direct_python_edges
    ] != ["/usr/lib/libSystem.B.dylib"]:
        raise R20ToolError("registered Python direct native edge changed")
    nodes = [
        node_by_path[key]
        for key in sorted(node_by_path, key=lambda value: value.encode("utf-8"))
    ]
    edges = [
        edge_by_key[key]
        for key in sorted(
            edge_by_key,
            key=lambda value: tuple(part.encode("utf-8") for part in value),
        )
    ]
    roots.sort(key=lambda value: value["entry_path"].encode("utf-8"))
    payload = {
        "format_version": 1,
        "architecture": "x86_64",
        "python_executable": os.fspath(executable),
        "roots": roots,
        "nodes": nodes,
        "edges": edges,
        "shared_cache_identity_sha256": str(
            shared_cache["shared_cache_identity_sha256"]
        ),
    }
    payload["native_loader_graph_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def parse_dyld_cache_header(path: str | Path) -> dict[str, str]:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise R20ToolError("dyld shared-cache base is not regular")
    with source.open("rb") as handle:
        raw = handle.read(104)
    if len(raw) < 104:
        raise R20ToolError("dyld shared-cache header is truncated")
    magic = raw[:16].rstrip(b"\x00").decode("ascii", errors="strict")
    if "x86_64" not in magic:
        raise R20ToolError("dyld shared-cache header is not x86_64")
    uuid_raw = raw[88:104]
    uuid_hex = uuid_raw.hex()
    cache_uuid = (
        f"{uuid_hex[0:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-"
        f"{uuid_hex[16:20]}-{uuid_hex[20:32]}"
    )
    return {"magic": magic, "uuid": cache_uuid}


def parse_shared_cache_images(text: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(
            r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12})\s+(/\S+)",
            line,
        )
        if match is None:
            raise R20ToolError(f"unsupported dyld cache image line: {line}")
        output.append(
            {"uuid": match.group(1).lower(), "install_name": match.group(2)}
        )
    if not output:
        raise R20ToolError("dyld_info emitted no x86_64 cache images")
    names = [value["install_name"] for value in output]
    if len(names) != len(set(names)):
        raise R20ToolError("duplicate dyld shared-cache image install name")
    output.sort(key=lambda value: value["install_name"].encode("utf-8"))
    return output


def _parse_key_value_output(text: str, *, label: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            if line.startswith(("+--", "|", "APFS ", "Snapshots for")):
                rows.append(["_heading", line])
                continue
            raise R20ToolError(f"unsupported {label} output line: {line}")
        key, value = line.split(":", 1)
        if not key.strip() or not value.strip():
            raise R20ToolError(f"malformed {label} key/value line: {line}")
        rows.append([key.strip(), value.strip()])
    if not rows:
        raise R20ToolError(f"{label} output is empty")
    return rows


def collect_shared_cache_identity(
    *,
    config: Mapping[str, Any],
    runner: FrozenCommandRunner,
) -> dict[str, Any]:
    base = Path(str(config["base_path"]))
    if base != X86_CACHE_BASE:
        raise R20ToolError("wrong dyld shared-cache family")
    header = parse_dyld_cache_header(base)
    uuid_stdout, uuid_stderr = runner.run_x86_dyld_info(
        ["-uuid", os.fspath(base)]
    )
    if uuid_stderr.strip():
        raise R20ToolError("x86_64 cache UUID inspection wrote stderr")
    if parse_dyld_uuid(uuid_stdout) != header["uuid"]:
        raise R20ToolError("x86_64 cache header/tool UUID mismatch")
    list_stdout, list_stderr = runner.run_x86_dyld_info(
        ["-all_dylibs", os.fspath(base)]
    )
    if list_stderr.strip():
        raise R20ToolError("x86_64 cache image inspection wrote stderr")
    images = parse_shared_cache_images(list_stdout)
    parent = base.parent
    components: list[dict[str, Any]] = []
    for entry in sorted(os.scandir(parent), key=lambda item: os.fsencode(item.name)):
        if not entry.name.startswith(base.name):
            continue
        path = Path(entry.path)
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            kind = "regular"
        elif stat.S_ISLNK(info.st_mode):
            kind = "symlink"
        else:
            raise R20ToolError(f"nonregular dyld cache component: {path}")
        row: dict[str, Any] = {
            "filename": entry.name,
            "kind": kind,
            "length": int(info.st_size),
        }
        if kind == "symlink":
            chain, final = resolve_symlink_chain(path, anchor=parent)
            row["literal_target"] = chain[0]["literal_target"]
            row["resolved_filename"] = lexical_relative(final, parent)
        components.append(row)
    if not components:
        raise R20ToolError("x86_64 shared-cache components absent")
    map_path = Path(str(config["map_path"]))
    map_raw = read_regular_bytes(map_path)
    map_text = map_raw.decode("utf-8", errors="strict")
    for image in images:
        count = sum(
            1 for line in map_text.splitlines() if image["install_name"] in line
        )
        if count != 1:
            raise R20ToolError(
                f"cache map does not identify image exactly once: "
                f"{image['install_name']}"
            )
    snapshot_out, snapshot_err = runner.run(
        "/usr/sbin/diskutil", ["apfs", "listSnapshots", "/"]
    )
    if snapshot_err.strip():
        raise R20ToolError("diskutil snapshot inspection wrote stderr")
    snapshot_rows = _parse_key_value_output(snapshot_out, label="diskutil snapshot")
    required_snapshot = dict(config["sealed_snapshot"])
    for key, value in required_snapshot.items():
        if [str(key), str(value)] not in snapshot_rows:
            raise R20ToolError(f"sealed snapshot identity changed: {key}")
    if str(required_snapshot.get("Sealed", "")).lower() not in {"yes", "true"}:
        raise R20ToolError("registered APFS snapshot is not sealed")
    info_out, info_err = runner.run("/usr/sbin/diskutil", ["info", "/"])
    if info_err.strip():
        raise R20ToolError("diskutil root inspection wrote stderr")
    disk_rows = _parse_key_value_output(info_out, label="diskutil info")
    if not any(
        key in {"Authenticated Root", "Sealed"}
        and value.lower() in {"yes", "true"}
        for key, value in disk_rows
    ):
        raise R20ToolError("authenticated-root identity is not affirmative")
    sw_out, sw_err = runner.run("/usr/bin/sw_vers", [])
    if sw_err.strip():
        raise R20ToolError("sw_vers wrote stderr")
    sw_rows = _parse_key_value_output(sw_out, label="sw_vers")
    uname_values: dict[str, str] = {}
    for flag, key in (("-m", "machine"), ("-r", "kernel"), ("-a", "full")):
        stdout, stderr = runner.run("/usr/bin/uname", [flag])
        if stderr.strip() or not stdout.strip() or "\n" in stdout.strip():
            raise R20ToolError(f"unsupported uname {flag} output")
        uname_values[key] = stdout.strip()
    cryptex_plist = Path(str(config["cryptex_system_version_plist"]))
    plist_raw = read_regular_bytes(cryptex_plist)
    try:
        plist = plistlib.loads(plist_raw)
    except Exception as exc:
        raise R20ToolError("malformed Cryptex SystemVersion plist") from exc
    if not isinstance(plist, dict) or "ProductBuildVersion" not in plist:
        raise R20ToolError("Cryptex signed build identity absent")
    dyld_identity = runner.inspect_codesign_target("/usr/lib/dyld")
    if dyld_identity != dict(config["dyld_codesign_identity"]):
        raise R20ToolError("/usr/lib/dyld code-signing identity changed")
    identity: dict[str, Any] = {
        "format_version": 1,
        "architecture": "x86_64",
        "base_path": os.fspath(base),
        "base_sha256": sha256_file(base),
        "header": header,
        "map_path": os.fspath(map_path),
        "map_sha256": hashlib.sha256(map_raw).hexdigest(),
        "components": components,
        "images": images,
        "diskutil_snapshot_rows": snapshot_rows,
        "diskutil_root_rows": disk_rows,
        "sw_vers_rows": sw_rows,
        "uname": uname_values,
        "cryptex_system_version_path": os.fspath(cryptex_plist),
        "cryptex_system_version_sha256": hashlib.sha256(plist_raw).hexdigest(),
        "cryptex_product_build_version": str(plist["ProductBuildVersion"]),
        "dyld_codesign_identity": dyld_identity,
    }
    identity["shared_cache_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    return identity


def _macho_paths_from_component(component: Mapping[str, Any]) -> list[Path]:
    anchor = Path(str(component["anchor"]))
    output: list[Path] = []
    for record in component["records"]:
        if record["kind"] != "regular":
            continue
        path = anchor / str(record["path"])
        if is_macho(path):
            output.append(path)
    return output


def validate_tool_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format_version") != 1 or manifest.get("protocol_id") != PROTOCOL_ID:
        raise R20ToolError("wrong R20 tool manifest identity")
    architecture = manifest.get("architecture")
    expected_architecture = {
        "physical_host": "arm64",
        "python_entry": os.fspath(PYTHON_ENTRY),
        "python_entry_kind": "symlink",
        "python_entry_literal_target": PYTHON_LINK_TARGET,
        "python_final": os.fspath(PYTHON_FINAL),
        "python_final_kind": "regular",
        "python_final_length": PYTHON_FINAL_LENGTH,
        "python_final_sha256": PYTHON_FINAL_SHA256,
        "python_mach_o_architectures": ["x86_64"],
        "python_runtime_machine": "x86_64",
        "execution_mode": "x86_64-under-rosetta",
        "shared_cache_base": os.fspath(X86_CACHE_BASE),
    }
    if architecture != expected_architecture:
        raise R20ToolError("R20 architecture table changed")
    distribution_sets = manifest.get("distribution_sets")
    if not isinstance(distribution_sets, dict):
        raise R20ToolError("distribution sets absent")
    for key, registered in (
        ("runtime", RUNTIME_DISTRIBUTIONS),
        ("test", TEST_DISTRIBUTIONS),
    ):
        rows = distribution_sets.get(key)
        if not isinstance(rows, list):
            raise R20ToolError(f"{key} distribution set absent")
        actual = [(row.get("name"), row.get("version")) for row in rows]
        if actual != list(registered):
            raise R20ToolError(f"{key} distribution set changed")
        for row in rows:
            if HEX64.fullmatch(str(row.get("metadata_sha256", ""))) is None:
                raise R20ToolError(f"{key} raw METADATA hash absent")


def validate_tool_freeze_environment(
    manifest: Mapping[str, Any],
    *,
    verify_command_identities: bool,
) -> FrozenCommandRunner:
    validate_tool_manifest_shape(manifest)
    validate_python_identity()
    for binding in manifest["tool_files"]:
        path = Path(str(binding["path"]))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise R20ToolError(f"tool file is not regular: {path}")
        if info.st_size != int(binding["length"]):
            raise R20ToolError(f"tool file length changed: {path}")
        if sha256_file(path) != str(binding["sha256"]):
            raise R20ToolError(f"tool file hash changed: {path}")
    validate_content_component(manifest["tool_stdlib_component"])
    runner = FrozenCommandRunner(manifest["inspection_tools"])
    runner.verify_files()
    if verify_command_identities:
        runner.verify_identities()
        verify_host_and_translation(runner)
    return runner


def materialize_runtime_closure(
    manifest: Mapping[str, Any],
    *,
    runner: FrozenCommandRunner,
) -> dict[str, Any]:
    paths = manifest["paths"]
    prefix = Path(str(paths["interpreter_prefix"]))
    sites = [Path(str(value)) for value in paths["site_packages"]]
    stdlib_component = build_stdlib_component(
        stdlib_root=str(paths["stdlib_root"]),
        site_packages=sites,
    )
    runtime_set = materialize_distribution_set(
        set_name="runtime",
        expected=manifest["distribution_sets"]["runtime"],
        registered=RUNTIME_DISTRIBUTIONS,
        prefix=prefix,
        site_packages=sites,
    )
    test_set = materialize_distribution_set(
        set_name="test",
        expected=manifest["distribution_sets"]["test"],
        registered=TEST_DISTRIBUTIONS,
        prefix=prefix,
        site_packages=sites,
    )
    project_component = build_content_component(
        name="project-runtime-tools",
        anchor=str(paths["workspace_root"]),
        roots=[str(value) for value in paths["project_roots"]],
        exclude_cache=True,
        allow_file_symlinks=False,
        absent_paths=[str(value) for value in paths.get("absent_project_paths", [])],
    )
    shared_cache = collect_shared_cache_identity(
        config=manifest["shared_cache"],
        runner=runner,
    )
    root_paths: list[Path] = [PYTHON_FINAL]
    root_paths.extend(_macho_paths_from_component(stdlib_component))
    for distribution_set in (runtime_set, test_set):
        for component in distribution_set["components"]:
            root_paths.extend(_macho_paths_from_component(component))
    unique_roots = sorted(
        {Path(os.path.abspath(os.fspath(value))) for value in root_paths},
        key=lambda value: os.fsencode(value),
    )
    native_graph = build_native_graph(
        root_paths=unique_roots,
        python_executable=PYTHON_FINAL,
        runner=runner,
        shared_cache=shared_cache,
    )
    components = [
        project_component,
        stdlib_component,
        *runtime_set["components"],
        *test_set["components"],
    ]
    aggregate_rows = [
        {"name": value["name"], "content_root_sha256": value["content_root_sha256"]}
        for value in components
    ]
    aggregate_rows.extend(
        [
            {
                "name": "shared-cache",
                "content_root_sha256": shared_cache[
                    "shared_cache_identity_sha256"
                ],
            },
            {
                "name": "native-loader-graph",
                "content_root_sha256": native_graph[
                    "native_loader_graph_sha256"
                ],
            },
        ]
    )
    return {
        "format_version": 1,
        "protocol_id": PROTOCOL_ID,
        "python_identity": validate_python_identity(),
        "project_component": project_component,
        "stdlib_component": stdlib_component,
        "runtime_distribution_set": runtime_set,
        "test_distribution_set": test_set,
        "shared_cache": shared_cache,
        "native_loader_graph": native_graph,
        "runtime_content_root_sha256": hashlib.sha256(
            canonical_json_bytes(aggregate_rows)
        ).hexdigest(),
    }


def validate_runtime_closure(expected: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    runner = validate_tool_freeze_environment(
        manifest,
        verify_command_identities=True,
    )
    rebuilt = materialize_runtime_closure(manifest, runner=runner)
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(expected)):
        raise R20ToolError("runtime/native content closure changed")


def validate_materialization_bundle(
    *,
    output_root: str | Path,
    tool_manifest: Mapping[str, Any],
    expected_result_sha256: str,
) -> dict[str, Any]:
    root = Path(output_root)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise R20ToolError("materialization root must be a regular directory")
    expected_names = {
        "ATTEMPT.json",
        "RESULT.json",
        "support_rows.json",
        "runtime_content_records.json",
        "test_content_records.json",
        "native_loader_graph.json",
        "materialization_summary.json",
    }
    actual_names = {entry.name for entry in os.scandir(root)}
    if actual_names != expected_names:
        raise R20ToolError("materialization artifact path set changed")
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in sorted(expected_names, key=lambda value: value.encode("utf-8")):
        path = root / name
        raw = read_regular_bytes(path)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        payloads[name] = load_canonical_json(
            path,
            final_lf=True,
            sort_keys=False,
        )
    if hashes["RESULT.json"] != expected_result_sha256:
        raise R20ToolError("materialization terminal marker hash changed")
    result = payloads["RESULT.json"]
    attempt = payloads["ATTEMPT.json"]
    if set(attempt) != {
        "format_version",
        "protocol_id",
        "attempt_id",
        "mode",
        "tool_manifest_sha256",
        "created_at",
    }:
        raise R20ToolError("materialization attempt schema changed")
    if set(result) != {
        "format_version",
        "protocol_id",
        "attempt_id",
        "mode",
        "status",
        "tool_manifest_sha256",
        "started_at",
        "completed_at",
        "outputs",
    }:
        raise R20ToolError("materialization result schema changed")
    if (
        attempt.get("protocol_id") != PROTOCOL_ID
        or attempt.get("attempt_id") != MATERIALIZATION_ATTEMPT_ID
        or attempt.get("mode") != "runtime_content_materialization"
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("attempt_id") != MATERIALIZATION_ATTEMPT_ID
        or result.get("mode") != "runtime_content_materialization"
        or result.get("status") != "success"
    ):
        raise R20ToolError("materialization attempt/result identity changed")
    manifest_sha = hashlib.sha256(
        canonical_json_bytes(dict(tool_manifest), final_lf=True, sort_keys=True)
    ).hexdigest()
    if (
        attempt.get("tool_manifest_sha256") != manifest_sha
        or result.get("tool_manifest_sha256") != manifest_sha
    ):
        raise R20ToolError("materialization manifest binding changed")
    output_rows = result.get("outputs", [])
    if not isinstance(output_rows, list):
        raise R20ToolError("materialization result outputs are malformed")
    expected_outputs = {
        str(value["filename"]): (int(value["length"]), str(value["sha256"]))
        for value in output_rows
    }
    if len(expected_outputs) != len(output_rows):
        raise R20ToolError("duplicate materialization result output")
    if set(expected_outputs) != expected_names - {"ATTEMPT.json", "RESULT.json"}:
        raise R20ToolError("materialization result output set changed")
    for name, (length, digest) in expected_outputs.items():
        path = root / name
        if path.lstat().st_size != length or hashes[name] != digest:
            raise R20ToolError(f"materialization output binding changed: {name}")
    support = payloads["support_rows.json"]
    if hashlib.sha256(canonical_json_bytes(support.get("rows"))).hexdigest() != support.get(
        "support_root_sha256"
    ):
        raise R20ToolError("materialized support root changed")
    runtime = payloads["runtime_content_records.json"]
    tests = payloads["test_content_records.json"]
    native = payloads["native_loader_graph.json"]
    expected_closure = {
        "format_version": 1,
        "protocol_id": PROTOCOL_ID,
        "python_identity": runtime["python_identity"],
        "project_component": runtime["project_component"],
        "stdlib_component": runtime["stdlib_component"],
        "runtime_distribution_set": runtime["runtime_distribution_set"],
        "test_distribution_set": tests["test_distribution_set"],
        "shared_cache": native["shared_cache"],
        "native_loader_graph": native["native_loader_graph"],
        "runtime_content_root_sha256": runtime["runtime_content_root_sha256"],
    }
    validate_runtime_closure(expected_closure, tool_manifest)
    summary = payloads["materialization_summary.json"]
    if set(summary) != {
        "format_version",
        "protocol_id",
        "attempt_id",
        "tool_manifest_sha256",
        "support_root_sha256",
        "runtime_content_root_sha256",
        "native_loader_graph_sha256",
        "artifacts",
        "status",
    }:
        raise R20ToolError("materialization summary schema changed")
    if summary.get("artifacts") != [
        value
        for value in output_rows
        if value["filename"] != "materialization_summary.json"
    ]:
        raise R20ToolError("materialization summary output binding changed")
    if (
        summary.get("status") != "success"
        or summary.get("support_root_sha256") != support["support_root_sha256"]
        or summary.get("runtime_content_root_sha256")
        != runtime["runtime_content_root_sha256"]
        or summary.get("native_loader_graph_sha256")
        != native["native_loader_graph"]["native_loader_graph_sha256"]
    ):
        raise R20ToolError("materialization summary binding changed")
    return {
        "result_sha256": hashes["RESULT.json"],
        "support_root_sha256": support["support_root_sha256"],
        "runtime_content_root_sha256": runtime["runtime_content_root_sha256"],
        "native_loader_graph_sha256": native["native_loader_graph"][
            "native_loader_graph_sha256"
        ],
    }


def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(target.parent)


def write_atomic_once(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise R20ToolError(f"create-once output already exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp")
    write_exclusive(temporary, payload)
    if target.exists() or target.is_symlink():
        raise R20ToolError(f"create-once output raced: {target}")
    os.rename(temporary, target)
    fsync_directory(target.parent)
