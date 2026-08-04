"""Unexecuted stdlib-only miniature-fixture tests for the R20 Stage-A tool."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import unittest
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "scripts" / "v3_structure_bos_runtime_tool_r20.py"
SPEC = importlib.util.spec_from_file_location("r20_runtime_tool_under_test", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen R20 runtime-tool test target")
r20 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r20)

pytestmark = pytest.mark.historical_frozen


UUID = "00112233-4455-6677-8899-aabbccddeeff"


def _write_x86_macho(path: Path) -> None:
    header = (
        b"\xcf\xfa\xed\xfe"
        + struct.pack("<ii", 0x01000007, 3)
        + b"\x00" * 20
    )
    path.write_bytes(header)


def _write_arm_macho(path: Path) -> None:
    header = (
        b"\xcf\xfa\xed\xfe"
        + struct.pack("<ii", 0x0100000C, 0)
        + b"\x00" * 20
    )
    path.write_bytes(header)


def _otool_text(dependency: str = "/usr/lib/libSystem.B.dylib") -> str:
    return "\n".join(
        (
            "/tmp/python:",
            "Load command 0",
            "      cmd LC_UUID",
            "  cmdsize 24",
            f"     uuid {UUID}",
            "Load command 1",
            "      cmd LC_RPATH",
            "  cmdsize 32",
            "     path @loader_path/lib (offset 12)",
            "Load command 2",
            "      cmd LC_LOAD_DYLIB",
            "  cmdsize 56",
            f"     name {dependency} (offset 24)",
            "",
        )
    )


class FakeRunner:
    def run(self, executable: str, arguments: list[str]):
        if executable != "/usr/bin/otool":
            raise AssertionError(executable)
        return _otool_text(), ""

    def run_x86_dyld_info(self, arguments: list[str]):
        return f"UUID: {UUID}\n", ""


class CanonicalAndFilesystemTests(unittest.TestCase):
    def test_canonical_json_is_compact_and_stable(self) -> None:
        payload = {"z": 1, "a": ["é", True, None]}
        raw = r20.canonical_json_bytes(payload)
        self.assertEqual(raw, b'{"a":["\xc3\xa9",true,null],"z":1}')
        self.assertEqual(hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw).hexdigest())

    def test_duplicate_json_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            path.write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaises(r20.R20ToolError):
                r20.load_canonical_json(path)

    def test_regular_component_round_trip_and_extra_file_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            tree.mkdir()
            (tree / "a.py").write_text("a = 1\n", encoding="utf-8")
            expected = r20.build_content_component(
                name="mini",
                anchor=root,
                roots=[tree],
            )
            r20.validate_content_component(expected)
            (tree / "extra.py").write_text("extra = 1\n", encoding="utf-8")
            with self.assertRaises(r20.R20ToolError):
                r20.validate_content_component(expected)

    def test_symlink_literal_and_target_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"target")
            link = root / "link.bin"
            link.symlink_to("target.bin")
            component = r20.build_content_component(
                name="links",
                anchor=root,
                roots=[link],
                allow_file_symlinks=True,
            )
            link_record = next(
                value for value in component["records"] if value["path"] == "link.bin"
            )
            self.assertEqual(link_record["literal_target"], "target.bin")
            self.assertEqual(link_record["resolved_path"], "target.bin")
            link.unlink()
            other = root / "other.bin"
            other.write_bytes(b"target")
            link.symlink_to("other.bin")
            with self.assertRaises(r20.R20ToolError):
                r20.validate_content_component(component)

    def test_symlink_escape_and_cycle_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "anchor"
            anchor.mkdir()
            outside = root / "outside"
            outside.write_bytes(b"x")
            escape = anchor / "escape"
            escape.symlink_to("../outside")
            with self.assertRaises(r20.R20ToolError):
                r20.resolve_symlink_chain(escape, anchor=anchor)
            first = anchor / "first"
            second = anchor / "second"
            first.symlink_to("second")
            second.symlink_to("first")
            with self.assertRaises(r20.R20ToolError):
                r20.resolve_symlink_chain(first, anchor=anchor)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires POSIX")
    def test_fifo_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(r20.R20ToolError):
                r20.build_content_component(
                    name="fifo",
                    anchor=root,
                    roots=[fifo],
                )

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix socket required")
    def test_unix_socket_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "socket"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(os.fspath(path))
                with self.assertRaises(r20.R20ToolError):
                    r20.build_content_component(
                        name="socket",
                        anchor=root,
                        roots=[path],
                    )
            finally:
                server.close()


class DistributionAndStdlibTests(unittest.TestCase):
    def test_manual_dist_info_and_owned_root_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            site = prefix / "lib" / "python3.12" / "site-packages"
            info = site / "demo-1.0.dist-info"
            package = site / "demo"
            info.mkdir(parents=True)
            package.mkdir()
            metadata = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n"
            (info / "METADATA").write_bytes(metadata)
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (info / "RECORD").write_text(
                "demo/__init__.py,,\n"
                "demo-1.0.dist-info/METADATA,,\n"
                "demo-1.0.dist-info/RECORD,,\n",
                encoding="utf-8",
            )
            expected = [
                {
                    "name": "demo",
                    "version": "1.0",
                    "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                }
            ]
            result = r20.materialize_distribution_set(
                set_name="mini",
                expected=expected,
                registered=(("demo", "1.0"),),
                prefix=prefix,
                site_packages=[site],
            )
            records = result["components"][0]["records"]
            self.assertTrue(
                any(value["path"].endswith("demo/__init__.py") for value in records)
            )
            self.assertTrue(
                any(value["path"].endswith("demo-1.0.dist-info/METADATA") for value in records)
            )

    def test_stdlib_excludes_site_packages_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stdlib = Path(directory) / "stdlib"
            site = stdlib / "site-packages"
            cache = stdlib / "__pycache__"
            site.mkdir(parents=True)
            cache.mkdir()
            (stdlib / "json.py").write_text("VALUE = 1\n", encoding="utf-8")
            (site / "foreign.py").write_text("VALUE = 2\n", encoding="utf-8")
            (cache / "json.pyc").write_bytes(b"cache")
            component = r20.build_stdlib_component(
                stdlib_root=stdlib,
                site_packages=[site],
            )
            paths = [value["path"] for value in component["records"]]
            self.assertEqual(paths, ["json.py"])


class MachOTests(unittest.TestCase):
    def test_thin_x86_64_magic_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "python"
            _write_x86_macho(path)
            slices = r20.macho_slices(path)
            self.assertEqual([value["architecture"] for value in slices], ["x86_64"])

    def test_arm_only_root_is_not_x86_64(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm"
            _write_arm_macho(path)
            slices = r20.macho_slices(path)
            self.assertEqual([value["architecture"] for value in slices], ["arm64"])

    def test_otool_parser_freezes_uuid_rpath_and_dependency(self) -> None:
        parsed = r20.parse_otool_load_commands(_otool_text())
        self.assertEqual(parsed["uuid"], UUID)
        self.assertEqual(parsed["rpaths"], ["@loader_path/lib"])
        self.assertEqual(
            parsed["dependencies"],
            [
                {
                    "kind": "LC_LOAD_DYLIB",
                    "install_name": "/usr/lib/libSystem.B.dylib",
                }
            ],
        )

    def test_unknown_otool_line_fails(self) -> None:
        with self.assertRaises(r20.R20ToolError):
            r20.parse_otool_load_commands("unregistered output\n")

    def test_unique_rpath_resolution_and_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loader = root / "bin" / "python"
            loader.parent.mkdir()
            loader.write_bytes(b"x")
            first = root / "lib1"
            second = root / "lib2"
            first.mkdir()
            second.mkdir()
            (first / "libvalue.dylib").write_bytes(b"one")
            resolved = r20.resolve_install_name(
                "@rpath/libvalue.dylib",
                loader=loader,
                executable=loader,
                image_rpaths=[os.fspath(first)],
                inherited_rpaths=[],
                shared_cache_images={},
            )
            self.assertEqual(
                resolved["resolution"]["target"],
                os.fspath(first / "libvalue.dylib"),
            )
            (second / "libvalue.dylib").write_bytes(b"two")
            with self.assertRaises(r20.R20ToolError):
                r20.resolve_install_name(
                    "@rpath/libvalue.dylib",
                    loader=loader,
                    executable=loader,
                    image_rpaths=[os.fspath(first), os.fspath(second)],
                    inherited_rpaths=[],
                    shared_cache_images={},
                )

    def test_graph_uses_actual_edge_and_does_not_invent_libpython(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            _write_x86_macho(python)
            shared = {
                "images": [
                    {
                        "install_name": "/usr/lib/libSystem.B.dylib",
                        "uuid": UUID,
                    }
                ],
                "shared_cache_identity_sha256": "a" * 64,
            }
            forced_resolution = {
                "literal_install_name": "/usr/lib/libSystem.B.dylib",
                "ordered_candidates": ["/usr/lib/libSystem.B.dylib"],
                "resolution": {
                    "kind": "shared_cache",
                    "candidate": "/usr/lib/libSystem.B.dylib",
                    "symlink_chain": [],
                    "target": "/usr/lib/libSystem.B.dylib",
                    "cache_image_uuid": UUID,
                },
            }
            with mock.patch.object(
                r20,
                "resolve_install_name",
                return_value=forced_resolution,
            ):
                graph = r20.build_native_graph(
                    root_paths=[python],
                    python_executable=python,
                    runner=FakeRunner(),
                    shared_cache=shared,
                )
            self.assertEqual([value["path"] for value in graph["nodes"]], [os.fspath(python)])
            self.assertNotIn("libpython", json.dumps(graph))


class SharedCacheTests(unittest.TestCase):
    def test_x86_cache_header_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dyld_shared_cache_x86_64"
            raw = bytearray(128)
            raw[:16] = b"dyld_v1  x86_64".ljust(16, b" ")
            raw[88:104] = bytes.fromhex("00112233445566778899aabbccddeeff")
            path.write_bytes(raw)
            parsed = r20.parse_dyld_cache_header(path)
            self.assertIn("x86_64", parsed["magic"])
            self.assertEqual(parsed["uuid"], UUID)

    def test_arm_cache_header_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dyld_shared_cache_arm64e"
            raw = bytearray(128)
            raw[:16] = b"dyld_v1  arm64e".ljust(16, b" ")
            path.write_bytes(raw)
            with self.assertRaises(r20.R20ToolError):
                r20.parse_dyld_cache_header(path)

    def test_cache_image_parser_rejects_unknown_format(self) -> None:
        with self.assertRaises(r20.R20ToolError):
            r20.parse_shared_cache_images("0x1234 /usr/lib/libSystem.B.dylib\n")


class CreateOnceTests(unittest.TestCase):
    def test_atomic_once_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RESULT.json"
            r20.write_atomic_once(path, b"first")
            with self.assertRaises(r20.R20ToolError):
                r20.write_atomic_once(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
