from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from smc_trader.artifact_stream import (
    bound_regular_file,
    canonical_json,
    canonical_record_sha256,
    publish_canonical_manifest,
    read_json_object,
)


def test_canonical_manifest_publish_is_atomic_no_clobber_and_hash_bound(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "case-index.manifest.json"
    payload = {"status": "complete", "records": 2}

    assert publish_canonical_manifest(destination, payload) == destination
    original = destination.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    assert original == canonical_json(payload)
    assert canonical_record_sha256(payload) == hashlib.sha256(original).hexdigest()
    assert read_json_object(
        destination,
        expected_sha256=digest,
        name="case index manifest",
        require_canonical=True,
    ) == payload

    with pytest.raises(FileExistsError):
        publish_canonical_manifest(destination, {"status": "replaced"})
    assert destination.read_bytes() == original
    assert tuple(tmp_path.glob(".case-index.manifest.json.*.tmp")) == ()

    destination.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        read_json_object(
            destination,
            expected_sha256=digest,
            name="case index manifest",
        )


def test_bound_regular_file_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_bytes(canonical_json({"status": "complete"}))
    assert bound_regular_file(root, manifest.name, name="manifest") == manifest

    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(ValueError, match="must be relative"):
        bound_regular_file(root, "../outside.json", name="manifest")
    with pytest.raises(ValueError, match="escaped its root"):
        bound_regular_file(
            root,
            outside,
            name="manifest",
            allow_absolute_within_root=True,
        )

    link = root / "link.json"
    link.symlink_to(manifest)
    with pytest.raises(FileNotFoundError, match="binding is missing"):
        bound_regular_file(root, link.name, name="manifest")
