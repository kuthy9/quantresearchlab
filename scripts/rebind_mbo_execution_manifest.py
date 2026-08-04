#!/usr/bin/env python3
"""Copy a verified revealed MBO minute artifact under a newer protocol hash."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.validation import load_validation_protocol  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validation-protocol",
        default="configs/validation_protocol_v2_1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    source_manifest = source.with_suffix(
        source.suffix + ".manifest.json"
    )
    destination = Path(args.output)
    destination_manifest = destination.with_suffix(
        destination.suffix + ".manifest.json"
    )
    if not source.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(
            "MBO rebind requires the verified parquet and adjacent manifest"
        )
    if destination.exists() or destination_manifest.exists():
        raise FileExistsError("refusing to overwrite rebound MBO artifacts")
    parent = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (
        int(parent.get("format_version", 0)) != 1
        or bool(parent.get("sealed_holdout_read"))
    ):
        raise ValueError("only a revealed non-holdout v1 MBO artifact may rebind")
    source_hash = _sha256(source)
    if parent.get("output_sha256") != source_hash:
        raise ValueError("parent MBO parquet does not match its manifest")
    start = pd.Timestamp(parent.get("start"))
    end = pd.Timestamp(parent.get("end_exclusive"))
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("parent MBO interval is invalid")
    validation = load_validation_protocol(args.validation_protocol)
    window = validation.classify_mbo(start, end)
    if window.role == "sealed_holdout":
        raise ValueError("rebind cannot reveal or copy a sealed MBO holdout")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    output_hash = _sha256(destination)
    if output_hash != source_hash:
        destination.unlink()
        raise IOError("rebound MBO bytes differ from the verified parent")
    payload = {
        **parent,
        "validation_protocol_version": validation.version,
        "validation_protocol_hash": validation.fingerprint,
        "validation_window_role": window.role,
        "output": str(destination),
        "output_sha256": output_hash,
        "sealed_holdout_read": False,
        "materialization_kind": "verified_manifest_rebind_without_value_change",
        "parent_artifact": str(source),
        "parent_artifact_sha256": source_hash,
        "parent_manifest": str(source_manifest),
        "parent_manifest_sha256": _sha256(source_manifest),
        "source_values_changed": False,
    }
    destination_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "sha256": output_hash,
                "window_role": window.role,
                "source_values_changed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
