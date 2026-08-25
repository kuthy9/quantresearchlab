from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd
import pytest

import scripts.run_semantic_signal_research as runner


def _legacy_bytes(records: list[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                runner.to_primitive(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _temps(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_streamed_jsonl_matches_legacy_bytes_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    records = [{"z": 2, "a": "alpha"}, {"event_id": "event:1", "ok": True}]
    expected = _legacy_bytes(records)
    path.write_bytes(b"legacy sentinel")

    receipt = runner._write_jsonl(path, records)

    assert path.read_bytes() == expected
    assert receipt == {
        "sha256": hashlib.sha256(expected).hexdigest(),
        "rows": 2,
    }
    assert _temps(path) == []


def test_streamed_jsonl_preserves_legacy_unicode_and_timestamp_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unicode.jsonl"
    clock = pd.Timestamp("2024-06-03T09:30:00-04:00")
    records = [
        {
            "标签": "策略✓",
            "clock": clock,
            "count": 7,
            "ratio": 1.25,
            "ready": True,
        }
    ]

    runner._write_jsonl(path, records)

    payload = path.read_bytes()
    assert payload == _legacy_bytes(records)
    assert "策略✓".encode() in payload
    assert json.loads(payload) == {
        "标签": "策略✓",
        "clock": clock.isoformat(),
        "count": 7,
        "ratio": 1.25,
        "ready": True,
    }


def test_streamed_jsonl_empty_file_has_canonical_receipt(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"

    receipt = runner._write_jsonl(path, iter(()))

    assert path.read_bytes() == b""
    assert receipt == {
        "sha256": hashlib.sha256(b"").hexdigest(),
        "rows": 0,
    }


def test_streamed_jsonl_consumes_one_shot_iterable_once(tmp_path: Path) -> None:
    class OneShot:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self) -> Iterator[Mapping[str, object]]:
            self.iterations += 1
            if self.iterations != 1:
                raise AssertionError("iterable consumed more than once")
            yield {"event_id": "one"}
            yield {"event_id": "two"}

    records = OneShot()
    receipt = runner._write_jsonl(tmp_path / "one-shot.jsonl", records)

    assert records.iterations == 1
    assert receipt["rows"] == 2


def test_streamed_jsonl_failure_removes_unique_temp(tmp_path: Path) -> None:
    path = tmp_path / "failure.jsonl"

    def records() -> Iterator[Mapping[str, object]]:
        yield {"event_id": "written-to-temp-only"}
        raise RuntimeError("injected iterator failure")

    with pytest.raises(RuntimeError, match="injected iterator failure"):
        runner._write_jsonl(path, records(), no_clobber=True)

    assert not path.exists()
    assert _temps(path) == []


def test_comparison_no_clobber_preserves_sentinel(tmp_path: Path) -> None:
    path = tmp_path / "sentinel.jsonl"
    path.write_bytes(b"do not replace")

    with pytest.raises(FileExistsError, match="already exists"):
        runner._write_jsonl(path, [{"event_id": "new"}], no_clobber=True)

    assert path.read_bytes() == b"do not replace"
    assert _temps(path) == []


def test_comparison_hard_link_is_commit_point_if_temp_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "commit.json"
    original_unlink = Path.unlink

    def fail_private_temp(self: Path, *args: object, **kwargs: object) -> None:
        if self.parent == tmp_path and self.name.startswith(f".{path.name}."):
            raise OSError("injected temp cleanup failure")
        original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", fail_private_temp)
        runner._publish_bytes(path, b"committed\n", no_clobber=True)

    assert path.read_bytes() == b"committed\n"
    for temporary in _temps(path):
        temporary.unlink()


def test_report_is_published_before_result_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    order: list[str] = []

    def capture(path: Path, payload: bytes, *, no_clobber: bool) -> None:
        del payload
        assert no_clobber is True
        order.append(path.name)

    monkeypatch.setattr(runner, "_publish_bytes", capture)

    runner._write_report_then_result(
        output,
        {"status": "complete"},
        "# report",
        no_clobber=True,
    )

    assert order == ["result.md", "result.json"]


def test_partial_comparison_bundle_has_no_marker_and_blocks_same_path_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "comparison.json"
    residual = runner._ledger_path(output, "event_study")
    runner._write_jsonl(
        residual,
        [{"event_id": "already-committed-ledger"}],
        no_clobber=True,
    )

    def fail_report(path: Path, payload: bytes, *, no_clobber: bool) -> None:
        del path, payload, no_clobber
        raise RuntimeError("injected mid-bundle failure")

    monkeypatch.setattr(runner, "_publish_bytes", fail_report)
    with pytest.raises(RuntimeError, match="mid-bundle"):
        runner._write_report_then_result(
            output,
            {"status": "complete"},
            "# report",
            no_clobber=True,
        )

    assert residual.is_file()
    assert not output.exists()
    with pytest.raises(FileExistsError, match="event_study"):
        runner._preflight_output_bundle(
            output,
            research_protocol_version=3,
            max_bars=None,
            force=False,
        )
