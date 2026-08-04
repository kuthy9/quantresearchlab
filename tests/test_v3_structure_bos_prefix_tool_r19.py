from __future__ import annotations

import ast
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import materialize_v3_structure_bos_prefix_r19 as r19


EXPECTED_IDENTITIES = {
    "r19-4h-long-confirmed": (
        "280bf46081961428cd9e719a",
        "a09828c3fa7292542766580f",
    ),
    "r19-4h-long-wick": (
        "280bf46081961428cd9e719a",
        "a09828c3fa7292542766580f",
    ),
    "r19-4h-long-opposed": (
        "7efb9d7fcf4659de209b6e21",
        "abc268b7a6515c91b80ceaee",
    ),
    "r19-4h-short-confirmed": (
        "0bd3836b0c511775cd0ba3f9",
        "c0d94200460f1eba352b350d",
    ),
    "r19-4h-short-wick": (
        "0bd3836b0c511775cd0ba3f9",
        "c0d94200460f1eba352b350d",
    ),
    "r19-4h-short-opposed": (
        "71731380e0ffb0aa66254f44",
        "908625edc1ba3893aa5d72d8",
    ),
    "r19-1h-long-opposed": (
        "a1a68953689f359702be285a",
        "d528d5fd4c8ebdbc40308b58",
    ),
    "r19-1h-short-confirmed": (
        "cb7c1289dfe22f3c87b3c895",
        "7c8713be99a422b74e062805",
    ),
    "r19-1h-short-wick": (
        "cb7c1289dfe22f3c87b3c895",
        "7c8713be99a422b74e062805",
    ),
    "r19-1h-short-opposed": (
        "02a225bb8cfbb2b71e6a00bd",
        "e1710c857d38f9b05b904f5b",
    ),
}


class FrozenSupportTests(unittest.TestCase):
    def test_tool_imports_only_the_standard_library(self) -> None:
        source = Path(r19.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed_roots = {
            "__future__",
            "argparse",
            "dataclasses",
            "datetime",
            "hashlib",
            "json",
            "os",
            "pathlib",
            "stat",
            "typing",
        }
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertLessEqual(imported_roots, allowed_roots)
        self.assertNotIn("pandas", imported_roots)
        self.assertNotIn("smc_trader", imported_roots)

    def test_all_ten_targets_have_exact_production_identities(self) -> None:
        rows = r19.build_support_rows()
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["target_id"] for row in rows],
            list(EXPECTED_IDENTITIES),
        )
        for row in rows:
            expected_swing, expected_bos = EXPECTED_IDENTITIES[
                row["target_id"]
            ]
            self.assertEqual(row["target_swing_id"], expected_swing)
            self.assertEqual(row["target_bos_id"], expected_bos)
            self.assertEqual(row["expected_matching_swing_count"], 1)
            self.assertEqual(row["expected_matching_bos_count"], 1)
            self.assertEqual(row["expected_admission_count"], 1)

    def test_exact_first_stage_ordinals_are_frozen_by_bucket(self) -> None:
        for row in r19.build_support_rows():
            expected = list(
                r19.EXPECTED_STAGE_ORDINALS[row["exact_bucket_enum"]]
            )
            self.assertEqual(row["expected_first_stage_ordinals"], expected)
            self.assertEqual(
                len(row["expected_first_stage_clocks"]),
                len(r19.STAGE_NAMES),
            )

    def test_exact_h1_stage_and_boundary_clocks(self) -> None:
        expected_by_bucket = {
            "confirmed_bos": [
                "2025-01-06T03:00:00-05:00",
                "2025-01-06T05:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
            ],
            "wick_only_no_close": [
                "2025-01-06T03:00:00-05:00",
                "2025-01-06T05:00:00-05:00",
                "2025-01-06T05:00:00-05:00",
                "2025-01-06T05:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
            ],
            "broken_or_opposed": [
                "2025-01-06T05:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
                "2025-01-06T06:00:00-05:00",
            ],
        }
        for row in r19.build_support_rows():
            if row["timeframe"] != "1H":
                continue
            self.assertEqual(
                row["expected_first_stage_clocks"],
                expected_by_bucket[row["exact_bucket_enum"]],
            )
            self.assertEqual(
                row["expected_boundary_clock"],
                "2025-01-06T06:00:00-05:00",
            )

    def test_exact_h4_stage_and_boundary_clocks(self) -> None:
        expected_by_bucket = {
            "confirmed_bos": [
                "2025-01-07T06:00:00-05:00",
                "2025-01-07T14:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
            ],
            "wick_only_no_close": [
                "2025-01-07T06:00:00-05:00",
                "2025-01-07T14:00:00-05:00",
                "2025-01-07T14:00:00-05:00",
                "2025-01-07T14:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
            ],
            "broken_or_opposed": [
                "2025-01-07T14:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
                "2025-01-07T17:00:00-05:00",
            ],
        }
        for row in r19.build_support_rows():
            if row["timeframe"] != "4H":
                continue
            self.assertEqual(
                row["expected_first_stage_clocks"],
                expected_by_bucket[row["exact_bucket_enum"]],
            )
            self.assertEqual(
                row["expected_boundary_clock"],
                "2025-01-07T17:00:00-05:00",
            )


class TemplateAndCommitmentTests(unittest.TestCase):
    def test_literal_mirror_is_tick_exact_and_involutive(self) -> None:
        for candle in (
            *r19.CANONICAL_LONG_PREFIX,
            r19.TERMINAL_CONFIRMED_LONG,
            r19.TERMINAL_WICK_LONG,
            r19.TERMINAL_OPPOSED_SHORT,
        ):
            mirrored = r19.mirror_candle(candle)
            self.assertEqual(r19.mirror_candle(mirrored), candle)
            self.assertEqual(
                mirrored.open_ticks,
                2 * r19.REFERENCE_PRICE_TICKS - candle.open_ticks,
            )
            self.assertEqual(
                mirrored.close_ticks,
                2 * r19.REFERENCE_PRICE_TICKS - candle.close_ticks,
            )
            self.assertEqual(
                mirrored.high_ticks,
                2 * r19.REFERENCE_PRICE_TICKS - candle.low_ticks,
            )
            self.assertEqual(
                mirrored.low_ticks,
                2 * r19.REFERENCE_PRICE_TICKS - candle.high_ticks,
            )

    def test_h1_and_h4_intervals_include_only_registered_time(self) -> None:
        h1 = r19.target_intervals("1H")
        h4 = r19.target_intervals("4H")
        self.assertEqual(len(h1), r19.TARGET_CANDLE_COUNT)
        self.assertEqual(len(h4), r19.TARGET_CANDLE_COUNT)
        self.assertTrue(
            all(end - start == timedelta(hours=1) for start, end in h1)
        )
        self.assertEqual(
            [int((end - start).total_seconds() // 60) for start, end in h4],
            [240, 240, 240, 240, 240, 180] * 2,
        )
        self.assertEqual(
            h4[6][0] - h4[5][1],
            timedelta(minutes=60),
        )

    def test_minute_expansion_reconstructs_registered_ohlc(self) -> None:
        candle = r19.CANONICAL_LONG_PREFIX[-1]
        start, end = r19.target_intervals("4H")[-1]
        bars = r19.expand_candle_to_minute_bars(
            candle,
            start=start,
            end=end,
        )
        self.assertEqual(len(bars), 180)
        self.assertEqual(bars[0]["start"], start.isoformat())
        self.assertEqual(
            bars[-1]["start"],
            (end - timedelta(minutes=1)).isoformat(),
        )
        self.assertEqual(bars[0]["open_ticks"], candle.open_ticks)
        self.assertEqual(
            max(int(bar["high_ticks"]) for bar in bars),
            candle.high_ticks,
        )
        self.assertEqual(
            min(int(bar["low_ticks"]) for bar in bars),
            candle.low_ticks,
        )
        self.assertEqual(bars[-1]["close_ticks"], candle.close_ticks)

    def test_length_framed_hash_matches_independent_tiny_reference(self) -> None:
        bars = (
            {
                "start": "2025-01-05T18:00:00-05:00",
                "open_ticks": 36,
                "high_ticks": 37,
                "low_ticks": 36,
                "close_ticks": 37,
                "volume": 100,
                "symbol": "NQH5",
                "instrument_id": 1,
            },
            {
                "start": "2025-01-05T18:01:00-05:00",
                "open_ticks": 37,
                "high_ticks": 37,
                "low_ticks": 35,
                "close_ticks": 35,
                "volume": 100,
                "symbol": "NQH5",
                "instrument_id": 1,
            },
        )
        digest = hashlib.sha256()
        for bar in bars:
            raw = json.dumps(
                bar,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big", signed=False))
            digest.update(raw)
        self.assertEqual(
            r19.commitment_for_bars(bars),
            digest.hexdigest(),
        )

    def test_each_support_prefix_is_exactly_reproducible(self) -> None:
        rows = r19.build_support_rows()
        for target, row in zip(r19.TARGETS, rows, strict=True):
            commitment = row["expected_prefix_commitment_sha256"]
            self.assertEqual(commitment, r19.prefix_commitment(target))
            self.assertEqual(len(commitment), 64)
            self.assertEqual(commitment, commitment.lower())
            int(commitment, 16)


class CliWriteBoundaryTests(unittest.TestCase):
    def test_write_requires_parent_attempt_marker_and_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "support_rows.json"
            with self.assertRaises(FileNotFoundError):
                r19.write_support_rows(output)

            (root / "ATTEMPT.json").write_bytes(b"{}\n")
            written = r19.write_support_rows(output)
            self.assertEqual(written, output)
            raw = output.read_bytes()
            self.assertEqual(raw, r19.support_rows_bytes())
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(
                json.loads(raw),
                list(r19.build_support_rows()),
            )
            with self.assertRaises(FileExistsError):
                r19.write_support_rows(output)

    def test_marker_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker_target = root / "real-marker.json"
            marker_target.write_bytes(b"{}\n")
            marker = root / "ATTEMPT.json"
            try:
                marker.symlink_to(marker_target.name)
            except OSError as error:
                self.skipTest(f"symlink creation is unavailable: {error}")
            with self.assertRaises(ValueError):
                r19.write_support_rows(root / "support_rows.json")

    def test_cli_rejects_any_other_output_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ATTEMPT.json").write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                r19.write_support_rows(root / "other.json")


if __name__ == "__main__":
    unittest.main()
