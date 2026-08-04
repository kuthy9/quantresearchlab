from __future__ import annotations

import ast
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import materialize_v3_structure_bos_prefix_r20 as r20


EXPECTED_IDENTITIES = {
    "r20-4h-long-confirmed": (
        "280bf46081961428cd9e719a",
        "a09828c3fa7292542766580f",
    ),
    "r20-4h-long-wick": (
        "280bf46081961428cd9e719a",
        "a09828c3fa7292542766580f",
    ),
    "r20-4h-long-opposed": (
        "7efb9d7fcf4659de209b6e21",
        "abc268b7a6515c91b80ceaee",
    ),
    "r20-4h-short-confirmed": (
        "0bd3836b0c511775cd0ba3f9",
        "c0d94200460f1eba352b350d",
    ),
    "r20-4h-short-wick": (
        "0bd3836b0c511775cd0ba3f9",
        "c0d94200460f1eba352b350d",
    ),
    "r20-4h-short-opposed": (
        "71731380e0ffb0aa66254f44",
        "908625edc1ba3893aa5d72d8",
    ),
    "r20-1h-long-opposed": (
        "a1a68953689f359702be285a",
        "d528d5fd4c8ebdbc40308b58",
    ),
    "r20-1h-short-confirmed": (
        "cb7c1289dfe22f3c87b3c895",
        "7c8713be99a422b74e062805",
    ),
    "r20-1h-short-wick": (
        "cb7c1289dfe22f3c87b3c895",
        "7c8713be99a422b74e062805",
    ),
    "r20-1h-short-opposed": (
        "02a225bb8cfbb2b71e6a00bd",
        "e1710c857d38f9b05b904f5b",
    ),
}


class IndependenceAndIdentityTests(unittest.TestCase):
    def test_tool_has_only_stdlib_imports_and_no_r19_reference(self) -> None:
        source = Path(r20.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed = {
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
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertLessEqual(imported, allowed)
        self.assertNotIn("pandas", imported)
        self.assertNotIn("smc_trader", imported)
        self.assertNotIn("r19", source.casefold())

    def test_architecture_is_statically_excluded_from_support(self) -> None:
        excluded = set(r20.ARCHITECTURE_EXCLUDED_FROM_SUPPORT)
        semantic_inputs = set(r20.SUPPORT_SEMANTIC_INPUTS)
        self.assertTrue(excluded)
        self.assertTrue(excluded.isdisjoint(semantic_inputs))
        rows = r20.build_support_rows()
        for row in rows:
            row_fields = set(row)
            self.assertTrue(excluded.isdisjoint(row_fields))
            self.assertFalse(
                any(
                    token in field.casefold()
                    for field in row_fields
                    for token in (
                        "architecture",
                        "rosetta",
                        "machine",
                        "macho",
                        "dyld",
                        "cache_family",
                    )
                )
            )

    def test_all_target_ids_and_production_identities_are_exact(self) -> None:
        rows = r20.build_support_rows()
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["target_id"] for row in rows],
            list(EXPECTED_IDENTITIES),
        )
        for row in rows:
            swing_id, bos_id = EXPECTED_IDENTITIES[row["target_id"]]
            self.assertEqual(row["target_swing_id"], swing_id)
            self.assertEqual(row["target_bos_id"], bos_id)
            self.assertEqual(row["expected_matching_swing_count"], 1)
            self.assertEqual(row["expected_matching_bos_count"], 1)
            self.assertEqual(row["expected_admission_count"], 1)

    def test_stage_ordinals_are_exact_for_each_bucket(self) -> None:
        for row in r20.build_support_rows():
            expected = list(
                r20.EXPECTED_STAGE_ORDINALS[row["exact_bucket_enum"]]
            )
            self.assertEqual(row["expected_first_stage_ordinals"], expected)
            self.assertEqual(row["exact_stage_vector"], list(r20.STAGE_NAMES))


class ClockTemplateAndPrefixTests(unittest.TestCase):
    def test_h1_stage_clocks_and_boundary_are_exact(self) -> None:
        clocks = {
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
        for row in r20.build_support_rows():
            if row["timeframe"] == "1H":
                self.assertEqual(
                    row["expected_first_stage_clocks"],
                    clocks[row["exact_bucket_enum"]],
                )
                self.assertEqual(
                    row["expected_boundary_clock"],
                    "2025-01-06T06:00:00-05:00",
                )

    def test_h4_stage_clocks_and_boundary_are_exact(self) -> None:
        clocks = {
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
        for row in r20.build_support_rows():
            if row["timeframe"] == "4H":
                self.assertEqual(
                    row["expected_first_stage_clocks"],
                    clocks[row["exact_bucket_enum"]],
                )
                self.assertEqual(
                    row["expected_boundary_clock"],
                    "2025-01-07T17:00:00-05:00",
                )

    def test_h4_schedule_has_only_the_registered_maintenance_gap(self) -> None:
        intervals = r20.target_intervals("4H")
        self.assertEqual(
            [
                int((end - start).total_seconds() // 60)
                for start, end in intervals
            ],
            [240, 240, 240, 240, 240, 180] * 2,
        )
        self.assertEqual(
            intervals[6][0] - intervals[5][1],
            timedelta(minutes=60),
        )

    def test_literal_mirror_is_exact_and_involutive(self) -> None:
        for candle in (
            *r20.CANONICAL_LONG_PREFIX,
            r20.TERMINAL_CONFIRMED_LONG,
            r20.TERMINAL_WICK_LONG,
            r20.TERMINAL_OPPOSED_SHORT,
        ):
            mirrored = r20.mirror_candle(candle)
            self.assertEqual(r20.mirror_candle(mirrored), candle)
            self.assertEqual(
                mirrored.open_ticks,
                2 * r20.REFERENCE_PRICE_TICKS - candle.open_ticks,
            )
            self.assertEqual(
                mirrored.high_ticks,
                2 * r20.REFERENCE_PRICE_TICKS - candle.low_ticks,
            )
            self.assertEqual(
                mirrored.low_ticks,
                2 * r20.REFERENCE_PRICE_TICKS - candle.high_ticks,
            )
            self.assertEqual(
                mirrored.close_ticks,
                2 * r20.REFERENCE_PRICE_TICKS - candle.close_ticks,
            )

    def test_minute_expansion_reconstructs_frozen_ohlc(self) -> None:
        candle = r20.CANONICAL_LONG_PREFIX[-1]
        start, end = r20.target_intervals("4H")[-1]
        bars = r20.expand_to_minute_bars(candle, start=start, end=end)
        self.assertEqual(len(bars), 180)
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
        self.assertEqual(bars[0]["start"], start.isoformat())
        self.assertEqual(
            bars[-1]["start"],
            (end - timedelta(minutes=1)).isoformat(),
        )

    def test_length_framing_matches_independent_tiny_reference(self) -> None:
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
            encoded = json.dumps(
                bar,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            digest.update(
                len(encoded).to_bytes(8, byteorder="big", signed=False)
            )
            digest.update(encoded)
        self.assertEqual(
            r20.commitment_for_bars(bars),
            digest.hexdigest(),
        )

    def test_every_prefix_commitment_is_reproducible(self) -> None:
        rows = r20.build_support_rows()
        for target, row in zip(r20.TARGETS, rows, strict=True):
            commitment = row["expected_prefix_commitment_sha256"]
            self.assertEqual(
                commitment,
                r20.expected_prefix_commitment(target),
            )
            self.assertEqual(len(commitment), 64)
            int(commitment, 16)


class ParentMarkerAndWriteTests(unittest.TestCase):
    def test_parent_marker_is_required_and_output_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "support_rows.json"
            with self.assertRaises(FileNotFoundError):
                r20.write_support_rows(output)

            (root / "ATTEMPT.json").write_bytes(b"{}\n")
            self.assertEqual(r20.write_support_rows(output), output)
            raw = output.read_bytes()
            self.assertEqual(raw, r20.support_rows_bytes())
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(
                json.loads(raw),
                list(r20.build_support_rows()),
            )
            with self.assertRaises(FileExistsError):
                r20.write_support_rows(output)

    def test_parent_marker_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "marker-target.json"
            target.write_bytes(b"{}\n")
            marker = root / "ATTEMPT.json"
            try:
                marker.symlink_to(target.name)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaises(ValueError):
                r20.write_support_rows(root / "support_rows.json")

    def test_output_name_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ATTEMPT.json").write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                r20.write_support_rows(root / "not-support.json")


if __name__ == "__main__":
    unittest.main()
