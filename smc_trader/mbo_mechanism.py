"""Causal, minute-bounded MBO mechanism features for Phase 6 research.

The module deliberately stops at observable order-flow facts.  In particular,
``T`` records alone define aggressor trade volume, ``F`` records describe the
passive resting-order fills, and ``M.size`` is retained as a reported
replacement size rather than being mislabeled as added or removed liquidity.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifact_stream import sha256_file
from .mbo import (
    F_BAD_TS_RECV,
    F_LAST,
    F_MAYBE_BAD_BOOK,
    F_MBP,
    F_PUBLISHER_SPECIFIC,
    F_SNAPSHOT,
    F_TOB,
    MBORecord,
    MBOReplayError,
    iter_complete_events,
)


MBO_MECHANISM_SCHEMA_VERSION = "phase6_mbo_minute_v1.0"

# This dictionary is part of the artifact identity.  Changing a rule requires
# a version/hash change rather than silently changing an existing feature.
MBO_MECHANISM_PROTOCOL: Mapping[str, Any] = {
    "schema_version": MBO_MECHANISM_SCHEMA_VERSION,
    "availability_clock": "ts_recv",
    "minute_clock": "decision_time=ceil(ts_recv,1m)",
    "trade_volume_source": "T_only",
    "trade_side": {"B": "aggressor_buy", "A": "aggressor_sell", "N": "unknown"},
    "aggressor_volume_imbalance": "(buy-sell)/(buy+sell); zero_without_directional_T",
    "fill_volume_source": "F_only_not_added_to_trade_volume",
    "fill_side": {"B": "passive_bid", "A": "passive_ask", "N": "unknown"},
    "modify_size_meaning": "reported_replacement_size_not_liquidity_delta",
    "excluded_packet_flags": [
        "F_SNAPSHOT",
        "F_BAD_TS_RECV",
        "F_MAYBE_BAD_BOOK",
        "F_TOB",
        "F_MBP",
    ],
    "retained_packet_flag_bits": {
        "F_LAST": F_LAST,
        "F_PUBLISHER_SPECIFIC": F_PUBLISHER_SPECIFIC,
    },
    "publisher_specific_flag": (
        "bit_2_marks_a_publisher_specific_event_and_does_not_invalidate_ts_recv"
    ),
    "reset_packet_excluded": True,
    "bbo_change_requires_consecutive_valid_clocks": True,
    "tick_size": 0.25,
}
MBO_MECHANISM_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        MBO_MECHANISM_PROTOCOL,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

EXCLUDED_FLOW_FLAGS = (
    F_SNAPSHOT | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK | F_TOB | F_MBP
)

FLOW_COMPONENT_COLUMNS = (
    "aggressor_buy_volume",
    "aggressor_sell_volume",
    "aggressor_unknown_volume",
    "aggressor_buy_trade_count",
    "aggressor_sell_trade_count",
    "aggressor_unknown_trade_count",
    "passive_bid_fill_volume",
    "passive_ask_fill_volume",
    "passive_unknown_fill_volume",
    "passive_bid_fill_count",
    "passive_ask_fill_count",
    "passive_unknown_fill_count",
    "displayed_bid_add_volume",
    "displayed_ask_add_volume",
    "displayed_unknown_add_volume",
    "displayed_bid_add_count",
    "displayed_ask_add_count",
    "displayed_unknown_add_count",
    "displayed_bid_cancel_volume",
    "displayed_ask_cancel_volume",
    "displayed_unknown_cancel_volume",
    "displayed_bid_cancel_count",
    "displayed_ask_cancel_count",
    "displayed_unknown_cancel_count",
    "displayed_bid_modify_reported_size",
    "displayed_ask_modify_reported_size",
    "displayed_unknown_modify_reported_size",
    "displayed_bid_modify_count",
    "displayed_ask_modify_count",
    "displayed_unknown_modify_count",
)

FLOW_DERIVED_COLUMNS = (
    "aggressor_trade_volume",
    "aggressor_trade_count",
    "aggressor_net_volume",
    "aggressor_volume_imbalance",
    "aggressor_contracts_per_second",
    "aggressor_trades_per_second",
)

BBO_SOURCE_COLUMNS = (
    "book_observed_at",
    "publisher_id",
    "sequence",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "top5_bid_size",
    "top5_ask_size",
    "depth_imbalance",
    "book_valid",
    "invalid_reason",
)

BBO_DERIVED_COLUMNS = (
    "book_age_seconds",
    "mid_price",
    "spread_points",
    "spread_ticks",
    "book_change_valid",
    "best_level_ofi_contracts",
    "depth_imbalance_change",
    "mid_change_points",
    "mid_change_ticks",
    "absolute_mid_impact_ticks_per_aggressor_contract",
    "signed_mid_impact_ticks_per_net_aggressor_contract",
    "book_valid_clock_fraction",
)

MBO_MECHANISM_COLUMNS = (
    "decision_time",
    "symbol",
    "instrument_id",
    *FLOW_COMPONENT_COLUMNS,
    *FLOW_DERIVED_COLUMNS,
    *BBO_SOURCE_COLUMNS,
    *BBO_DERIVED_COLUMNS,
)

COUNT_COLUMNS = tuple(
    column
    for column in FLOW_COMPONENT_COLUMNS
    if column.endswith("_count")
) + ("aggressor_trade_count",)


class MBOMechanismArtifactError(ValueError):
    """Raised when a Phase 6 minute artifact is not identity-safe."""


def completed_minute_clock(value: Any) -> pd.Timestamp:
    """Return the completed minute clock using the causal receive timestamp."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise MBOReplayError("MBO ts_recv must be timezone aware")
    # MBOBook replay flushes clocks only before the next receive timestamp, so
    # an event exactly at a decision clock is already knowable at that clock.
    return timestamp.tz_convert("UTC").ceil("min")


def _side_label(side: str) -> str:
    return {"B": "bid", "A": "ask", "N": "unknown"}.get(str(side), "")


def flow_updates(action: str, side: str, size: float) -> dict[str, float | int]:
    """Map one eligible MBO record to unambiguous additive minute facts."""

    action = str(action).strip().upper()
    side = str(side).strip().upper()
    size = float(size)
    if not math.isfinite(size) or size < 0:
        raise MBOReplayError("MBO flow record contains invalid size")
    label = _side_label(side)
    if not label:
        raise MBOReplayError(f"unknown MBO side: {side}")
    if action == "T":
        aggressor = {"B": "buy", "A": "sell", "N": "unknown"}[side]
        return {
            f"aggressor_{aggressor}_volume": size,
            f"aggressor_{aggressor}_trade_count": 1,
        }
    if action == "F":
        return {
            f"passive_{label}_fill_volume": size,
            f"passive_{label}_fill_count": 1,
        }
    if action in {"A", "C", "M"}:
        operation = {"A": "add", "C": "cancel", "M": "modify"}[action]
        size_name = "reported_size" if action == "M" else "volume"
        return {
            f"displayed_{label}_{operation}_{size_name}": size,
            f"displayed_{label}_{operation}_count": 1,
        }
    if action in {"R", "N"}:
        return {}
    raise MBOReplayError(f"unknown MBO action: {action}")


def aggregate_flow_records(
    records: Iterable[MBORecord],
    *,
    decision_times: Sequence[Any],
    instrument_id: int,
) -> pd.DataFrame:
    """Reference packet-preserving flow aggregation used by tests/small inputs.

    Production raw partitions use the vectorized materializer, but both paths
    share :func:`flow_updates` and the same packet exclusion semantics.
    """

    clocks = pd.DatetimeIndex(pd.to_datetime(list(decision_times), utc=True))
    if clocks.has_duplicates:
        raise ValueError("decision clocks must be unique")
    allowed = set(clocks)
    values: dict[pd.Timestamp, dict[str, float]] = {
        clock: {column: 0.0 for column in FLOW_COMPONENT_COLUMNS}
        for clock in clocks
    }
    for packet in iter_complete_events(records):
        selected = tuple(
            record for record in packet if int(record.instrument_id) == int(instrument_id)
        )
        if not selected:
            continue
        flags = 0
        for record in selected:
            flags |= int(record.flags)
        if flags & EXCLUDED_FLOW_FLAGS or any(record.action == "R" for record in selected):
            continue
        for record in selected:
            clock = completed_minute_clock(record.ts_recv)
            if clock not in allowed:
                continue
            for column, change in flow_updates(record.action, record.side, record.size).items():
                values[clock][column] += float(change)
    rows = [
        {"decision_time": clock, **values[clock]}
        for clock in clocks
    ]
    return derive_flow_features(pd.DataFrame(rows))


def derive_flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing flow components and derive T-only imbalance/velocity."""

    values = pd.DataFrame(frame).copy()
    if "decision_time" not in values:
        raise ValueError("flow frame requires decision_time")
    values["decision_time"] = pd.to_datetime(
        values["decision_time"], errors="coerce", utc=True
    )
    if values["decision_time"].isna().any() or values["decision_time"].duplicated().any():
        raise ValueError("flow frame contains invalid or duplicate decision clocks")
    for column in FLOW_COMPONENT_COLUMNS:
        if column not in values:
            values[column] = 0.0
        values[column] = pd.to_numeric(values[column], errors="coerce").fillna(0.0)
        if (values[column] < 0).any() or not np.isfinite(values[column]).all():
            raise ValueError(f"flow component is invalid: {column}")
    values["aggressor_trade_volume"] = values[
        [
            "aggressor_buy_volume",
            "aggressor_sell_volume",
            "aggressor_unknown_volume",
        ]
    ].sum(axis=1)
    values["aggressor_trade_count"] = values[
        [
            "aggressor_buy_trade_count",
            "aggressor_sell_trade_count",
            "aggressor_unknown_trade_count",
        ]
    ].sum(axis=1)
    values["aggressor_net_volume"] = (
        values["aggressor_buy_volume"] - values["aggressor_sell_volume"]
    )
    directional = values["aggressor_buy_volume"] + values["aggressor_sell_volume"]
    values["aggressor_volume_imbalance"] = np.divide(
        values["aggressor_net_volume"],
        directional,
        out=np.zeros(len(values), dtype=float),
        where=directional.to_numpy(dtype=float) > 0,
    )
    values["aggressor_contracts_per_second"] = (
        values["aggressor_trade_volume"] / 60.0
    )
    values["aggressor_trades_per_second"] = (
        values["aggressor_trade_count"] / 60.0
    )
    for column in COUNT_COLUMNS:
        rounded = np.rint(values[column].to_numpy(dtype=float))
        if not np.allclose(values[column], rounded, atol=0.0):
            raise ValueError(f"flow count is not integral: {column}")
        values[column] = rounded.astype("int64")
    return values


def _best_level_ofi(
    bid: pd.Series,
    ask: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    previous_bid = bid.shift(1)
    previous_ask = ask.shift(1)
    previous_bid_size = bid_size.shift(1)
    previous_ask_size = ask_size.shift(1)
    bid_change = np.select(
        [bid > previous_bid, bid == previous_bid, bid < previous_bid],
        [bid_size, bid_size - previous_bid_size, -previous_bid_size],
        default=np.nan,
    )
    ask_change = np.select(
        [ask < previous_ask, ask == previous_ask, ask > previous_ask],
        [-ask_size, previous_ask_size - ask_size, previous_ask_size],
        default=np.nan,
    )
    return pd.Series(bid_change + ask_change, index=bid.index, dtype="float64")


def build_minute_mechanism_frame(
    clocks: pd.DataFrame,
    flow: pd.DataFrame,
    bbo: pd.DataFrame,
    *,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """Join exact causal clocks with flow and BBO, then derive mechanisms."""

    if not math.isfinite(float(tick_size)) or float(tick_size) <= 0:
        raise ValueError("tick_size must be positive")
    clock_values = pd.DataFrame(clocks).copy()
    required_clock = {"decision_time", "symbol", "instrument_id"}
    missing = sorted(required_clock - set(clock_values))
    if missing:
        raise ValueError(f"clock frame is missing fields: {missing}")
    clock_values["decision_time"] = pd.to_datetime(
        clock_values["decision_time"], errors="coerce", utc=True
    )
    if (
        clock_values["decision_time"].isna().any()
        or clock_values["decision_time"].duplicated().any()
    ):
        raise ValueError("clock frame contains invalid or duplicate clocks")
    clock_values = clock_values.sort_values("decision_time", kind="stable")

    flow_values = derive_flow_features(flow)
    unknown_flow = set(flow_values["decision_time"]) - set(clock_values["decision_time"])
    if unknown_flow:
        raise ValueError("flow frame contains clocks outside the causal OHLCV clock")
    values = clock_values.merge(
        flow_values,
        on="decision_time",
        how="left",
        validate="one_to_one",
    )
    for column in (*FLOW_COMPONENT_COLUMNS, *FLOW_DERIVED_COLUMNS):
        values[column] = values[column].fillna(0)

    bbo_values = pd.DataFrame(bbo).copy()
    required_bbo = {"decision_time", "symbol", "instrument_id", *BBO_SOURCE_COLUMNS}
    missing = sorted(required_bbo - set(bbo_values))
    if missing:
        raise ValueError(f"BBO frame is missing fields: {missing}")
    bbo_values["decision_time"] = pd.to_datetime(
        bbo_values["decision_time"], errors="coerce", utc=True
    )
    bbo_values["book_observed_at"] = pd.to_datetime(
        bbo_values["book_observed_at"], errors="coerce", utc=True
    )
    if bbo_values["decision_time"].isna().any() or bbo_values["decision_time"].duplicated().any():
        raise ValueError("BBO frame contains invalid or duplicate clocks")
    if set(bbo_values["decision_time"]) != set(clock_values["decision_time"]):
        raise ValueError("BBO clocks do not exactly match causal OHLCV clocks")
    if not bbo_values[["decision_time", "symbol", "instrument_id"]].sort_values(
        "decision_time", kind="stable"
    ).reset_index(drop=True).equals(
        clock_values[["decision_time", "symbol", "instrument_id"]].reset_index(drop=True)
    ):
        raise ValueError("BBO symbol/instrument identity differs from OHLCV clock")
    bbo_values = bbo_values.drop(columns=["symbol", "instrument_id"])
    values = values.merge(
        bbo_values,
        on="decision_time",
        how="left",
        validate="one_to_one",
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)

    values["book_valid"] = values["book_valid"].fillna(False).astype(bool)
    valid = values["book_valid"]
    numeric_book = (
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "top5_bid_size",
        "top5_ask_size",
        "depth_imbalance",
    )
    for column in numeric_book:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    if (
        values.loc[valid, "book_observed_at"].isna().any()
        or values.loc[valid, list(numeric_book)].isna().any().any()
        or (values.loc[valid, "book_observed_at"] > values.loc[valid, "decision_time"]).any()
        or (values.loc[valid, "bid"] >= values.loc[valid, "ask"]).any()
        or (values.loc[valid, ["bid_size", "ask_size", "top5_bid_size", "top5_ask_size"]] < 0)
        .any()
        .any()
    ):
        raise ValueError("BBO source marks an invalid observation as valid")

    values["book_age_seconds"] = (
        values["decision_time"] - values["book_observed_at"]
    ).dt.total_seconds()
    values["mid_price"] = (values["bid"] + values["ask"]) / 2.0
    values["spread_points"] = values["ask"] - values["bid"]
    values["spread_ticks"] = values["spread_points"] / float(tick_size)
    consecutive = values["decision_time"].diff().eq(pd.Timedelta(minutes=1))
    pair_valid = valid & valid.shift(1, fill_value=False) & consecutive
    values["book_change_valid"] = pair_valid
    ofi = _best_level_ofi(
        values["bid"], values["ask"], values["bid_size"], values["ask_size"]
    )
    values["best_level_ofi_contracts"] = ofi.where(pair_valid)
    values["depth_imbalance_change"] = values["depth_imbalance"].diff().where(pair_valid)
    values["mid_change_points"] = values["mid_price"].diff().where(pair_valid)
    values["mid_change_ticks"] = values["mid_change_points"] / float(tick_size)
    values["absolute_mid_impact_ticks_per_aggressor_contract"] = np.divide(
        values["mid_change_ticks"].abs(),
        values["aggressor_trade_volume"],
        out=np.full(len(values), np.nan, dtype=float),
        where=(
            values["mid_change_ticks"].notna()
            & values["aggressor_trade_volume"].gt(0)
        ).to_numpy(),
    )
    values["signed_mid_impact_ticks_per_net_aggressor_contract"] = np.divide(
        values["mid_change_ticks"],
        values["aggressor_net_volume"],
        out=np.full(len(values), np.nan, dtype=float),
        where=(
            values["mid_change_ticks"].notna()
            & values["aggressor_net_volume"].ne(0)
        ).to_numpy(),
    )
    values["book_valid_clock_fraction"] = valid.astype(float)
    return values.loc[:, list(MBO_MECHANISM_COLUMNS)]


def validate_mbo_mechanism_frame(
    frame: pd.DataFrame,
    *,
    expected_start: Any | None = None,
    expected_end: Any | None = None,
    expected_symbol: str | None = None,
    expected_instrument_id: int | None = None,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Validate and normalize the public minute-artifact contract."""

    values = pd.DataFrame(frame).copy()
    missing = sorted(set(MBO_MECHANISM_COLUMNS) - set(values))
    extra = sorted(set(values) - set(MBO_MECHANISM_COLUMNS))
    if missing or extra:
        raise MBOMechanismArtifactError(
            f"MBO mechanism schema differs: missing={missing}, extra={extra}"
        )
    values = values.loc[:, list(MBO_MECHANISM_COLUMNS)]
    values["decision_time"] = pd.to_datetime(
        values["decision_time"], errors="coerce", utc=True
    )
    values["book_observed_at"] = pd.to_datetime(
        values["book_observed_at"], errors="coerce", utc=True
    )
    if (
        values["decision_time"].isna().any()
        or values["decision_time"].duplicated().any()
        or not values["decision_time"].is_monotonic_increasing
    ):
        raise MBOMechanismArtifactError("artifact decision clocks are not unique/ordered")
    if expected_rows is not None and len(values) != int(expected_rows):
        raise MBOMechanismArtifactError("artifact row count differs from registered count")
    if expected_start is not None:
        start = pd.Timestamp(expected_start)
        if start.tzinfo is None:
            raise ValueError("expected_start must be timezone aware")
        if (values["decision_time"] < start.tz_convert("UTC")).any():
            raise MBOMechanismArtifactError("artifact contains a pre-window clock")
    if expected_end is not None:
        end = pd.Timestamp(expected_end)
        if end.tzinfo is None:
            raise ValueError("expected_end must be timezone aware")
        if (values["decision_time"] >= end.tz_convert("UTC")).any():
            raise MBOMechanismArtifactError("artifact contains a post-window clock")
    if expected_symbol is not None and set(values["symbol"].astype(str)) != {
        str(expected_symbol)
    }:
        raise MBOMechanismArtifactError("artifact symbol differs from registered contract")
    values["instrument_id"] = pd.to_numeric(
        values["instrument_id"], errors="raise"
    ).astype("int64")
    if expected_instrument_id is not None and set(values["instrument_id"]) != {
        int(expected_instrument_id)
    }:
        raise MBOMechanismArtifactError(
            "artifact instrument_id differs from registered contract"
        )

    numeric = [
        column
        for column in MBO_MECHANISM_COLUMNS
        if column
        not in {
            "decision_time",
            "symbol",
            "instrument_id",
            "book_observed_at",
            "book_valid",
            "book_change_valid",
            "invalid_reason",
        }
    ]
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        if np.isinf(values[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise MBOMechanismArtifactError(f"artifact contains infinity: {column}")
    for column in FLOW_COMPONENT_COLUMNS:
        if values[column].isna().any() or (values[column] < 0).any():
            raise MBOMechanismArtifactError(f"artifact flow component is invalid: {column}")
    for column in COUNT_COLUMNS:
        if not np.allclose(values[column], np.rint(values[column]), atol=0.0):
            raise MBOMechanismArtifactError(f"artifact count is not integral: {column}")
    expected_trade_volume = values[
        ["aggressor_buy_volume", "aggressor_sell_volume", "aggressor_unknown_volume"]
    ].sum(axis=1)
    expected_trade_count = values[
        [
            "aggressor_buy_trade_count",
            "aggressor_sell_trade_count",
            "aggressor_unknown_trade_count",
        ]
    ].sum(axis=1)
    if not np.allclose(values["aggressor_trade_volume"], expected_trade_volume):
        raise MBOMechanismArtifactError("T-only aggressor volume identity failed")
    if not np.allclose(values["aggressor_trade_count"], expected_trade_count):
        raise MBOMechanismArtifactError("T-only aggressor count identity failed")
    if not np.allclose(
        values["aggressor_net_volume"],
        values["aggressor_buy_volume"] - values["aggressor_sell_volume"],
    ):
        raise MBOMechanismArtifactError("aggressor net-volume identity failed")
    valid = values["book_valid"].fillna(False).astype(bool)
    change_valid = values["book_change_valid"].fillna(False).astype(bool)
    change_columns = (
        "best_level_ofi_contracts",
        "depth_imbalance_change",
        "mid_change_points",
        "mid_change_ticks",
    )
    if (
        values.loc[valid, "book_observed_at"].isna().any()
        or (values.loc[valid, "book_observed_at"] > values.loc[valid, "decision_time"]).any()
        or (values.loc[valid, "bid"] >= values.loc[valid, "ask"]).any()
        or (values.loc[valid, "book_age_seconds"] < 0).any()
        or not values["book_valid_clock_fraction"].isin({0.0, 1.0}).all()
        or not np.array_equal(
            values["book_valid_clock_fraction"].to_numpy(dtype=float),
            valid.astype(float).to_numpy(),
        )
        or values.loc[change_valid, list(change_columns)].isna().any().any()
        or values.loc[~change_valid, list(change_columns)].notna().any().any()
    ):
        raise MBOMechanismArtifactError("artifact contains invalid causal BBO fields")
    values["book_valid"] = valid
    values["book_change_valid"] = change_valid
    for column in COUNT_COLUMNS:
        values[column] = values[column].astype("int64")
    return values


def _resolve_bound_path(path: str, manifest_path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    # Materializers write repository-relative paths from the project root.
    repository = Path(__file__).resolve().parents[1]
    candidate = repository / value
    return candidate if candidate.exists() else manifest_path.parent / value


def load_mbo_mechanism_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    verify_lineage: bool = True,
    expected_manifest_sha256: str | None = None,
    expected_start: Any | None = None,
    expected_end: Any | None = None,
    expected_symbol: str | None = None,
    expected_instrument_id: int | None = None,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Load a hash-bound Phase 6 artifact and fail closed on identity drift."""

    source = Path(path)
    manifest_source = (
        Path(manifest_path)
        if manifest_path is not None
        else source.with_suffix(source.suffix + ".manifest.json")
    )
    if not source.is_file() or source.is_symlink():
        raise MBOMechanismArtifactError("MBO mechanism artifact is not a regular file")
    if not manifest_source.is_file() or manifest_source.is_symlink():
        raise MBOMechanismArtifactError("MBO mechanism manifest is not a regular file")
    if (
        expected_manifest_sha256 is not None
        and sha256_file(manifest_source) != str(expected_manifest_sha256)
    ):
        raise MBOMechanismArtifactError("MBO mechanism manifest SHA-256 mismatch")
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise MBOMechanismArtifactError("MBO mechanism manifest root is invalid")
    if (
        manifest.get("artifact_kind") != "mbo_minute_mechanism_features"
        or manifest.get("feature_schema_version") != MBO_MECHANISM_SCHEMA_VERSION
        or manifest.get("feature_protocol_sha256")
        != MBO_MECHANISM_PROTOCOL_SHA256
    ):
        raise MBOMechanismArtifactError("MBO mechanism protocol identity mismatch")
    output = manifest.get("output")
    if not isinstance(output, Mapping) or output.get("sha256") != sha256_file(source):
        raise MBOMechanismArtifactError("MBO mechanism output SHA-256 mismatch")
    if verify_lineage:
        bindings = manifest.get("load_verified_lineage")
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
            raise MBOMechanismArtifactError("MBO mechanism lineage bindings are absent")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise MBOMechanismArtifactError("MBO mechanism lineage binding is invalid")
            bound = _resolve_bound_path(str(binding.get("path", "")), manifest_source)
            if (
                not bound.is_file()
                or bound.is_symlink()
                or sha256_file(bound) != str(binding.get("sha256", ""))
            ):
                raise MBOMechanismArtifactError(
                    f"MBO mechanism lineage SHA-256 mismatch: {binding.get('role', '')}"
                )
    frame = validate_mbo_mechanism_frame(
        pd.read_parquet(source),
        expected_start=(
            expected_start if expected_start is not None else manifest.get("start")
        ),
        expected_end=(expected_end if expected_end is not None else manifest.get("end_exclusive")),
        expected_symbol=(
            expected_symbol if expected_symbol is not None else manifest.get("symbol")
        ),
        expected_instrument_id=(
            expected_instrument_id
            if expected_instrument_id is not None
            else manifest.get("instrument_id")
        ),
        expected_rows=(expected_rows if expected_rows is not None else output.get("rows")),
    )
    return frame


__all__ = [
    "BBO_DERIVED_COLUMNS",
    "BBO_SOURCE_COLUMNS",
    "EXCLUDED_FLOW_FLAGS",
    "FLOW_COMPONENT_COLUMNS",
    "FLOW_DERIVED_COLUMNS",
    "MBOMechanismArtifactError",
    "MBO_MECHANISM_COLUMNS",
    "MBO_MECHANISM_PROTOCOL",
    "MBO_MECHANISM_PROTOCOL_SHA256",
    "MBO_MECHANISM_SCHEMA_VERSION",
    "aggregate_flow_records",
    "build_minute_mechanism_frame",
    "completed_minute_clock",
    "derive_flow_features",
    "flow_updates",
    "load_mbo_mechanism_artifact",
    "validate_mbo_mechanism_frame",
]
