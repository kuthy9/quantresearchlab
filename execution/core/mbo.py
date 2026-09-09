"""Memory-bounded MBO book reconstruction and minute execution reality."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from .execution import TopOfBook, TopOfBookExecutionProvider
from contract.market import (
    Bar,
    aware_timestamp,
)
from .execution import ExecutionRealityInput


F_LAST = 128
F_TOB = 64
F_SNAPSHOT = 32
F_MBP = 16
F_BAD_TS_RECV = 8
F_MAYBE_BAD_BOOK = 4
F_PUBLISHER_SPECIFIC = 2
VALID_ACTIONS = frozenset({"A", "M", "C", "R", "T", "F", "N"})
VALID_SIDES = frozenset({"A", "B", "N"})
MBO_COLUMNS = (
    "ts_recv",
    "ts_event",
    "publisher_id",
    "instrument_id",
    "action",
    "side",
    "price",
    "size",
    "order_id",
    "flags",
    "sequence",
)


class MBOReplayError(RuntimeError):
    """Raised when MBO ordering or schema cannot support a causal quote."""


def assert_mbo_source_allowed(
    source: str | Path,
    *,
    allow_sealed_holdout: bool = False,
) -> Path:
    """Reject a sealed tree before opening market-data contents."""

    path = Path(source).expanduser().resolve(strict=False)
    start = path if path.is_dir() else path.parent
    if not allow_sealed_holdout:
        for directory in (start, *start.parents):
            if (directory / ".HOLDOUT_SEALED").exists():
                raise MBOReplayError(
                    "sealed MBO holdout cannot be read before the explicit reveal"
                )
    return path


def _enum_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii").strip().upper()
    text = str(value).strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if text.startswith("B'") and text.endswith("'"):
        text = text[2:-1]
    return text


def _action_code(value: Any) -> str:
    if isinstance(value, Integral):
        integer = int(value)
        if integer in range(7):
            return ("A", "C", "M", "R", "T", "F", "N")[integer]
        if 0 <= integer <= 255:
            return chr(integer).strip().upper()
    text = _enum_text(value)
    return {
        "ADD": "A",
        "CANCEL": "C",
        "MODIFY": "M",
        "CLEAR": "R",
        "CLEARBOOK": "R",
        "CLEAR_BOOK": "R",
        "TRADE": "T",
        "FILL": "F",
        "NONE": "N",
    }.get(text, text)


def _side_code(value: Any) -> str:
    if isinstance(value, Integral):
        integer = int(value)
        if integer in range(3):
            return ("B", "A", "N")[integer]
        if 0 <= integer <= 255:
            return chr(integer).strip().upper()
    text = _enum_text(value)
    return {
        "ASK": "A",
        "BID": "B",
        "NONE": "N",
    }.get(text, text)


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    if isinstance(value, Integral):
        timestamp = pd.Timestamp(int(value), unit="ns", tz="UTC")
    else:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
    return aware_timestamp(timestamp, name=name)


def _price(value: Any, *, fixed_point: bool) -> float:
    result = float(value)
    if fixed_point:
        result /= 1_000_000_000.0
    if not math.isfinite(result) or result <= 0:
        raise MBOReplayError("MBO record contains an invalid price")
    return result


@dataclass(frozen=True)
class MBORecord:
    ts_recv: pd.Timestamp
    ts_event: pd.Timestamp
    publisher_id: int
    instrument_id: int
    action: str
    side: str
    price: float | None
    size: float
    order_id: int
    flags: int
    sequence: int

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        fixed_point_price: bool = False,
    ) -> "MBORecord":
        get = (
            value.get
            if isinstance(value, Mapping)
            else lambda name, default=None: getattr(value, name, default)
        )
        action = _action_code(get("action"))
        side = _side_code(get("side"))
        if action not in VALID_ACTIONS or side not in VALID_SIDES:
            raise MBOReplayError(f"unknown MBO action/side: {action}/{side}")
        raw_price = get("price")
        price = (
            None
            if action == "R" or raw_price is None or pd.isna(raw_price)
            else _price(raw_price, fixed_point=fixed_point_price)
        )
        size = float(get("size", 0.0))
        if not math.isfinite(size) or size < 0:
            raise MBOReplayError("MBO record contains invalid size")
        return cls(
            ts_recv=_timestamp(get("ts_recv"), name="mbo.ts_recv"),
            ts_event=_timestamp(get("ts_event"), name="mbo.ts_event"),
            publisher_id=int(get("publisher_id")),
            instrument_id=int(get("instrument_id")),
            action=action,
            side=side,
            price=price,
            size=size,
            order_id=int(get("order_id", 0)),
            flags=int(get("flags")),
            sequence=int(get("sequence")),
        )


@dataclass(frozen=True)
class RestingOrder:
    side: str
    price: float
    size: float


@dataclass(frozen=True)
class MBOBookSnapshot:
    observed_at: pd.Timestamp
    publisher_id: int
    instrument_id: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    top5_bid_size: float
    top5_ask_size: float
    depth_imbalance: float
    sequence: int


class MBOOrderBook:
    """Fail-closed level-3 book for one publisher/instrument stream."""

    def __init__(self, publisher_id: int, instrument_id: int) -> None:
        self.publisher_id = int(publisher_id)
        self.instrument_id = int(instrument_id)
        self.orders: dict[int, RestingOrder] = {}
        self.level_sizes: dict[str, dict[float, float]] = {"B": {}, "A": {}}
        self._bid_heap: list[float] = []
        self._ask_heap: list[float] = []
        self.requires_snapshot = True
        self.valid = False
        self.invalid_reason = "snapshot_missing"
        self.last_recv: pd.Timestamp | None = None
        self.last_sequence: int | None = None

    def _reset(self) -> None:
        self.orders.clear()
        self.level_sizes = {"B": {}, "A": {}}
        self._bid_heap.clear()
        self._ask_heap.clear()

    def _invalidate(self, reason: str, *, require_snapshot: bool = True) -> None:
        self.valid = False
        self.invalid_reason = reason
        if require_snapshot:
            self.requires_snapshot = True

    def _change_level(self, side: str, price: float, change: float) -> None:
        levels = self.level_sizes[side]
        value = levels.get(price, 0.0) + change
        if value < -1e-9:
            raise MBOReplayError("negative price-level size")
        if value <= 1e-9:
            levels.pop(price, None)
            return
        if price not in levels:
            heapq.heappush(
                self._bid_heap if side == "B" else self._ask_heap,
                -price if side == "B" else price,
            )
        levels[price] = value

    def _add(self, record: MBORecord) -> None:
        if (
            record.order_id <= 0
            or record.order_id in self.orders
            or record.side not in {"A", "B"}
            or record.price is None
            or record.size <= 0
        ):
            raise MBOReplayError("invalid or duplicate add")
        order = RestingOrder(record.side, record.price, record.size)
        self.orders[record.order_id] = order
        self._change_level(order.side, order.price, order.size)

    def _remove(self, order_id: int) -> RestingOrder:
        order = self.orders.pop(order_id)
        self._change_level(order.side, order.price, -order.size)
        return order

    def _apply(self, record: MBORecord) -> None:
        if record.action == "R":
            self._reset()
            return
        if record.action == "A":
            self._add(record)
            return
        if record.action not in {"M", "C"}:
            return
        current = self.orders.get(record.order_id)
        if current is None:
            raise MBOReplayError(f"unknown_{record.action.lower()}")
        if current.side != record.side or record.price is None:
            raise MBOReplayError("MBO mutation changes side or lacks price")
        if record.action == "C":
            if not math.isclose(current.price, record.price, abs_tol=1e-9):
                raise MBOReplayError("cancel price mismatch")
            if record.size > current.size + 1e-9:
                raise MBOReplayError("cancel exceeds resting size")
            remaining = current.size - record.size
            self._remove(record.order_id)
            if remaining > 1e-9:
                amended = MBORecord(
                    **{
                        **record.__dict__,
                        "action": "A",
                        "size": remaining,
                    }
                )
                self._add(amended)
            return
        self._remove(record.order_id)
        if record.size > 0:
            self._add(
                MBORecord(
                    **{
                        **record.__dict__,
                        "action": "A",
                    }
                )
            )

    def _best_bid(self) -> float | None:
        while self._bid_heap and -self._bid_heap[0] not in self.level_sizes["B"]:
            heapq.heappop(self._bid_heap)
        return -self._bid_heap[0] if self._bid_heap else None

    def _best_ask(self) -> float | None:
        while self._ask_heap and self._ask_heap[0] not in self.level_sizes["A"]:
            heapq.heappop(self._ask_heap)
        return self._ask_heap[0] if self._ask_heap else None

    def apply_complete_event(
        self,
        records: Sequence[MBORecord],
        *,
        capture_snapshot: bool = True,
    ) -> MBOBookSnapshot | None:
        if not records or not records[-1].flags & F_LAST:
            self._invalidate("incomplete_vendor_event")
            return None
        if {
            (record.publisher_id, record.instrument_id) for record in records
        } != {(self.publisher_id, self.instrument_id)}:
            self._invalidate("mixed_book_key")
            return None
        receive_times = [record.ts_recv for record in records]
        if any(right < left for left, right in zip(receive_times, receive_times[1:])):
            self._invalidate("receive_time_regression")
            return None
        if self.last_recv is not None and receive_times[0] < self.last_recv:
            self._invalidate("stream_receive_time_regression")
            return None
        self.last_recv = receive_times[-1]
        self.last_sequence = records[-1].sequence
        flags = 0
        for record in records:
            flags |= record.flags
        is_snapshot = bool(flags & F_SNAPSHOT)
        if flags & F_MAYBE_BAD_BOOK:
            self._invalidate("maybe_bad_book_flag")
            return None
        if flags & (F_TOB | F_MBP):
            self._invalidate("aggregate_record_in_mbo_stream")
            return None
        if flags & F_BAD_TS_RECV and not is_snapshot:
            self._invalidate("bad_receive_time_flag")
            return None
        actions = [record.action for record in records]
        has_reset = "R" in actions
        if is_snapshot:
            if (
                actions[0] != "R"
                or actions.count("R") != 1
                or any(action not in {"R", "A"} for action in actions[1:])
                or not records[1:]
                or not all(
                    record.flags & F_SNAPSHOT for record in records[1:]
                )
            ):
                self._invalidate("malformed_snapshot")
                return None
            self._reset()
            self.requires_snapshot = False
        elif has_reset:
            if actions[0] != "R" or actions.count("R") != 1:
                self._invalidate("malformed_reset")
                return None
            self._reset()
            self.requires_snapshot = False
        if self.requires_snapshot:
            return None
        try:
            for record in records:
                self._apply(record)
        except (KeyError, MBOReplayError) as error:
            self._invalidate(str(error))
            return None
        bid = self._best_bid()
        ask = self._best_ask()
        if bid is None or ask is None or bid >= ask:
            self._invalidate("crossed_or_incomplete_book", require_snapshot=False)
            return None
        self.valid = True
        self.invalid_reason = ""
        return self.snapshot() if capture_snapshot else None

    def snapshot(self) -> MBOBookSnapshot | None:
        """Capture current BBO/depth without replaying another event."""

        if (
            not self.valid
            or self.last_recv is None
            or self.last_sequence is None
        ):
            return None
        bid = self._best_bid()
        ask = self._best_ask()
        if bid is None or ask is None or bid >= ask:
            self._invalidate("crossed_or_incomplete_book", require_snapshot=False)
            return None
        bid_prices = sorted(self.level_sizes["B"], reverse=True)[:5]
        ask_prices = sorted(self.level_sizes["A"])[:5]
        top5_bid = sum(self.level_sizes["B"][price] for price in bid_prices)
        top5_ask = sum(self.level_sizes["A"][price] for price in ask_prices)
        total = top5_bid + top5_ask
        return MBOBookSnapshot(
            observed_at=self.last_recv,
            publisher_id=self.publisher_id,
            instrument_id=self.instrument_id,
            bid=float(bid),
            ask=float(ask),
            bid_size=float(self.level_sizes["B"][bid]),
            ask_size=float(self.level_sizes["A"][ask]),
            top5_bid_size=float(top5_bid),
            top5_ask_size=float(top5_ask),
            depth_imbalance=float((top5_bid - top5_ask) / total),
            sequence=self.last_sequence,
        )


def iter_complete_events(records: Iterable[MBORecord]) -> Iterator[tuple[MBORecord, ...]]:
    pending: list[MBORecord] = []
    for record in records:
        pending.append(record)
        if record.flags & F_LAST:
            yield tuple(pending)
            pending.clear()
    if pending:
        raise MBOReplayError("MBO source ends inside an incomplete vendor event")


def iter_parquet_mbo_records(
    paths: Sequence[str | Path],
    *,
    instrument_ids: Sequence[int],
    batch_size: int = 100_000,
    progress: Callable[[Path], None] | None = None,
) -> Iterator[MBORecord]:
    """Stream selected instruments in source order from daily Parquet files."""

    import pyarrow.dataset as ds

    selected = tuple(sorted({int(value) for value in instrument_ids}))
    if not selected:
        return
    for value in sorted(Path(path) for path in paths):
        path = assert_mbo_source_allowed(value)
        dataset = ds.dataset(path, format="parquet")
        missing = sorted(set(MBO_COLUMNS) - set(dataset.schema.names))
        if missing:
            raise MBOReplayError(f"{path} is missing MBO fields: {missing}")
        scanner = dataset.scanner(
            columns=list(MBO_COLUMNS),
            filter=ds.field("instrument_id").isin(selected),
            batch_size=int(batch_size),
            use_threads=False,
        )
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            for row in frame.itertuples(index=False):
                yield MBORecord.from_value(row)
        if progress is not None:
            progress(path)


def iter_dbn_mbo_records(
    path: str | Path,
    *,
    instrument_ids: Sequence[int],
    allow_sealed_holdout: bool = False,
) -> Iterator[MBORecord]:
    """Stream a compressed DBN store record-by-record without ``to_df``."""

    try:
        import databento as db
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("DBN MBO replay requires the 'dbn' optional dependency") from error
    from shares.core.io import require_materialized

    source = assert_mbo_source_allowed(
        path,
        allow_sealed_holdout=allow_sealed_holdout,
    )
    require_materialized(source)
    selected = {int(value) for value in instrument_ids}
    store = db.DBNStore.from_file(source)
    for value in store:
        if not hasattr(value, "action") or not hasattr(value, "instrument_id"):
            continue
        if int(value.instrument_id) not in selected:
            continue
        yield MBORecord.from_value(value, fixed_point_price=True)


class MinuteExecutionRealityStore:
    """Exact-clock lookup for pre-materialized MBO minute observations."""

    REQUIRED = {
        "decision_time",
        "instrument_id",
        "book_observed_at",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "book_valid",
    }

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        commission_per_side: float = 2.25,
        tick_size: float = 0.25,
    ) -> None:
        values = pd.DataFrame(frame).copy()
        missing = sorted(self.REQUIRED - set(values))
        if missing:
            raise ValueError(f"MBO minute store is missing fields: {missing}")
        values["decision_time"] = pd.to_datetime(
            values["decision_time"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert("America/New_York")
        values["book_observed_at"] = pd.to_datetime(
            values["book_observed_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert("America/New_York")
        values["book_valid"] = values["book_valid"].fillna(False).astype(bool)
        values["instrument_id"] = pd.to_numeric(
            values["instrument_id"],
            errors="raise",
        ).astype("int64")
        if values["decision_time"].isna().any() or values["decision_time"].duplicated().any():
            raise ValueError("MBO minute store has invalid or duplicate decision clocks")
        valid = values["book_valid"]
        numeric_columns = ("bid", "ask", "bid_size", "ask_size")
        for column in numeric_columns:
            values[column] = pd.to_numeric(values[column], errors="coerce")
        if (
            values.loc[valid, "book_observed_at"].isna().any()
            or values.loc[valid, list(numeric_columns)].isna().any().any()
            or (values.loc[valid, "bid"] >= values.loc[valid, "ask"]).any()
            or (values.loc[valid, ["bid_size", "ask_size"]] < 0).any().any()
        ):
            raise ValueError("MBO minute store contains an invalid marked-valid book")
        if (
            valid
            & (values["book_observed_at"] > values["decision_time"])
        ).any():
            raise ValueError("MBO minute store contains a future book snapshot")
        self.frame = values.set_index("decision_time").sort_index()
        self.provider = TopOfBookExecutionProvider(
            tick_size=tick_size,
            commission_per_side=commission_per_side,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        allow_sealed_holdout: bool = False,
        **kwargs: Any,
    ) -> "MinuteExecutionRealityStore":
        from shares.core.io import require_materialized

        source = assert_mbo_source_allowed(
            path,
            allow_sealed_holdout=allow_sealed_holdout,
        )
        require_materialized(source)
        return cls(pd.read_parquet(source), **kwargs)

    def for_bar(
        self,
        bar: Bar,
        *,
        deadline: pd.Timestamp,
        quantity: int = 1,
    ) -> ExecutionRealityInput:
        decision_clock = bar.end
        if decision_clock not in self.frame.index:
            return ExecutionRealityInput(
                deadline=deadline,
                quantity=quantity,
                source="mbo_minute_missing",
                data_age_seconds=61.0,
            )
        row = self.frame.loc[decision_clock]
        if isinstance(row, pd.DataFrame):
            raise ValueError("MBO minute store clock is not unique")
        if (
            not bool(row["book_valid"])
            or int(row["instrument_id"]) != int(bar.instrument_id)
        ):
            return ExecutionRealityInput(
                deadline=deadline,
                quantity=quantity,
                source="mbo_book_invalid",
                data_age_seconds=61.0,
            )
        book = TopOfBook(
            observed_at=row["book_observed_at"],
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            bid_size=float(row["bid_size"]),
            ask_size=float(row["ask_size"]),
        )
        observed = self.provider.observe(
            book,
            decision_clock=decision_clock,
            deadline=deadline,
            quantity=quantity,
        )
        return ExecutionRealityInput(
            **{
                **observed.__dict__,
                "source": "mbo_reconstructed",
                "depth_imbalance": (
                    float(row["depth_imbalance"])
                    if "depth_imbalance" in row and pd.notna(row["depth_imbalance"])
                    else observed.depth_imbalance
                ),
            }
        )


__all__ = [
    "F_LAST",
    "F_PUBLISHER_SPECIFIC",
    "F_SNAPSHOT",
    "MBOBookSnapshot",
    "MBOOrderBook",
    "MBORecord",
    "MBOReplayError",
    "MinuteExecutionRealityStore",
    "assert_mbo_source_allowed",
    "iter_complete_events",
    "iter_dbn_mbo_records",
    "iter_parquet_mbo_records",
]
