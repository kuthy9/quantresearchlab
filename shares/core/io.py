"""Fail-closed NQ data inspection, loading, and minute densification."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterator

import pandas as pd

from .market_clock import (
    MARKET_TIMEZONE,
    is_registered_trading_minute,
    scheduled_gap_kind,
)
from contract.market import Bar


UF_DATALESS = 0x40000000
NQ_OUTRIGHT = re.compile(r"^NQ[HMUZ]\d{1,2}$")


class DataMaterializationError(RuntimeError):
    pass


class DataContinuityError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceStatus:
    path: Path
    exists: bool
    materialized: bool
    dataless: bool
    logical_bytes: int
    allocated_blocks: int


@dataclass(frozen=True)
class LoadedOHLCV:
    frame: pd.DataFrame
    source: Path
    source_role: str
    contract_selection_causal: bool
    warnings: tuple[str, ...]


def inspect_source(path: str | Path) -> SourceStatus:
    source = Path(path)
    if not source.exists():
        return SourceStatus(source, False, False, False, 0, 0)
    stat = os.stat(source, follow_symlinks=False)
    flags = int(getattr(stat, "st_flags", 0))
    dataless = bool(flags & UF_DATALESS)
    return SourceStatus(
        path=source,
        exists=True,
        materialized=bool(source.is_file() and not dataless),
        dataless=dataless,
        logical_bytes=int(stat.st_size),
        allocated_blocks=int(getattr(stat, "st_blocks", 0)),
    )


def require_materialized(path: str | Path) -> SourceStatus:
    status = inspect_source(path)
    if not status.materialized:
        state = "dataless placeholder" if status.dataless else "missing/non-file"
        raise DataMaterializationError(f"{status.path} is {state}; refusing to read")
    return status


def audit_nq_catalog(root: str | Path = "data") -> tuple[SourceStatus, ...]:
    base = Path(root)
    known = (
        base / "raw/nq_ohlcv_1m/glbx-mdp3-20170101-20211231.ohlcv-1m.dbn.zst",
        base / "raw/nq_ohlcv_1m/glbx-mdp3-20220101-20251231.ohlcv-1m.csv",
        base / "raw/nq_ohlcv_1m/glbx-mdp3-20260101-20260714.ohlcv-1m.dbn.zst",
        base / "processed/nq_1m_databento_front_through_20260713T1959ET.parquet",
        base / "raw/nq_mbo/legacy_parquet/manifest.json",
        base / "raw/nq_mbo/holdout_dbn/glbx-mdp3-20240801-20241201.mbo.dbn.zst",
    )
    return tuple(inspect_source(path) for path in known)


def _read_native_dbn(path: Path) -> pd.DataFrame:
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "native DBN loading requires the optional 'databento' dependency"
        ) from exc
    frame = db.DBNStore.from_file(path).to_df(map_symbols=True)
    if "ts_event" not in frame.columns:
        if frame.index.name != "ts_event":
            raise ValueError("DBN frame has no ts_event column or index")
        frame = frame.reset_index()
    return frame


def load_raw_multicontract(path: str | Path) -> pd.DataFrame:
    """Load Databento rows without pretending they are a continuous contract."""

    source = Path(path)
    require_materialized(source)
    name = source.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.zst") or name.endswith(".csv.gz"):
        raw = pd.read_csv(source)
    elif name.endswith(".dbn") or name.endswith(".dbn.zst"):
        raw = _read_native_dbn(source)
    else:
        raise ValueError(f"unsupported raw Databento source: {source.name}")
    required = {"ts_event", "open", "high", "low", "close", "volume", "symbol", "instrument_id"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw Databento fields missing: {missing}")
    work = raw.loc[:, sorted(required)].copy()
    work["ts"] = pd.to_datetime(work["ts_event"], errors="coerce", utc=True).dt.tz_convert(
        MARKET_TIMEZONE
    )
    work = work.loc[
        work["ts"].notna()
        & work["symbol"].fillna("").astype(str).map(lambda value: bool(NQ_OUTRIGHT.match(value)))
    ].copy()
    for column in ("open", "high", "low", "close", "volume", "instrument_id"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(
        subset=["ts", "open", "high", "low", "close", "volume", "instrument_id"]
    )
    work["instrument_id"] = work["instrument_id"].astype("int64")
    return work[
        ["ts", "open", "high", "low", "close", "volume", "symbol", "instrument_id"]
    ].sort_values("ts", kind="stable")


def build_previous_session_front(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each Globex session using only the prior completed session's volume.

    The first source session has no causal predecessor and is omitted. No
    fallback to current-session volume is allowed.
    """

    work = pd.DataFrame(raw).copy()
    required = {"ts", "open", "high", "low", "close", "volume", "symbol", "instrument_id"}
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(f"causal front input fields missing: {missing}")
    timestamps = pd.to_datetime(work["ts"], errors="coerce", utc=True).dt.tz_convert(
        MARKET_TIMEZONE
    )
    if timestamps.isna().any():
        raise ValueError("causal front input has invalid timestamps")
    work["ts"] = timestamps
    # Session volume is an exchange-clock statistic. Off-session vendor
    # records must not influence the prior-session winner or enter replay.
    work = work.loc[work["ts"].map(is_registered_trading_minute)].copy()
    if work.empty:
        raise ValueError("causal front input has no registered market minutes")
    timestamps = work["ts"]
    local_day = timestamps.dt.tz_localize(None).dt.normalize()
    after_open = (timestamps.dt.hour >= 18)
    work["session_date"] = (
        local_day + pd.to_timedelta(after_open.astype(int), unit="D")
    ).dt.date.astype(str)
    session_volume = (
        work.groupby(["session_date", "symbol", "instrument_id"], as_index=False)[
            "volume"
        ]
        .sum()
        .rename(columns={"volume": "prior_session_volume"})
    )
    winner = (
        session_volume.sort_values(
            ["session_date", "prior_session_volume", "symbol"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("session_date", keep="first")
        .sort_values("session_date", kind="stable")
        .reset_index(drop=True)
    )
    session_end = (
        work.groupby("session_date", as_index=False)["ts"]
        .max()
        .rename(columns={"ts": "prior_information_cutoff"})
    )
    winner = winner.merge(session_end, on="session_date", how="left", validate="one_to_one")
    selection = pd.DataFrame(
        {
            "session_date": winner["session_date"].iloc[1:].to_numpy(),
            "symbol": winner["symbol"].shift(1).iloc[1:].to_numpy(),
            "instrument_id": winner["instrument_id"].shift(1).iloc[1:].to_numpy(),
            "selection_source_session": winner["session_date"].shift(1).iloc[1:].to_numpy(),
            "prior_session_volume": winner["prior_session_volume"].shift(1).iloc[1:].to_numpy(),
            "selection_known_at": winner["prior_information_cutoff"].shift(1).iloc[1:].to_numpy(),
        }
    )
    selected = work.merge(
        selection[["session_date", "symbol", "instrument_id"]],
        on=["session_date", "symbol", "instrument_id"],
        how="inner",
        validate="many_to_one",
    )
    selected = selected.sort_values("ts", kind="stable")
    selected_sessions = set(selected["session_date"].astype(str))
    missing_sessions = sorted(set(selection["session_date"].astype(str)) - selected_sessions)
    if missing_sessions:
        raise DataContinuityError(
            "prior-session selected contract is absent in current raw session(s): "
            + ", ".join(missing_sessions[:10])
        )
    if selected["ts"].duplicated().any():
        raise ValueError("causal front selection produced duplicate minute timestamps")
    frame = _normalize_frame(selected.drop(columns=["session_date"]).set_index("ts"))
    if frame.empty:
        raise ValueError("causal previous-session front selection is empty")
    return frame, selection


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(frame).copy()
    if "ts" in values.columns:
        timestamp = pd.to_datetime(values.pop("ts"), errors="coerce", utc=True)
    elif "ts_event" in values.columns:
        timestamp = pd.to_datetime(values.pop("ts_event"), errors="coerce", utc=True)
    elif isinstance(values.index, pd.DatetimeIndex):
        timestamp = pd.to_datetime(values.index, errors="coerce", utc=True)
    else:
        raise ValueError("OHLCV source has no timestamp")
    values.index = pd.DatetimeIndex(timestamp).tz_convert(MARKET_TIMEZONE)
    values.index.name = "ts"
    values = values.loc[~values.index.isna()].copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(values.columns))
    if missing:
        raise ValueError(f"OHLCV source fields missing: {missing}")
    for column in required:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    if "symbol" not in values:
        values["symbol"] = "NQ"
    if "instrument_id" not in values:
        values["instrument_id"] = 0
    values["symbol"] = values["symbol"].astype(str)
    values["instrument_id"] = pd.to_numeric(
        values["instrument_id"], errors="raise"
    ).astype("int64")
    values = values.dropna(subset=[*required]).sort_index(kind="stable")
    if values.index.has_duplicates:
        raise ValueError("OHLCV source has duplicate timestamps")
    invalid = (
        (values["high"] < values[["open", "close"]].max(axis=1))
        | (values["low"] > values[["open", "close"]].min(axis=1))
        | (values["high"] < values["low"])
        | (values["volume"] < 0)
    )
    if invalid.any():
        raise ValueError(f"OHLCV source has {int(invalid.sum())} invalid bars")
    return values[
        ["open", "high", "low", "close", "volume", "symbol", "instrument_id"]
    ]


def load_ohlcv(
    path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> LoadedOHLCV:
    source = Path(path)
    require_materialized(source)
    lower_name = source.name.lower()
    warnings: list[str] = []
    causal_contract_selection = True
    start_ts = None if start is None else pd.Timestamp(start)
    end_ts = None if end is None else pd.Timestamp(end)
    if start_ts is not None:
        start_ts = (
            start_ts.tz_localize(MARKET_TIMEZONE)
            if start_ts.tzinfo is None
            else start_ts.tz_convert(MARKET_TIMEZONE)
        )
    if end_ts is not None:
        end_ts = (
            end_ts.tz_localize(MARKET_TIMEZONE)
            if end_ts.tzinfo is None
            else end_ts.tz_convert(MARKET_TIMEZONE)
        )
    if (
        start_ts is not None
        and end_ts is not None
        and end_ts <= start_ts
    ):
        raise ValueError("OHLCV read interval must be positive")
    if lower_name.endswith(".parquet"):
        filters = []
        if start_ts is not None:
            filters.append(("ts", ">=", start_ts))
        if end_ts is not None:
            filters.append(("ts", "<", end_ts))
        raw = pd.read_parquet(
            source,
            filters=filters or None,
        )
        source_role = "processed_continuous_front"
        if "databento_front" in lower_name:
            causal_contract_selection = False
            warnings.append(
                "legacy continuous-front parquet selected the dominant contract "
                "with complete same-day volume; roll-sensitive use is research-only"
            )
    elif (
        lower_name.endswith(".csv")
        or lower_name.endswith(".csv.zst")
        or lower_name.endswith(".csv.gz")
        or lower_name.endswith(".dbn")
        or lower_name.endswith(".dbn.zst")
    ):
        raise ValueError(
            "raw multi-contract input cannot be replayed directly; use "
            "load_raw_multicontract + build_previous_session_front first"
        )
    else:
        raise ValueError(f"unsupported OHLCV source: {source.name}")
    frame = _normalize_frame(raw)
    if start_ts is not None:
        frame = frame.loc[frame.index >= start_ts]
    if end_ts is not None:
        frame = frame.loc[frame.index < end_ts]
    return LoadedOHLCV(
        frame=frame,
        source=source,
        source_role=source_role,
        contract_selection_causal=causal_contract_selection,
        warnings=tuple(warnings),
    )


def iter_completed_bars(
    frame: pd.DataFrame,
    *,
    maximum_no_trade_gap_minutes: int = 5,
    densify_one_missing_minute: bool | None = None,
    allow_data_gap_reset: bool = False,
) -> Iterator[Bar]:
    """Yield the registered causal clock with bounded no-trade densification.

    Databento emits no OHLCV record for a minute without a trade.  A missing
    minute is synthesized only when the independent exchange clock says the
    market was open, the contract is unchanged, and the open-minute run is no
    longer than ``maximum_no_trade_gap_minutes``.  Closed-market minutes are
    skipped; every other discontinuity fails closed.

    ``densify_one_missing_minute`` is retained only as a compatibility switch:
    true selects a one-minute cap and false disables densification.
    """

    if densify_one_missing_minute is not None:
        maximum_no_trade_gap_minutes = int(bool(densify_one_missing_minute))
    maximum_no_trade_gap_minutes = int(maximum_no_trade_gap_minutes)
    if maximum_no_trade_gap_minutes < 0:
        raise ValueError("maximum_no_trade_gap_minutes must be nonnegative")

    source = _normalize_frame(frame)
    # Some legacy materializations contain isolated vendor records inside a
    # registered exchange closure (notably the historical 16:15 settlement
    # pause).  They are not tradable minutes and must not inflate completed
    # higher-timeframe candle coverage.  Raw front materialization applies the
    # same calendar filter; this keeps older processed fronts causally aligned.
    source = source.loc[
        source.index.map(is_registered_trading_minute)
    ].copy()
    prior: Bar | None = None
    for timestamp, row in source.iterrows():
        data_gap_before_minutes = 0
        current = Bar(
            start=timestamp,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            symbol=str(row.symbol),
            instrument_id=int(row.instrument_id),
        )
        if prior is not None and current.start != prior.end:
            same_contract = (
                current.symbol == prior.symbol
                and current.instrument_id == prior.instrument_id
            )
            open_missing: list[pd.Timestamp] = []
            cursor = prior.end
            while cursor < current.start:
                if is_registered_trading_minute(cursor):
                    open_missing.append(cursor)
                cursor += pd.Timedelta(minutes=1)
            if open_missing:
                if (
                    not same_contract
                    or len(open_missing) > maximum_no_trade_gap_minutes
                ):
                    if not allow_data_gap_reset or not same_contract:
                        raise DataContinuityError(
                            "unresolved open-market gap "
                            f"{prior.end} -> {current.start}: "
                            f"{len(open_missing)} missing trading minute(s), "
                            f"cap={maximum_no_trade_gap_minutes}, "
                            f"same_contract={same_contract}"
                        )
                    data_gap_before_minutes = len(open_missing)
                    current = Bar(
                        start=current.start,
                        open=current.open,
                        high=current.high,
                        low=current.low,
                        close=current.close,
                        volume=current.volume,
                        symbol=current.symbol,
                        instrument_id=current.instrument_id,
                        data_gap_before_minutes=data_gap_before_minutes,
                    )
                else:
                    for missing_start in open_missing:
                        synthetic = Bar(
                            start=missing_start,
                            open=prior.close,
                            high=prior.close,
                            low=prior.close,
                            close=prior.close,
                            volume=0.0,
                            symbol=prior.symbol,
                            instrument_id=prior.instrument_id,
                            synthetic_no_trade=True,
                        )
                        yield synthetic
                        prior = synthetic
            if (
                prior.end != current.start
                and scheduled_gap_kind(prior.end, current.start) is None
                and data_gap_before_minutes == 0
            ):
                raise DataContinuityError(
                    f"unresolved market-minute gap {prior.end} -> {current.start}"
                )
        yield current
        prior = current


def data_tree_materialization(root: str | Path) -> dict[str, int]:
    """Summarize placeholders without opening their contents."""

    files = materialized = dataless = logical = 0
    for directory, _, names in os.walk(Path(root)):
        for name in names:
            status = inspect_source(Path(directory) / name)
            files += 1
            materialized += int(status.materialized)
            dataless += int(status.dataless)
            logical += status.logical_bytes
    return {
        "files": files,
        "materialized_files": materialized,
        "dataless_files": dataless,
        "logical_bytes": logical,
    }


__all__ = [
    "DataContinuityError",
    "DataMaterializationError",
    "LoadedOHLCV",
    "SourceStatus",
    "audit_nq_catalog",
    "build_previous_session_front",
    "data_tree_materialization",
    "inspect_source",
    "iter_completed_bars",
    "load_raw_multicontract",
    "load_ohlcv",
    "require_materialized",
]
