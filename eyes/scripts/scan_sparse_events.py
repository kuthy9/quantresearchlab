#!/usr/bin/env python3
"""Record the Eye's sparse structural events over a window — the Eye alone.

The registered Eye (``shares/core/eye_factory.build_eye``) is driven over
every completed 1m bar of the window; no Brain, LLM, Risk or execution runs.
Two tables are written per calendar month (New York) under ``--out``:

- ``events_YYYY-MM.parquet``: one row per event of ``RECORD_KINDS`` on the
  5m / 15m / 1H / 4H scales, plus the 1m sweeps / acceptances of a level
  whose own scale is 15m / 1H / 4H;
- ``bars_YYYY-MM.parquet``: one row per completed 5m / 15m / 1H / 4H bar.

Every row carries the snapshot context of its bar (``context_columns``).  The
recording is outcome-blind: no forward price is read here.

The pass is recorded as it goes: every ``--checkpoint-bars`` bars (and at each
month change) the buffered rows are written as parquet parts
(``events_YYYY-MM_NNNNN.parquet`` / ``bars_YYYY-MM_NNNNN.parquet``), the Eye
(reader and observer) is pickled to ``<state-dir>/eye.pkl`` and
``manifest.json`` lists the committed parts.  A pass that dies -- an Eye
error, a killed process -- records the failure in the manifest; ``--resume``
restores the last checkpoint and continues, rewriting any part the dead pass
wrote after it.  The Eye's audit journal lives in ``<state-dir>/journal`` so a
checkpoint can reach its spilled events.  The study that reads these tables is
``eyes/scripts/event_edge_study.py``; the pre-registration is
``eyes/docs/plans/2026-09-22-event-edge-2023.md``.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import traceback
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract.market import Timeframe  # noqa: E402

MARKET_TIMEZONE = "America/New_York"
DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
DEFAULT_MODEL = "configs/model.json"

RECORD_KINDS = frozenset({
    "mss_core_confirmed", "qualified_bos", "displacement_observed", "sweep_confirmed",
    "acceptance_confirmed", "structure_direction_confirmed", "level_reached",
})
RECORD_TIMEFRAMES = frozenset({"5m", "15m", "1H", "4H"})
HTF_LEVEL_TIMEFRAMES = frozenset({"15m", "1H", "4H"})
ONE_MINUTE_LEVEL_KINDS = frozenset({"sweep_confirmed", "acceptance_confirmed"})
CONTEXT_TIMEFRAMES = (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5)
_EVIDENCE_TEXT = (
    "session_name", "session_phase", "scope", "level_id", "source_kind", "resolution", "bos_id",
    "displacement_id", "terminal_reason", "structure_id", "range_id",
)
_EVIDENCE_FLAGS = ("displacement_context_present", "legacy_mss_qualified_context")


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def note_level_timeframes(events: Iterable[Any], levels: dict[str, str]) -> None:
    """Remember the scale the Eye published for every level id it names."""

    for event in events:
        evidence = event.evidence
        level_id = evidence.get("level_id")
        source = evidence.get("source_timeframe")
        if level_id and source:
            levels[str(level_id)] = str(source)


def event_row(event: Any, levels: Mapping[str, str]) -> dict[str, Any] | None:
    """The sparse row for one recorded event, or ``None``."""

    kind = _value(event.kind)
    tf = _value(event.timeframe)
    if kind not in RECORD_KINDS:
        return None
    evidence = event.evidence
    level_id = evidence.get("level_id")
    level_tf = None if level_id is None else levels.get(str(level_id))
    if tf == "1m":
        if kind not in ONE_MINUTE_LEVEL_KINDS or level_tf not in HTF_LEVEL_TIMEFRAMES:
            return None
    elif tf not in RECORD_TIMEFRAMES:
        return None
    row: dict[str, Any] = {
        "known_at": event.known_at,
        "event_time": event.event_time,
        "kind": kind,
        "tf": tf,
        "direction": None if event.direction is None else _value(event.direction),
        "side": event.side,
        "price": event.price,
        "lifecycle": evidence.get("lifecycle", event.lifecycle),
        "level_tf": level_tf,
        "prior_sweep": evidence.get("prior_sweep") is not None if "prior_sweep" in evidence else None,
    }
    for key in _EVIDENCE_TEXT:
        value = evidence.get(key)
        row[key] = None if value is None else str(value)
    for key in _EVIDENCE_FLAGS:
        value = evidence.get(key)
        row[key] = None if value is None else bool(value)
    return row


def bar_row(event: Any) -> dict[str, Any] | None:
    """The row for one completed bar of a recorded scale, or ``None``."""

    if _value(event.kind) != "bar_completed" or _value(event.timeframe) not in RECORD_TIMEFRAMES:
        return None
    evidence = event.evidence
    close = evidence.get("close", event.price)
    return {
        "known_at": event.known_at,
        "tf": _value(event.timeframe),
        "close": None if close is None else float(close),
        "real_completed": evidence.get("real_completed"),
    }


def context_columns(snapshot: Any) -> dict[str, Any]:
    """The snapshot facts every row carries, flattened per scale."""

    states = snapshot.timeframe_states
    context: dict[str, Any] = {"close_1m": float(snapshot.price)}
    for timeframe in CONTEXT_TIMEFRAMES:
        name = timeframe.value
        state = states.get(timeframe)
        if state is None:
            for column in ("ext", "int", "phase", "leg", "last_leg", "disp_dir", "disp_age_min", "protected_intact",
                           "range_loc", "range_label", "range_active", "atr"):
                context[f"{column}_{name}"] = None
            continue
        structure, delivery, window = state.structure, state.delivery, state.range
        displacement_at = getattr(delivery, "displacement_at", None)
        lifecycle = getattr(window, "lifecycle", None)
        context.update({
            f"ext_{name}": None if structure.external_direction is None else _value(structure.external_direction),
            f"int_{name}": None if structure.internal_direction is None else _value(structure.internal_direction),
            f"phase_{name}": None if delivery.phase is None else _value(delivery.phase),
            f"leg_{name}": None if delivery.active_leg_direction is None else _value(delivery.active_leg_direction),
            f"last_leg_{name}": None if delivery.last_leg_direction is None else _value(delivery.last_leg_direction),
            f"disp_dir_{name}": None if delivery.displacement_direction is None else _value(delivery.displacement_direction),
            f"disp_age_min_{name}": None if displacement_at is None
            else float((snapshot.asof - displacement_at) / pd.Timedelta(minutes=1)),
            f"protected_intact_{name}": structure.protected_swing_intact,
            f"range_loc_{name}": window.normalized_location,
            f"range_label_{name}": window.location_label,
            f"range_active_{name}": lifecycle is not None and lifecycle not in ("invalidated", "broken"),
            f"atr_{name}": state.quality.atr,
        })
    one_minute = states.get(Timeframe.M1)
    context["atr_1m"] = None if one_minute is None else one_minute.quality.atr
    session = snapshot.session
    context.update({
        "session_name": getattr(session, "name", None),
        "session_phase": getattr(session, "phase", None),
        "session_elapsed_min": getattr(session, "elapsed_minutes", None),
    })
    return context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> tuple[str | None, bool]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return rev, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False


@dataclass
class ScanProgress:
    """What a checkpoint must carry besides the Eye itself."""

    seen: int = 0
    last_bar_start: pd.Timestamp | None = None
    seq: int = 0
    parts: list[dict[str, Any]] = field(default_factory=list)
    levels: dict[str, str] = field(default_factory=dict)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_part(out: Path, table: str, month: str, seq: int, rows: list[dict]) -> dict[str, Any]:
    name = f"{table}_{month}_{seq:05d}.parquet"
    tmp = out / (name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, out / name)
    return {"seq": seq, "month": month, "table": table, "file": name, "rows": len(rows)}


def _months(parts: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    months: dict[str, dict[str, int]] = {}
    for part in parts:
        month = months.setdefault(part["month"], {"events": 0, "bars": 0})
        month[part["table"]] += int(part["rows"])
    return dict(sorted(months.items()))


def restorable(observer: Any) -> bool:
    """Whether a pickle of the Eye would restore with its audit journal.

    The audit store binds its journal file on its first spill; a pickle taken
    before it would restore a store that can never spill again.
    """

    store = getattr(observer, "audit_store", None)
    return store is not None and store.cold_count > 0


def checkpoint(out: Path, state_path: Path, reader: Any, observer: Any, progress: ScanProgress,
               pending: dict[str, dict[str, list[dict]]], manifest: dict[str, Any], *,
               pickle_eye: bool = True) -> None:
    """Write the pending rows as parts, pickle the Eye, then commit the manifest."""

    for month in sorted(pending):
        for table in ("events", "bars"):
            rows = pending[month][table]
            if rows:
                progress.parts.append(_write_part(out, table, month, progress.seq, rows))
        progress.seq += 1
    pending.clear()
    if pickle_eye:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(state_path.name + ".tmp")
        with tmp.open("wb") as handle:
            # Progress goes as a plain mapping so the pickle names no class of
            # this script, which runs as ``__main__``.
            pickle.dump((reader, observer, asdict(progress)), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, state_path)
    manifest["parts"] = list(progress.parts)
    manifest["months"] = _months(progress.parts)
    manifest["progress"] = {
        "bars_seen": progress.seen,
        "last_bar_start": None if progress.last_bar_start is None else str(progress.last_bar_start),
        "checkpoint": str(state_path) if pickle_eye else None,
        "checkpoint_bytes": state_path.stat().st_size if pickle_eye else None,
        "checkpointed_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _write_json(out / "manifest.json", manifest)


def load_checkpoint(state_path: Path) -> tuple[Any, Any, ScanProgress]:
    """The reader, observer and progress of the last committed checkpoint."""

    with Path(state_path).open("rb") as handle:
        reader, observer, progress = pickle.load(handle)
    if type(progress) is not dict:
        raise ValueError("the checkpoint does not carry scan progress")
    return reader, observer, ScanProgress(**progress)


def read_parts(out: Path, table: str) -> pd.DataFrame:
    """Concatenate the committed parts of one table, in commit order."""

    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    files = [part["file"] for part in manifest.get("parts", ()) if part["table"] == table]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(Path(out) / name) for name in files], ignore_index=True)


def run_scan(bars: Iterable[Any], *, reader: Any, observer: Any, progress: ScanProgress, out: Path,
             state_path: Path, manifest: dict[str, Any], emit_from: pd.Timestamp, checkpoint_bars: int,
             progress_bars: int = 0) -> ScanProgress:
    """Drive the Eye over ``bars`` from ``progress`` on, checkpointing as it goes.

    Bars at or before ``progress.last_bar_start`` were consumed by the
    checkpoint and are skipped.  A failure is recorded in the manifest (the
    last checkpoint stays the committed state) and re-raised.  No checkpoint
    is taken before the Eye is ``restorable``; its rows stay pending until one is.
    """

    if checkpoint_bars <= 0:
        raise ValueError("checkpoint_bars must be positive")
    out.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("failures", [])
    pending: dict[str, dict[str, list[dict]]] = {}
    month: str | None = None
    started, started_seen = time.monotonic(), progress.seen
    try:
        for bar in bars:
            if progress.last_bar_start is not None and bar.start <= progress.last_bar_start:
                continue
            observation = observer.observe(reader.on_bar(bar))
            progress.seen += 1
            progress.last_bar_start = bar.start
            update = observation.events_this_update
            note_level_timeframes(update, progress.levels)
            snapshot = observation.market_snapshot
            month_changed = False
            if snapshot is not None:
                asof = observation.asof.tz_convert(MARKET_TIMEZONE)
                if asof >= emit_from:
                    this_month = asof.strftime("%Y-%m")
                    month_changed = month is not None and this_month != month
                    month = this_month
                    rows = [row for row in (event_row(event, progress.levels) for event in update) if row is not None]
                    closes = [row for row in (bar_row(event) for event in update) if row is not None]
                    if rows or closes:
                        context = context_columns(snapshot)
                        buffer = pending.setdefault(this_month, {"events": [], "bars": []})
                        buffer["events"].extend({**row, **context} for row in rows)
                        buffer["bars"].extend({**row, **context} for row in closes)
            if (progress.seen % checkpoint_bars == 0 or month_changed) and restorable(observer):
                checkpoint(out, state_path, reader, observer, progress, pending, manifest)
            if progress_bars and progress.seen % progress_bars == 0:
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
                elapsed = time.monotonic() - started
                rate = (progress.seen - started_seen) / elapsed if elapsed > 0 else float("nan")
                print(f"{progress.seen} bars  {observation.asof}  {elapsed / 60:.1f} min  {rate:.1f} bars/s  "
                      f"max rss {rss_mb:.0f} MB  levels {len(progress.levels)}", file=sys.stderr, flush=True)
    except BaseException as exc:
        manifest["failures"].append({
            "bars_seen": progress.seen,
            "last_bar_start": None if progress.last_bar_start is None else str(progress.last_bar_start),
            "committed_bars_seen": manifest.get("progress", {}).get("bars_seen", 0),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
            "failed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        _write_json(out / "manifest.json", manifest)
        raise
    checkpoint(out, state_path, reader, observer, progress, pending, manifest, pickle_eye=restorable(observer))
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warmup-start", required=True, help="New York time the Eye starts from")
    parser.add_argument("--emit-start", required=True, help="New York time rows start being recorded")
    parser.add_argument("--end", required=True, help="New York time the tape is loaded up to")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="checkpoint and audit-journal directory (default: <out>/_state)")
    parser.add_argument("--checkpoint-bars", type=int, default=5000)
    parser.add_argument("--progress-bars", type=int, default=20000)
    parser.add_argument("--resume", action="store_true", help="continue from the last committed checkpoint")
    args = parser.parse_args(argv)

    from shares.core.eye_factory import build_eye
    from shares.core.io import iter_completed_bars, load_ohlcv
    from shares.core.scale_registry import parse_scale_specs, scale_registry_id
    from eyes.core.semantics import load_semantic_selection

    out = args.out if args.out.is_absolute() else ROOT / args.out
    state_dir = out / "_state" if args.state_dir is None else (
        args.state_dir if args.state_dir.is_absolute() else ROOT / args.state_dir)
    state_path = state_dir / "eye.pkl"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.source
    model_path = ROOT / args.model
    model = json.loads(model_path.read_text(encoding="utf-8"))
    selection = load_semantic_selection(model["semantic_selection"], root=ROOT)
    rev, dirty = _git()
    window = {"source": args.source, "warmup_start": args.warmup_start, "emit_start": args.emit_start, "end": args.end}
    identity = {
        "window": window,
        "source_sha256": _sha256(source),
        "model": args.model,
        "model_sha256": _sha256(model_path),
        "atomic_definition_identity": selection.atomic_definition_identity,
        "scale_registry_id": scale_registry_id(parse_scale_specs(model["scales"])),
        "record_kinds": sorted(RECORD_KINDS),
        "record_timeframes": sorted(RECORD_TIMEFRAMES),
        "checkpoint_bars": args.checkpoint_bars,
    }
    segment = {
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "git_revision": rev,
        "git_dirty": dirty,
    }
    manifest_path = out / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if args.resume:
        if existing is None or "progress" not in existing or not state_path.exists():
            raise SystemExit("--resume needs a committed checkpoint in the output directory")
        drift = {key: (existing.get(key), value) for key, value in identity.items() if existing.get(key) != value}
        if drift:
            raise SystemExit(f"--resume refused: the pass identity changed: {sorted(drift)}")
        manifest = existing
        reader, observer, progress = load_checkpoint(state_path)
        segment["resumed_from_bars_seen"] = progress.seen
    else:
        if existing is not None and existing.get("progress"):
            raise SystemExit(f"{manifest_path} already holds committed progress; pass --resume or use a new --out")
        manifest = {**identity, "segments": [], "failures": [] if existing is None else existing.get("failures", [])}
        state_dir.mkdir(parents=True, exist_ok=True)
        reader, observer = build_eye(model_path, root=ROOT, audit_journal_dir=state_dir / "journal")
        progress = ScanProgress()
        segment["resumed_from_bars_seen"] = 0
    manifest.setdefault("segments", []).append(segment)
    manifest.pop("finished_at", None)
    _write_json(manifest_path, manifest)

    loaded = load_ohlcv(source, start=args.warmup_start, end=args.end)
    emit_from = pd.Timestamp(args.emit_start, tz=MARKET_TIMEZONE)
    started = time.monotonic()
    run_scan(iter_completed_bars(loaded.frame), reader=reader, observer=observer, progress=progress, out=out,
             state_path=state_path, manifest=manifest, emit_from=emit_from, checkpoint_bars=args.checkpoint_bars,
             progress_bars=args.progress_bars)
    segment.update({
        "bars_seen": progress.seen,
        "minutes": round((time.monotonic() - started) / 60, 1),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)),
        "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    manifest.update({"bars_seen": progress.seen, "finished_at": segment["finished_at"]})
    _write_json(manifest_path, manifest)
    print(json.dumps({"bars_seen": progress.seen, "months": manifest["months"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
