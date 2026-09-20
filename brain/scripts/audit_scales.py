"""Audit the per-scale facts the Brain reads, without an LLM.

    .venv/bin/python -m brain.scripts.audit_scales \
        --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00

Drives the Eye over the window exactly as ``run_llm_brain.py`` does and
prints, per scale, one line per *change point* of the discrete facts
(external / internal direction, last BOS and MSS, protection, the reset,
the active and last leg, the phase, the displacement's direction) beside
the close and the continuous facts (the forming leg's excursion in that
scale's ATRs, the displacement's age in bars).  It is how the Eye layer of
the direction fix is judged before any LLM run (spec
``brain/docs/specs/2026-09-18-direction-eye-brain-execution-design.md``
§1.5).

    .venv/bin/python -m brain.scripts.audit_scales --triggers outputs/brain_journal/<run>

replays a run's LLM-call triggers under the controller config as it is now
(bookkeeping kinds, the relation debounce) and prints the calls it would
keep and their sharp-move coverage — the deterministic check of a
wake/sleep change (spec §2.3)."""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from brain.core.eye_view import delivery_payload, reset_payload
from brain.core.journal import JournalReader
from brain.core.sleep_controller import ControllerConfig
from brain.scripts._run_identity import DEFAULT_MODEL, DEFAULT_SOURCE, MARKET_TIMEZONE, ROOT, RunWindow, drive
from brain.scripts.summarize_run import load_bars_for, sharp_move_coverage
from contract.market import Timeframe

DEFAULT_CONTROLLER = ROOT / "brain" / "configs" / "sleep_controller.json"

SCALES: tuple[str, ...] = ("4H", "1H", "15m", "5m")
# Facts that move with every bar; a change point is read on the others.
CONTINUOUS: frozenset[str] = frozenset({"forming_leg_atr", "displacement_age_bars"})


def _direction(value: Any) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def scale_facts(snapshot: Any, timeframe: Timeframe, *, known_at: pd.Timestamp) -> dict[str, Any]:
    """The facts of one scale as the Brain's view publishes them."""
    state = snapshot.timeframe_states[timeframe]
    structure = state.structure
    delivery = delivery_payload(state, known_at)
    reset = reset_payload(state, known_at)
    return {
        "external": _direction(structure.external_direction),
        "internal": _direction(structure.internal_direction),
        "last_bos": _direction(structure.last_bos_direction),
        "last_mss": _direction(structure.last_mss_direction),
        "protected_intact": structure.protected_swing_intact,
        "reset": None if reset is None else reset["direction"],
        "active_leg": delivery["active_leg_direction"],
        "last_leg": delivery["last_leg_direction"],
        "phase": delivery["phase"],
        "displacement": delivery["displacement_direction"],
        "forming_leg_atr": delivery["forming_leg_atr"],
        "displacement_age_bars": delivery["displacement_age_bars"],
    }


def change_points(
    rows: Iterable[tuple[Any, float, Mapping[str, Any]]], *, ignore: frozenset[str] = CONTINUOUS
) -> list[tuple[Any, float, Mapping[str, Any]]]:
    """The rows whose discrete facts differ from the previous row's."""
    kept: list[tuple[Any, float, Mapping[str, Any]]] = []
    previous: dict[str, Any] | None = None
    for row in rows:
        discrete = {key: value for key, value in row[2].items() if key not in ignore}
        if discrete != previous:
            kept.append(row)
            previous = discrete
    return kept


def replay_triggers(
    calls: Iterable[tuple[pd.Timestamp, str, Sequence[str], Mapping[str, str]]], *, config: ControllerConfig
) -> list[pd.Timestamp]:
    """The call times a run would keep under ``config``: every WAKE, and an
    UPDATE with a reason that is a reaction (an evidence id whose kind is not
    bookkeeping) or a watched alias whose last kept trigger is at least
    ``relation_change_debounce_bars`` minutes old.  ``calls`` rows are
    ``(known_at, trigger kind, reasons, evidence kind by id)``."""
    window = pd.Timedelta(minutes=config.relation_change_debounce_bars)
    last: dict[str, pd.Timestamp] = {}
    kept: list[pd.Timestamp] = []
    for known_at, kind, reasons, kinds_by_id in calls:
        when = pd.Timestamp(known_at)
        if kind == "WAKE":
            kept.append(when)
            continue
        reaction = any(
            reason.startswith("ev_") and kinds_by_id.get(reason, "") not in config.bookkeeping_kinds
            for reason in reasons
        )
        aliases = [
            reason for reason in reasons
            if not reason.startswith("ev_") and (reason not in last or when - last[reason] >= window)
        ]
        if reaction or aliases:
            kept.append(when)
            for alias in aliases:
                last[alias] = when
    return kept


def _replay_run(run_dir: Path, config: ControllerConfig) -> int:
    reader = JournalReader(run_dir)
    run = reader.run()
    calls = []
    for episode_id in reader.episode_ids():
        for record in reader.records(episode_id):
            if record.record != "llm_call":
                continue
            llm_input = record.payload.get("input") or {}
            trigger = llm_input.get("trigger") or {}
            kinds = {item["evidence_id"]: str(item["kind"]) for item in llm_input.get("new_evidence", ())}
            calls.append((record.known_at, str(trigger.get("kind")), [str(r) for r in trigger.get("reasons", ())], kinds))
    kept = replay_triggers(calls, config=config)
    print(f"run {run.get('run_id')}: {len(calls)} calls journaled, {len(kept)} kept under the controller as configured "
          f"(debounce {config.relation_change_debounce_bars} bars, {len(config.bookkeeping_kinds)} bookkeeping kinds)")
    bars = load_bars_for(run)
    if bars:
        before = sharp_move_coverage(bars, call_times=[row[0] for row in calls])
        after = sharp_move_coverage(bars, call_times=kept)
        print(f"sharp-move coverage as run: {before['covered']}/{before['sharp_moves']} (RTH {before['rth_covered']}/{before['rth_sharp_moves']}); "
              f"kept: {after['covered']}/{after['sharp_moves']} (RTH {after['rth_covered']}/{after['rth_sharp_moves']})")
    return 0


def _line(when: pd.Timestamp, close: float, facts: Mapping[str, Any]) -> str:
    stamp = pd.Timestamp(when).tz_convert(MARKET_TIMEZONE).strftime("%m-%d %H:%M")
    forming = facts["forming_leg_atr"]
    age = facts["displacement_age_bars"]
    return (
        f"{stamp} {close:9.2f}  ext={facts['external'] or '-':<5} int={facts['internal'] or '-':<5} "
        f"bos={facts['last_bos'] or '-':<5} mss={facts['last_mss'] or '-':<5} intact={str(facts['protected_intact']):<5} "
        f"reset={facts['reset'] or '-':<5} active={facts['active_leg'] or '-':<5} last={facts['last_leg'] or '-':<5} "
        f"phase={facts['phase']:<16} forming={'-' if forming is None else f'{forming:+.2f}':>6} "
        f"disp={facts['displacement'] or '-'}/{'-' if age is None else age}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="the Eye's configs/model.json")
    parser.add_argument("--warmup-start")
    parser.add_argument("--emit-start")
    parser.add_argument("--end")
    parser.add_argument("--triggers", metavar="RUN_DIR", help="replay this run's call triggers under the controller config instead of auditing the Eye")
    parser.add_argument("--controller", default=str(DEFAULT_CONTROLLER))
    parser.add_argument("--scales", default=",".join(SCALES), help="comma-separated, default 4H,1H,15m,5m")
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.triggers:
        return _replay_run(Path(args.triggers), ControllerConfig.from_json(Path(args.controller)))
    if not (args.warmup_start and args.emit_start and args.end):
        raise SystemExit("--warmup-start, --emit-start and --end are required to audit the Eye")
    scales = tuple(Timeframe(value) for value in args.scales.split(","))
    window = RunWindow(Path(args.source), args.warmup_start, args.emit_start, args.end)
    rows: dict[Timeframe, list[tuple[pd.Timestamp, float, dict[str, Any]]]] = {scale: [] for scale in scales}

    def on_observation(observation: Any, emitting: bool, bar: Any) -> None:
        if not emitting:
            return
        snapshot = observation.market_snapshot
        known_at = pd.Timestamp(snapshot.asof).tz_convert("UTC")
        for scale in scales:
            if scale in snapshot.timeframe_states:
                rows[scale].append((known_at, float(bar.close), scale_facts(snapshot, scale, known_at=known_at)))

    seen = drive(
        window, model_path=Path(args.model_path), root=ROOT, on_observation=on_observation,
        progress_every=args.progress_every, log=lambda text: print(text, file=sys.stderr),
    )
    print(f"# {seen} bars seen; emit window {args.emit_start} -> {args.end} ({MARKET_TIMEZONE})")
    for scale in scales:
        points = change_points(rows[scale])
        print(f"\n== {scale.value}: {len(points)} change points over {len(rows[scale])} bars")
        for when, close, facts in points:
            print(_line(when, close, facts))
    return 0


__all__ = ["CONTINUOUS", "SCALES", "change_points", "main", "replay_triggers", "scale_facts"]


if __name__ == "__main__":
    raise SystemExit(main())
