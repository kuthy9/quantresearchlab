from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from contract.eye import EventKind, EventOrigin, MarketEvent
from contract.market import SMC_SEMANTIC_VERSION
from contract.market import Direction, Timeframe
from eyes.scripts.scan_sparse_events import bar_row, context_columns, event_row, note_level_timeframes

TZ = "America/New_York"
T0 = pd.Timestamp("2023-03-01 10:00", tz=TZ)


def _event(kind, tf, *, direction=None, side=None, evidence=None, lifecycle=None, minutes=0):
    at = T0 + pd.Timedelta(minutes=minutes)
    return MarketEvent(
        event_id=f"{minutes}:{kind.value}:{tf.value}", kind=kind, observed_at=at, timeframe=tf, side=side, price=100.0,
        strength=0.5, details={} if evidence is None else evidence, lifecycle=lifecycle, event_time=at, known_at=at,
        semantic_version=SMC_SEMANTIC_VERSION, origin=EventOrigin.SEMANTIC_ATOMIC, direction=direction,
    )


def test_a_15m_mss_is_recorded_with_its_context_evidence() -> None:
    ev = _event(EventKind.MSS_CORE_CONFIRMED, Timeframe.M15, direction=Direction.LONG, side="above", evidence={
        "session_name": "new_york_am", "session_phase": "opening_drive", "bos_id": "b1", "scope": "opposed",
        "prior_sweep": "sw1", "displacement_context_present": True, "legacy_mss_qualified_context": False,
    })
    row = event_row(ev, {})
    assert row["kind"] == "mss_core_confirmed" and row["tf"] == "15m" and row["direction"] == "long"
    assert row["prior_sweep"] is True and row["displacement_context_present"] is True and row["scope"] == "opposed"
    assert row["known_at"] == T0 and row["session_name"] == "new_york_am" and row["bos_id"] == "b1"


def test_1m_events_are_dropped_unless_they_sweep_a_higher_scale_level() -> None:
    levels: dict[str, str] = {}
    note_level_timeframes([_event(EventKind.LEVEL_REACHED, Timeframe.M15, evidence={"level_id": "swing:a", "source_timeframe": "15m"}),
                           _event(EventKind.LEVEL_REACHED, Timeframe.M1, evidence={"level_id": "swing:b", "source_timeframe": "1m"})], levels)
    assert levels == {"swing:a": "15m", "swing:b": "1m"}
    htf = _event(EventKind.SWEEP_CONFIRMED, Timeframe.M1, direction=Direction.SHORT, side="above", evidence={"level_id": "swing:a"})
    own = _event(EventKind.SWEEP_CONFIRMED, Timeframe.M1, direction=Direction.SHORT, side="above", evidence={"level_id": "swing:b"})
    unknown = _event(EventKind.SWEEP_CONFIRMED, Timeframe.M1, direction=Direction.LONG, side="below", evidence={"level_id": "swing:z"})
    assert event_row(htf, levels)["level_tf"] == "15m"
    assert event_row(own, levels) is None and event_row(unknown, levels) is None
    assert event_row(_event(EventKind.MSS_CORE_CONFIRMED, Timeframe.M1, direction=Direction.LONG), levels) is None


def test_displacement_rows_carry_their_lifecycle_and_other_kinds_are_ignored() -> None:
    disp = _event(EventKind.DISPLACEMENT_OBSERVED, Timeframe.M5, direction=Direction.SHORT, evidence={
        "lifecycle": "active", "terminal_reason": None, "displacement_id": "d1"})
    row = event_row(disp, {})
    assert row["lifecycle"] == "active" and row["displacement_id"] == "d1" and row["tf"] == "5m"
    assert event_row(_event(EventKind.FVG_CREATED, Timeframe.M15, direction=Direction.LONG), {}) is None
    assert event_row(_event(EventKind.SWING_CONFIRMED, Timeframe.H1), {}) is None


def test_a_completed_bar_of_a_recorded_scale_is_a_bar_row() -> None:
    bar = _event(EventKind.BAR_COMPLETED, Timeframe.M15, evidence={"close": 101.0, "real_completed": True})
    assert bar_row(bar) == {"known_at": T0, "tf": "15m", "close": 101.0, "real_completed": True}
    assert bar_row(_event(EventKind.BAR_COMPLETED, Timeframe.M1, evidence={"close": 1.0})) is None
    assert bar_row(_event(EventKind.QUALIFIED_BOS, Timeframe.M15)) is None


def _tf_state(ext, phase, leg, atr, loc=None, disp_at=None):
    return SimpleNamespace(
        structure=SimpleNamespace(external_direction=ext, internal_direction=None, protected_swing_intact=True),
        delivery=SimpleNamespace(phase=SimpleNamespace(value=phase), active_leg_direction=leg, last_leg_direction=None,
                                 displacement_direction=None, displacement_at=disp_at),
        range=SimpleNamespace(normalized_location=loc, location_label=None if loc is None else "discount", lifecycle=None if loc is None else "active"),
        quality=SimpleNamespace(atr=atr),
    )


def test_context_columns_flatten_every_scale_and_the_session() -> None:
    snapshot = SimpleNamespace(
        asof=T0, price=100.0,
        timeframe_states={
            Timeframe.H4: _tf_state(Direction.LONG, "expansion", Direction.LONG, 40.0),
            Timeframe.H1: _tf_state(None, "transition", Direction.SHORT, 20.0),
            Timeframe.M15: _tf_state(Direction.SHORT, "retracement", Direction.LONG, 10.0, loc=0.3,
                                     disp_at=T0 - pd.Timedelta(minutes=45)),
            Timeframe.M5: _tf_state(Direction.SHORT, "expansion", Direction.SHORT, 5.0),
            Timeframe.M1: _tf_state(Direction.SHORT, "expansion", Direction.SHORT, 2.0),
        },
        session=SimpleNamespace(name="new_york_am", phase="opening_drive", elapsed_minutes=30),
    )
    ctx = context_columns(snapshot)
    assert ctx["ext_4H"] == "long" and ctx["ext_1H"] is None and ctx["phase_15m"] == "retracement"
    assert ctx["leg_1H"] == "short" and ctx["atr_15m"] == 10.0 and ctx["atr_1m"] == 2.0 and ctx["close_1m"] == 100.0
    assert ctx["range_loc_15m"] == 0.3 and ctx["range_active_15m"] is True and ctx["range_active_1H"] is False
    assert ctx["disp_age_min_15m"] == 45.0 and ctx["disp_age_min_5m"] is None
    assert ctx["session_name"] == "new_york_am" and ctx["session_elapsed_min"] == 30


def _eye(journal, *, hot_window_minutes=None):
    """The registered Eye; a shorter audit hot window makes it spill within a test tape."""

    import json
    from pathlib import Path

    from shares.core.eye_factory import build_eye

    root = Path(__file__).resolve().parents[2]
    model_path = root / "configs" / "model.json"
    if hot_window_minutes is not None:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["observer"]["audit_hot_window_minutes"] = hot_window_minutes
        model_path = Path(journal).parent / "model.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(model), encoding="utf-8")
    return build_eye(model_path, root=root, audit_journal_dir=journal)


def test_a_crashed_scan_resumes_from_its_last_checkpoint_and_matches_an_uninterrupted_one(tmp_path) -> None:
    import json

    import pytest

    from eyes.scripts.scan_sparse_events import ScanProgress, load_checkpoint, read_parts, run_scan
    from eyes.tests.test_facts_on_every_scale import _noisy
    from shares.tests.helpers import session_bars

    bars = _noisy(session_bars(1)[:420], seed=11)
    emit_from = bars[0].start

    def scan(out, tape, *, reader=None, observer=None, progress=None, manifest=None):
        if reader is None:
            reader, observer = _eye(out / "_state" / "journal", hot_window_minutes=60)
        return run_scan(tape, reader=reader, observer=observer, progress=progress or ScanProgress(),
                        out=out, state_path=out / "_state" / "eye.pkl", manifest={} if manifest is None else manifest,
                        emit_from=emit_from, checkpoint_bars=100)

    straight = tmp_path / "straight"
    done = scan(straight, bars)
    assert done.seen == len(bars)

    crashed = tmp_path / "crashed"

    def dying():
        yield from bars[:250]
        raise RuntimeError("the machine overheated")

    with pytest.raises(RuntimeError, match="overheated"):
        scan(crashed, dying())
    manifest = json.loads((crashed / "manifest.json").read_text())
    assert manifest["progress"]["bars_seen"] == 200
    assert manifest["failures"][-1]["bars_seen"] == 250 and "overheated" in manifest["failures"][-1]["error"]

    import pickle

    # The progress is a plain mapping: a pass run as ``__main__`` must not pickle
    # a class only that script can resolve.
    with (crashed / "_state" / "eye.pkl").open("rb") as handle:
        assert type(pickle.load(handle)[2]) is dict
    reader, observer, progress = load_checkpoint(crashed / "_state" / "eye.pkl")
    assert progress.seen == 200 and progress.last_bar_start == bars[199].start
    resumed = scan(crashed, bars, reader=reader, observer=observer, progress=progress, manifest=manifest)
    assert resumed.seen == len(bars)

    for table in ("events", "bars"):
        expected, actual = read_parts(straight, table), read_parts(crashed, table)
        assert len(expected) > 0
        pd.testing.assert_frame_equal(actual, expected)


def test_the_eye_is_not_pickled_before_its_audit_store_has_spilled(tmp_path) -> None:
    import json

    from eyes.scripts.scan_sparse_events import ScanProgress, read_parts, run_scan
    from eyes.tests.test_facts_on_every_scale import _noisy
    from shares.tests.helpers import session_bars

    bars = _noisy(session_bars(1)[:150], seed=5)
    reader, observer = _eye(tmp_path / "_state" / "journal")
    run_scan(bars, reader=reader, observer=observer, progress=ScanProgress(), out=tmp_path,
             state_path=tmp_path / "_state" / "eye.pkl", manifest={}, emit_from=bars[0].start, checkpoint_bars=50)

    assert observer.audit_store.cold_count == 0
    assert not (tmp_path / "_state" / "eye.pkl").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["progress"]["bars_seen"] == 150 and manifest["progress"]["checkpoint"] is None
    assert len(read_parts(tmp_path, "bars")) > 0
