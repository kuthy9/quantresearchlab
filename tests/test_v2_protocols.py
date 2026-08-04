from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pandas as pd
import pytest

from scripts.materialize_mbo_execution import (
    _iter_selected_parquet_records_preserving_packets,
    _replay_records,
    _verify_partition_manifest,
)
from smc_trader.calibration import (
    CalibrationError,
    model_code_fingerprint,
    monotone_reliability_points,
)
from smc_trader.execution import TopOfBook, TopOfBookExecutionProvider
from smc_trader.io import DataContinuityError, iter_completed_bars
from smc_trader.market_clock import (
    is_registered_trading_minute,
    scheduled_gap_kind,
    special_session_close,
)
from smc_trader.mbo import (
    F_BAD_TS_RECV,
    F_LAST,
    F_SNAPSHOT,
    MBOOrderBook,
    MBORecord,
    MBOReplayError,
    MinuteExecutionRealityStore,
)
from smc_trader.model import (
    Action,
    Bar,
    HypothesisSequenceState,
    MarketBelief,
    PlaybookPhase,
    SequenceStepState,
    VetoCode,
)
from smc_trader.playbook_registry import load_playbook_registry
from smc_trader.playbooks import PlaybookBrain
from smc_trader.risk import RiskLimits, StructuralRiskEngine
from smc_trader.validation import (
    FrozenPathTestRecorder,
    load_validation_protocol,
)

from .helpers import engine_snapshot, flat_account


TZ = "America/New_York"


def test_minute_execution_store_rejects_dataless_before_parquet_read(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "mbo-minute.parquet"
    source.write_bytes(b"placeholder")
    opened = False

    def reject_materialization(path):
        assert path == source.resolve()
        raise FileNotFoundError("source is a dataless placeholder")

    def unexpected_read(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("parquet reader must not open a dataless file")

    monkeypatch.setattr(
        "smc_trader.io.require_materialized",
        reject_materialization,
    )
    monkeypatch.setattr("smc_trader.mbo.pd.read_parquet", unexpected_read)
    with pytest.raises(FileNotFoundError, match="dataless"):
        MinuteExecutionRealityStore.from_parquet(source)
    assert not opened


def _record(
    *,
    action: str,
    side: str,
    price: float | None,
    size: float,
    order_id: int,
    flags: int,
    sequence: int,
) -> MBORecord:
    clock = pd.Timestamp("2024-06-03 09:30:00", tz="UTC") + pd.Timedelta(
        nanoseconds=sequence
    )
    return MBORecord(
        ts_recv=clock,
        ts_event=clock,
        publisher_id=1,
        instrument_id=100,
        action=action,
        side=side,
        price=price,
        size=size,
        order_id=order_id,
        flags=flags,
        sequence=sequence,
    )


def test_preregistered_registry_contains_exactly_three_ordered_protocols() -> None:
    registry = load_playbook_registry()
    assert len(registry.protocols) == 3
    assert all(len(protocol.required_sequence) >= 3 for protocol in registry.protocols)
    assert all(protocol.invalidation["retrospective_rewrite"] is False for protocol in registry.protocols)


def test_terminal_playbook_acknowledges_once_then_clears_stale_setup() -> None:
    snapshot = engine_snapshot()
    key = snapshot.decision.best_hypothesis_key
    assert key is not None
    hypothesis = snapshot.belief.hypotheses[key]
    terminal = replace(
        hypothesis,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=snapshot.observation.asof,
    )
    brain = PlaybookBrain()
    brain._belief = MarketBelief(
        asof=snapshot.observation.asof,
        hypotheses={key: terminal},
    )
    next_clock = snapshot.observation.asof + pd.Timedelta(minutes=1)
    observation = replace(snapshot.observation, asof=next_clock)
    updated = brain.update(observation).hypotheses[key]
    assert updated.phase is PlaybookPhase.INACTIVE
    assert updated.sequence is not None
    assert updated.sequence.setup_id is None
    assert updated.plan is None


def test_mbo_snapshot_reconstructs_real_top_of_book() -> None:
    snapshot_flag = F_SNAPSHOT
    records = (
        _record(
            action="R",
            side="N",
            price=None,
            size=0,
            order_id=0,
            flags=snapshot_flag,
            sequence=1,
        ),
        _record(
            action="A",
            side="B",
            price=20_000.0,
            size=12,
            order_id=1,
            flags=snapshot_flag,
            sequence=2,
        ),
        _record(
            action="A",
            side="A",
            price=20_000.25,
            size=9,
            order_id=2,
            flags=snapshot_flag | F_LAST,
            sequence=3,
        ),
    )
    book = MBOOrderBook(1, 100)
    quote = book.apply_complete_event(records)
    assert quote is not None
    assert quote.ask - quote.bid == 0.25
    assert quote.bid_size == 12
    assert quote.ask_size == 9
    assert quote.observed_at == records[-1].ts_recv


def test_mbo_vendor_snapshot_clear_may_precede_snapshot_flagged_adds() -> None:
    records = (
        _record(
            action="R",
            side="N",
            price=None,
            size=0,
            order_id=0,
            flags=F_BAD_TS_RECV,
            sequence=0,
        ),
        _record(
            action="A",
            side="B",
            price=20_000.0,
            size=12,
            order_id=1,
            flags=F_BAD_TS_RECV | F_SNAPSHOT,
            sequence=1,
        ),
        _record(
            action="A",
            side="A",
            price=20_000.25,
            size=9,
            order_id=2,
            flags=F_BAD_TS_RECV | F_SNAPSHOT | F_LAST,
            sequence=2,
        ),
    )
    quote = MBOOrderBook(1, 100).apply_complete_event(records)
    assert quote is not None
    assert (quote.bid, quote.ask) == (20_000.0, 20_000.25)


def test_daily_replay_accepts_separate_clear_then_snapshot_add_packet() -> None:
    records = (
        _record(
            action="R",
            side="N",
            price=None,
            size=0,
            order_id=0,
            flags=F_LAST,
            sequence=0,
        ),
        _record(
            action="A",
            side="B",
            price=20_000.0,
            size=12,
            order_id=1,
            flags=F_SNAPSHOT,
            sequence=1,
        ),
        _record(
            action="A",
            side="A",
            price=20_000.25,
            size=9,
            order_id=2,
            flags=F_SNAPSHOT | F_LAST,
            sequence=2,
        ),
    )
    decision = pd.Timestamp("2024-06-03 09:31:00", tz="UTC")
    bars = [Bar(decision, 20_000, 20_001, 19_999, 20_000, 1, "NQM4", 100)]

    rows, event_count = _replay_records(
        records,
        bars,
        end=decision + pd.Timedelta(minutes=1),
        require_initial_snapshot=True,
        selected_instrument_ids={100},
    )

    assert event_count == 2
    assert len(rows) == 1
    assert rows[0]["book_valid"]
    assert (rows[0]["bid"], rows[0]["ask"]) == (20_000.0, 20_000.25)


def test_mbo_replay_can_defer_depth_capture_without_changing_quote() -> None:
    records = (
        _record(
            action="R",
            side="N",
            price=None,
            size=0,
            order_id=0,
            flags=F_SNAPSHOT,
            sequence=0,
        ),
        _record(
            action="A",
            side="B",
            price=20_000.0,
            size=12,
            order_id=1,
            flags=F_SNAPSHOT,
            sequence=1,
        ),
        _record(
            action="A",
            side="A",
            price=20_000.25,
            size=9,
            order_id=2,
            flags=F_SNAPSHOT | F_LAST,
            sequence=2,
        ),
    )
    book = MBOOrderBook(1, 100)
    assert book.apply_complete_event(records, capture_snapshot=False) is None
    assert book.valid
    quote = book.snapshot()
    assert quote is not None
    assert (quote.bid, quote.ask, quote.bid_size, quote.ask_size) == (
        20_000.0,
        20_000.25,
        12.0,
        9.0,
    )


def test_mbo_unknown_mutation_fails_closed_until_new_snapshot() -> None:
    book = MBOOrderBook(1, 100)
    bad = _record(
        action="C",
        side="B",
        price=20_000.0,
        size=1,
        order_id=99,
        flags=F_LAST,
        sequence=1,
    )
    assert book.apply_complete_event((bad,)) is None
    assert book.requires_snapshot
    assert not book.valid


def test_dbn_enum_and_integer_codes_normalize_without_guessing_ascii() -> None:
    clock = pd.Timestamp("2024-06-03 09:30:00", tz="UTC")
    add = MBORecord.from_value(
        {
            "ts_recv": clock.value,
            "ts_event": clock.value,
            "publisher_id": 1,
            "instrument_id": 100,
            "action": 0,
            "side": 0,
            "price": 20_000.0,
            "size": 1,
            "order_id": 1,
            "flags": F_LAST,
            "sequence": 1,
        }
    )
    assert (add.action, add.side) == ("A", "B")
    clear = MBORecord.from_value(
        {
            **add.__dict__,
            "action": "Action.CLEAR",
            "side": "Side.NONE",
            "price": None,
        }
    )
    assert (clear.action, clear.side) == ("R", "N")


def test_selected_execution_instrument_still_rejects_nonpositive_price() -> None:
    bad_out = _record(
        action="A",
        side="B",
        price=-15.0,
        size=1,
        order_id=1,
        flags=F_LAST,
        sequence=1,
    )
    decision = pd.Timestamp("2024-06-03 09:31:00", tz="UTC")
    bar = Bar(decision, 100, 101, 99, 100, 1, "NQM4", 100)

    with pytest.raises(MBOReplayError, match="selected execution instrument"):
        _replay_records(
            [bad_out],
            [bar],
            end=decision + pd.Timedelta(minutes=1),
            require_initial_snapshot=False,
            selected_instrument_ids={100},
        )


def test_parquet_packet_selection_ignores_valid_negative_spread_price(
    tmp_path,
) -> None:
    clock = pd.Timestamp("2024-06-03 09:30:00", tz="UTC")
    path = tmp_path / "packet.parquet"
    pd.DataFrame(
        [
            {
                "ts_recv": clock,
                "ts_event": clock,
                "publisher_id": 1,
                "instrument_id": 100,
                "action": "A",
                "side": "B",
                "price": 20_000.0,
                "size": 1,
                "order_id": 1,
                "flags": 0,
                "sequence": 1,
            },
            {
                "ts_recv": clock + pd.Timedelta(nanoseconds=1),
                "ts_event": clock + pd.Timedelta(nanoseconds=1),
                "publisher_id": 1,
                "instrument_id": 9947,
                "action": "A",
                "side": "B",
                "price": -15.0,
                "size": 1,
                "order_id": 2,
                "flags": F_LAST,
                "sequence": 2,
            },
        ]
    ).to_parquet(path, index=False)

    records = list(
        _iter_selected_parquet_records_preserving_packets(
            path,
            batch_size=1,
            selected_instrument_ids={100},
        )
    )

    assert len(records) == 1
    assert records[0].instrument_id == 100
    assert records[0].price == 20_000.0
    assert records[0].flags & F_LAST


def test_observed_bbo_does_not_invent_constant_slippage() -> None:
    clock = pd.Timestamp("2024-06-03 09:30:00", tz="UTC")
    provider = TopOfBookExecutionProvider()
    book = TopOfBook(clock, 20_000.0, 20_000.25, 12, 9)
    observed = provider.observe(
        book,
        decision_clock=clock,
        deadline=clock + pd.Timedelta(hours=1),
        quantity=1,
    )
    assert observed.expected_slippage_points == 0.0
    assert not observed.anomalies
    insufficient = provider.observe(
        book,
        decision_clock=clock,
        deadline=clock + pd.Timedelta(hours=1),
        quantity=20,
    )
    assert "insufficient_top_of_book_depth" in insufficient.anomalies


def test_monotone_calibration_cannot_invert_probability_order() -> None:
    probabilities = [0.1] * 20 + [0.3] * 20 + [0.6] * 20 + [0.9] * 20
    outcomes = [0] * 15 + [1] * 5 + [0] * 5 + [1] * 15
    outcomes += [0] * 10 + [1] * 10 + [0] * 2 + [1] * 18
    points = monotone_reliability_points(
        probabilities,
        outcomes,
        bins=4,
        minimum_bin_episodes=20,
    )
    assert len(points) >= 2
    assert all(
        left.calibrated_probability <= right.calibrated_probability
        for left, right in zip(points[:-1], points[1:])
    )


def test_calibration_refuses_insufficient_episode_bins() -> None:
    with pytest.raises(CalibrationError):
        monotone_reliability_points(
            [0.2, 0.8],
            [0, 1],
            bins=2,
            minimum_bin_episodes=20,
        )


def test_calibration_refuses_duplicate_raw_probability_levels() -> None:
    with pytest.raises(CalibrationError):
        monotone_reliability_points(
            [0.5] * 40,
            [0, 1] * 20,
            bins=2,
            minimum_bin_episodes=20,
        )


def test_validation_protocol_protects_sealed_holdout_boundary() -> None:
    protocol = load_validation_protocol()
    window = protocol.classify_ohlcv(
        pd.Timestamp("2026-05-01", tz=TZ),
        pd.Timestamp("2026-06-01", tz=TZ),
    )
    assert window.role == "sealed_holdout"
    with pytest.raises(ValueError):
        protocol.classify_ohlcv(
            pd.Timestamp("2026-03-31", tz=TZ),
            pd.Timestamp("2026-04-02", tz=TZ),
        )


def test_frozen_path_uses_only_later_bar_and_resolves_ambiguity_as_failure() -> None:
    snapshot = engine_snapshot()
    key = snapshot.decision.best_hypothesis_key
    hypothesis = snapshot.belief.hypotheses[key]
    sequence = HypothesisSequenceState(
        protocol_version="test",
        protocol_hash="b" * 64,
        setup_id="setup",
        steps=(
            SequenceStepState(
                "one",
                True,
                1.0,
                snapshot.observation.asof - pd.Timedelta(minutes=2),
            ),
            SequenceStepState(
                "two",
                True,
                1.0,
                snapshot.observation.asof - pd.Timedelta(minutes=1),
            ),
        ),
        started_at=snapshot.observation.asof - pd.Timedelta(minutes=2),
    )
    hypothesis = replace(hypothesis, sequence=sequence)
    belief = replace(snapshot.belief, hypotheses={key: hypothesis})
    snapshot = replace(snapshot, belief=belief)
    recorder = FrozenPathTestRecorder(
        config_hash="c" * 64,
        code_hash="d" * 64,
    )
    recorder.observe(snapshot)
    bar = Bar(
        start=snapshot.observation.asof,
        open=100.0,
        high=104.0,
        low=97.0,
        close=101.0,
        volume=100,
        symbol=snapshot.observation.symbol,
        instrument_id=snapshot.observation.instrument_id,
    )
    recorder.on_bar(bar)
    result = recorder.results[0]
    assert result.outcome == "invalidation"
    assert not result.success
    assert result.ambiguous_same_bar
    assert result.decision_time < result.resolved_at
    assert result.raw_probability == hypothesis.probability
    assert result.code_hash == "d" * 64
    assert result.formation_minutes == 1
    assert result.entry_touched
    assert (
        result.invalidation_source_id
        == hypothesis.plan.invalidation.source_level_id
    )
    assert result.target_source_id == hypothesis.plan.targets[0].level_id


def test_frozen_path_requires_entry_and_never_credits_same_bar_target() -> None:
    snapshot = engine_snapshot()
    key = snapshot.decision.best_hypothesis_key
    hypothesis = snapshot.belief.hypotheses[key]
    sequence = HypothesisSequenceState(
        protocol_version="test",
        protocol_hash="b" * 64,
        setup_id="entry-ordering",
        steps=(
            SequenceStepState(
                "one",
                True,
                1.0,
                snapshot.observation.asof - pd.Timedelta(minutes=1),
            ),
        ),
        started_at=snapshot.observation.asof - pd.Timedelta(minutes=1),
    )
    hypothesis = replace(hypothesis, sequence=sequence)
    snapshot = replace(
        snapshot,
        belief=replace(snapshot.belief, hypotheses={key: hypothesis}),
    )
    recorder = FrozenPathTestRecorder(
        config_hash="c" * 64,
        code_hash="d" * 64,
    )
    recorder.observe(snapshot)

    # Target trades before the 100.0 planned entry is touched.
    recorder.on_bar(
        Bar(
            start=snapshot.observation.asof,
            open=101.0,
            high=104.0,
            low=100.5,
            close=102.0,
            volume=100,
            symbol=snapshot.observation.symbol,
            instrument_id=snapshot.observation.instrument_id,
        )
    )
    assert recorder.results == ()

    # Entry and target are both inside the next OHLC bar; target ordering is
    # unknowable, so the conservative path remains open.
    recorder.on_bar(
        Bar(
            start=snapshot.observation.asof + pd.Timedelta(minutes=1),
            open=101.0,
            high=104.0,
            low=99.0,
            close=101.0,
            volume=100,
            symbol=snapshot.observation.symbol,
            instrument_id=snapshot.observation.instrument_id,
        )
    )
    assert recorder.results == ()

    recorder.on_bar(
        Bar(
            start=snapshot.observation.asof + pd.Timedelta(minutes=2),
            open=101.0,
            high=103.5,
            low=100.5,
            close=103.0,
            volume=100,
            symbol=snapshot.observation.symbol,
            instrument_id=snapshot.observation.instrument_id,
        )
    )
    result = recorder.results[0]
    assert result.outcome == "target"
    assert result.success
    assert result.entry_touched
    assert not result.ambiguous_same_bar
    assert result.mfe_R == pytest.approx(
        (103.5 - hypothesis.plan.planned_entry)
        / hypothesis.plan.risk_points
    )


def test_model_code_fingerprint_is_stable_and_sha256_shaped() -> None:
    first = model_code_fingerprint()
    assert first == model_code_fingerprint()
    assert len(first) == 64
    int(first, 16)


def test_constant_execution_assumption_cannot_pass_entry_risk() -> None:
    snapshot = engine_snapshot()
    observation = replace(
        snapshot.observation,
        anomalies=("execution_constant_assumption",),
    )
    result = StructuralRiskEngine(RiskLimits(maximum_cost_R=0.50)).review(
        replace(snapshot.decision, asof=observation.asof),
        observation,
        flat_account(),
    )
    assert result.final_action is Action.ABSTAIN
    assert VetoCode.DATA_ANOMALY in result.vetoes


def test_registered_calendar_handles_holiday_and_good_friday_sessions() -> None:
    early = pd.Timestamp("2024-01-15 13:00", tz=TZ)
    reopen = pd.Timestamp("2024-01-15 18:00", tz=TZ)
    assert special_session_close(early) == early
    assert scheduled_gap_kind(early, reopen) == "registered_special_session_closure"
    good_friday = pd.Timestamp("2023-04-07 09:15", tz=TZ)
    sunday = pd.Timestamp("2023-04-09 18:00", tz=TZ)
    assert scheduled_gap_kind(good_friday, sunday) == (
        "registered_abbreviated_good_friday_closure"
    )
    assert not is_registered_trading_minute(
        pd.Timestamp("2026-06-19 13:00", tz=TZ)
    )
    assert is_registered_trading_minute(
        pd.Timestamp("2026-06-19 12:59", tz=TZ)
    )
    assert not is_registered_trading_minute(
        pd.Timestamp("2024-01-15 14:00", tz=TZ)
    )
    assert not is_registered_trading_minute(
        pd.Timestamp("2018-07-03 13:15", tz=TZ)
    )
    assert not is_registered_trading_minute(
        pd.Timestamp("2024-07-03 13:15", tz=TZ)
    )
    assert not is_registered_trading_minute(
        pd.Timestamp("2023-06-19 13:00", tz=TZ)
    )
    assert is_registered_trading_minute(
        pd.Timestamp("2021-11-26 13:14", tz=TZ)
    )
    assert not is_registered_trading_minute(
        pd.Timestamp("2021-11-26 13:15", tz=TZ)
    )
    first_fallback_hour = pd.Timestamp("2017-11-05 05:30", tz="UTC")
    second_fallback_hour = pd.Timestamp("2017-11-05 06:30", tz="UTC")
    assert not is_registered_trading_minute(first_fallback_hour)
    assert not is_registered_trading_minute(second_fallback_hour)


def _minute_frame(starts: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(value, tz=TZ) for value in starts])
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1.0,
            "symbol": "NQH24",
            "instrument_id": 100,
        },
        index=index,
    )


def test_open_market_no_trade_minutes_are_bounded_and_densified() -> None:
    bars = tuple(
        iter_completed_bars(
            _minute_frame(
                [
                    "2024-01-16 09:30",
                    "2024-01-16 09:34",
                ]
            )
        )
    )
    assert [bar.start.minute for bar in bars] == [30, 31, 32, 33, 34]
    assert [bar.synthetic_no_trade for bar in bars] == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_open_market_no_trade_run_over_cap_fails_closed() -> None:
    with pytest.raises(DataContinuityError, match="6 missing trading minute"):
        tuple(
            iter_completed_bars(
                _minute_frame(
                    [
                        "2024-01-16 09:30",
                        "2024-01-16 09:37",
                    ]
                )
            )
        )


def test_calendar_closure_does_not_get_synthetic_minutes() -> None:
    bars = tuple(
        iter_completed_bars(
            _minute_frame(
                [
                    "2024-01-12 16:59",
                    "2024-01-14 18:00",
                ]
            )
        )
    )
    assert len(bars) == 2
    assert not any(bar.synthetic_no_trade for bar in bars)


def test_off_session_vendor_row_is_removed_before_causal_aggregation() -> None:
    bars = tuple(
        iter_completed_bars(
            _minute_frame(
                [
                    "2018-05-29 16:14",
                    "2018-05-29 16:15",
                    "2018-05-29 16:30",
                ]
            )
        )
    )
    assert [bar.start.strftime("%H:%M") for bar in bars] == [
        "16:14",
        "16:30",
    ]
    assert not any(bar.synthetic_no_trade for bar in bars)


def test_mbo_partition_hashes_are_verified_before_replay(tmp_path) -> None:
    partition = tmp_path / "date=2024-06-03" / "part-test.parquet"
    partition.parent.mkdir()
    partition.write_bytes(b"registered-mbo-partition")
    manifest = {
        "files": [
            {
                "path": "date=2024-06-03/part-test.parquet",
                "sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fingerprint, verified = _verify_partition_manifest(tmp_path, [partition])

    assert verified == 1
    assert fingerprint == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_mbo_partition_hash_mismatch_fails_before_replay(tmp_path) -> None:
    partition = tmp_path / "date=2024-06-03" / "part-test.parquet"
    partition.parent.mkdir()
    partition.write_bytes(b"corrupted")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "date=2024-06-03/part-test.parquet",
                        "sha256": hashlib.sha256(b"expected").hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MBOReplayError, match="hash mismatch"):
        _verify_partition_manifest(tmp_path, [partition])


def test_multi_book_vendor_packet_is_split_after_complete_grouping() -> None:
    def record(
        instrument_id: int,
        action: str,
        side: str,
        price: float | None,
        order_id: int,
        flags: int,
        sequence: int,
    ) -> MBORecord:
        return replace(
            _record(
                action=action,
                side=side,
                price=price,
                size=1,
                order_id=order_id,
                flags=flags,
                sequence=sequence,
            ),
            instrument_id=instrument_id,
        )

    records = [
        record(100, "R", "N", None, 0, F_SNAPSHOT, 1),
        record(100, "A", "B", 99.0, 1, F_SNAPSHOT, 2),
        record(100, "A", "A", 101.0, 2, F_SNAPSHOT | F_LAST, 3),
        record(101, "R", "N", None, 0, F_SNAPSHOT, 4),
        record(101, "A", "B", 199.0, 3, F_SNAPSHOT, 5),
        record(101, "A", "A", 201.0, 4, F_SNAPSHOT | F_LAST, 6),
        # One complete vendor packet carries book mutations for two keys.
        record(100, "A", "B", 98.0, 5, 0, 7),
        record(101, "A", "B", 198.0, 6, F_LAST, 8),
    ]
    first_start = pd.Timestamp("2024-06-03 09:30", tz="UTC")
    bars = [
        Bar(first_start, 100, 101, 99, 100, 1, "NQM4", 100),
        Bar(first_start + pd.Timedelta(minutes=1), 200, 201, 199, 200, 1, "NQU4", 101),
    ]

    rows, event_count = _replay_records(
        records,
        bars,
        end=first_start + pd.Timedelta(minutes=3),
        require_initial_snapshot=True,
        selected_instrument_ids={100, 101},
    )

    assert event_count == 3
    assert len(rows) == 2
    assert all(row["book_valid"] for row in rows)
    assert [row["instrument_id"] for row in rows] == [100, 101]
