from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
    Timeframe,
    content_hash,
    to_primitive,
)
from smc_trader.observation import CausalObserver, ObserverConfig
from smc_trader.semantic_audit import FixedSemanticCaseSelector
import smc_trader.semantic_discovery_runner as discovery_module
from smc_trader.semantic_discovery_runner import (
    BoundedSemanticSelector,
    CasePacketTransaction,
    DiscoveryCheckpointStore,
    ParquetBatchSource,
    SemanticDiscoveryReplay,
    SemanticDiscoveryRunner,
    SourceChunk,
    _bar_sha256,
    _chain,
    _chain_primitive_mapping,
    _source_bar,
    _source_bars,
    iter_provenanced_completed_bars,
    source_row_sha256,
)


def _frame(starts: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [pd.Timestamp(value, tz="America/New_York") for value in starts],
        name="ts",
    )
    close = pd.Series(
        [100.0 + index for index in range(len(starts))],
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": 10.0,
            "symbol": "NQH3",
            "instrument_id": 1,
        },
        index=index,
    )


def _complete_histories(
    asof: pd.Timestamp,
) -> dict[Timeframe, tuple[Candle, ...]]:
    durations = {
        Timeframe.H4: 240,
        Timeframe.H1: 60,
        Timeframe.M5: 5,
        Timeframe.M1: 1,
    }
    result: dict[Timeframe, tuple[Candle, ...]] = {}
    for timeframe, duration in durations.items():
        count = 8 if timeframe is Timeframe.M1 else 1
        result[timeframe] = tuple(
            Candle(
                timeframe=timeframe,
                start=asof
                - pd.Timedelta(minutes=duration * (count - index)),
                end=asof
                - pd.Timedelta(
                    minutes=duration * (count - index - 1)
                ),
                open=99.75,
                high=100.50,
                low=99.50,
                close=100.0,
                volume=10.0,
                symbol="NQH3",
                instrument_id=1,
                observed_minutes=duration,
                expected_minutes=duration,
                complete=True,
            )
            for index in range(count)
        )
    return result


def _chunk(
    frame: pd.DataFrame,
    *,
    ordinal_start: int = 0,
) -> SourceChunk:
    return SourceChunk(
        frame=frame,
        ordinals=tuple(
            range(ordinal_start, ordinal_start + len(frame))
        ),
    )


def _audit_contract() -> dict:
    return {
        "audit_id": "SYNTHETIC-SEMANTIC-AUDIT",
        "bindings": {"primitive_protocol_sha256": "a" * 64},
        "selection": {
            "timeframes": ["1m"],
            "directions": ["long"],
            "case_classes": ["confirmed_bos"],
            "calendar_years": [2023],
            "cases_per_bucket": 1,
        },
        "review_schema": {"issue_codes": []},
        "pre_reveal_forbidden_fields": [
            "future_path",
            "outcome",
            "pnl",
            "action",
        ],
    }


def _replay(*, selector=True) -> SemanticDiscoveryReplay:
    return SemanticDiscoveryReplay(
        reader=CausalMarketReader(maximum_history=64),
        observer=CausalObserver(
            ObserverConfig(
                memory_events=32,
                minimum_bars={
                    timeframe: 1 for timeframe in Timeframe
                },
            )
        ),
        selector=(
            BoundedSemanticSelector(_audit_contract())
            if selector
            else None
        ),
    )


def _bos(
    *,
    asof: pd.Timestamp,
    direction: Direction,
    case_class: str,
    suffix: str,
) -> BreakOfStructureState:
    if case_class == "wick_only_no_close":
        lifecycle = BOSLifecycle.PENDING
        scope = BOSScope.CONTINUATION
    elif case_class == "broken_or_opposed":
        lifecycle = BOSLifecycle.CONFIRMED
        scope = BOSScope.OPPOSED
    else:
        lifecycle = BOSLifecycle.CONFIRMED
        scope = BOSScope.CONTINUATION
    return BreakOfStructureState(
        bos_id=f"bos-{suffix}",
        timeframe=Timeframe.M1,
        direction=direction,
        lifecycle=lifecycle,
        scope=scope,
        target_swing_id=f"swing-{suffix}",
        source_structure_id=f"structure-{suffix}",
        target_price=100.0,
        target_ticks=400,
        pending_at=asof - pd.Timedelta(minutes=2),
        resolved_at=(
            None if lifecycle is BOSLifecycle.PENDING else asof
        ),
        age_bars=2,
        attempt_count=(
            1 if lifecycle is BOSLifecycle.PENDING else 0
        ),
        last_attempt_at=(
            asof if lifecycle is BOSLifecycle.PENDING else None
        ),
        attempt_clocks=(
            (asof,)
            if lifecycle is BOSLifecycle.PENDING
            else ()
        ),
    )


def _semantic_fixture_objects(
    item: BreakOfStructureState,
) -> tuple[SwingPoint, StructureSequenceState | None]:
    case_clock = (
        item.resolved_at
        or item.last_attempt_at
        or item.pending_at + pd.Timedelta(minutes=1)
    )
    swing = SwingPoint(
        swing_id=item.target_swing_id,
        timeframe=item.timeframe,
        symbol="NQH3",
        instrument_id=1,
        side=(
            SwingSide.HIGH
            if item.direction is Direction.LONG
            else SwingSide.LOW
        ),
        price=item.target_price,
        price_ticks=item.target_ticks,
        pivot_start=case_clock - pd.Timedelta(minutes=5),
        pivot_end=case_clock - pd.Timedelta(minutes=4),
        observed_at=case_clock - pd.Timedelta(minutes=3),
        confirmed_at=case_clock - pd.Timedelta(minutes=3),
        lifecycle=(
            SwingLifecycle.BROKEN
            if item.lifecycle is BOSLifecycle.CONFIRMED
            else SwingLifecycle.CONFIRMED
        ),
        relation=(
            SwingRelation.HH
            if item.direction is Direction.LONG
            else SwingRelation.LL
        ),
        age_bars=3,
        broken_at=(
            case_clock
            if item.lifecycle is BOSLifecycle.CONFIRMED
            else None
        ),
        failure_reason=(
            "close_beyond_swing"
            if item.lifecycle is BOSLifecycle.CONFIRMED
            else None
        ),
    )
    if item.source_structure_id is None:
        return swing, None
    opposed = item.scope is BOSScope.OPPOSED
    source_direction = (
        (
            Direction.SHORT
            if item.direction is Direction.LONG
            else Direction.LONG
        )
        if opposed
        else item.direction
    )
    structure = StructureSequenceState(
        structure_id=item.source_structure_id,
        timeframe=item.timeframe,
        direction=source_direction,
        lifecycle=(
            StructureLifecycle.BROKEN
            if opposed
            else StructureLifecycle.CONFIRMED
        ),
        formed_at=case_clock - pd.Timedelta(minutes=8),
        confirmed_at=case_clock - pd.Timedelta(minutes=6),
        broken_at=case_clock if opposed else None,
        high_run=2,
        low_run=2,
        sequence_count=2,
        latest_high_id=(
            swing.swing_id
            if swing.side is SwingSide.HIGH
            else f"fixture-high-{item.bos_id}"
        ),
        latest_low_id=(
            swing.swing_id
            if swing.side is SwingSide.LOW
            else f"fixture-low-{item.bos_id}"
        ),
        protected_swing_id=f"fixture-protected-{item.bos_id}",
        protected_price=99.0,
        cumulative_magnitude_atr=2.0,
        age_bars=4,
        failure_reason=(
            "protected_level_close_break" if opposed else None
        ),
    )
    return swing, structure


class _FixtureSemanticReplay(SemanticDiscoveryReplay):
    """Production reader/observer with deterministic test-only BOS injection."""

    def __init__(
        self,
        *,
        selector: BoundedSemanticSelector | None,
        schedule: dict[int, tuple[BreakOfStructureState, ...]],
    ) -> None:
        super().__init__(
            reader=CausalMarketReader(maximum_history=64),
            observer=CausalObserver(
                ObserverConfig(
                    memory_events=32,
                    minimum_bars={
                        timeframe: 1 for timeframe in Timeframe
                    },
                )
            ),
            selector=selector,
        )
        self.schedule = schedule

    def on_receipt(self, receipt):
        selector = self.selector
        self.selector = None
        observation = super().on_receipt(receipt)
        self.selector = selector
        injected = (
            self.schedule.get(receipt.source_row_ordinal, ())
            if receipt.real_source_bar
            else ()
        )
        if injected:
            frame = observation.frame(Timeframe.M1)
            frames = dict(observation.frames)
            semantic_objects = tuple(
                _semantic_fixture_objects(item)
                for item in injected
            )
            frames[Timeframe.M1] = replace(
                frame,
                swings=(
                    *frame.swings,
                    *(item[0] for item in semantic_objects),
                ),
                structures=(
                    *frame.structures,
                    *(
                        item[1]
                        for item in semantic_objects
                        if item[1] is not None
                    ),
                ),
                structure_breaks=injected,
            )
            observation = replace(observation, frames=frames)
        if receipt.real_source_bar and selector is not None:
            selector.observe(
                observation,
                receipt,
                histories=_complete_histories(observation.asof),
                source_prefix_root=self.source_prefix_root,
                produced_bar_prefix_root=self.produced_bar_prefix_root,
                source_rows_admitted=self.source_rows_admitted,
                produced_bars=self.produced_bars,
                reset_epoch=self.reset_epoch,
            )
        self.last_observation = observation
        return observation


class _FixtureRunner(SemanticDiscoveryRunner):
    def __init__(self, *args, event_schedule, **kwargs):
        self._event_schedule = event_schedule
        super().__init__(*args, **kwargs)

    def _new_replay(self, *, selector):
        return _FixtureSemanticReplay(
            selector=selector,
            schedule=self._event_schedule,
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integration_contracts(
    tmp_path: Path,
    *,
    frame: pd.DataFrame,
    calendar_years: tuple[int, ...] = (2023,),
    authorization: str = "synthetic integration tests only",
    status: str = "synthetic_test_contract",
) -> tuple[
    dict,
    dict,
    ParquetBatchSource,
    Path,
    str,
    str,
    dict[str, str],
]:
    tmp_path.mkdir(parents=True)
    source_path = tmp_path / "fixture.parquet"
    frame.to_parquet(source_path)
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "tick_size": 0.25,
                "point_value": 20.0,
                "observer": {
                    "memory_events": 32,
                    "minimum_bars": {
                        "4H": 1,
                        "1H": 1,
                        "5m": 1,
                        "1m": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    audit = {
        "audit_id": "SYNTHETIC-INTEGRATION-AUDIT",
        "bindings": {
            "primitive_protocol_sha256": "a" * 64,
            "causal_source_sha256": _sha(source_path),
        },
        "source": {
            "path": str(source_path),
            "start": frame.index[0].isoformat(),
            "end_exclusive": (
                frame.index[-1] + pd.Timedelta(minutes=1)
            ).isoformat(),
            "window_role": "synthetic_fixture",
        },
        "selection": {
            "timeframes": ["1m"],
            "directions": ["long", "short"],
            "case_classes": [
                "confirmed_bos",
                "wick_only_no_close",
                "broken_or_opposed",
            ],
            "calendar_years": list(calendar_years),
            "cases_per_bucket": 1,
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(audit, sort_keys=True),
        encoding="utf-8",
    )
    implementation_hashes = {
        "semantic_discovery_runner_sha256": "1" * 64,
        "semantic_audit_sha256": "2" * 64,
        "io_sha256": "3" * 64,
        "causal_sha256": "4" * 64,
        "market_clock_sha256": "5" * 64,
        "observation_sha256": "6" * 64,
        "structure_sha256": "7" * 64,
    }
    runner_contract = {
        "runner_id": "SYNTHETIC-INTEGRATION-RUNNER",
        "status": status,
        "authorization": authorization,
        "bindings": {
            "model_config_sha256": _sha(model_path),
            "blind_audit_contract_sha256": _sha(audit_path),
        },
        "source_iterator": {
            "batch_rows": 2,
            "maximum_no_trade_gap_minutes": 5,
            "allow_data_gap_reset": True,
        },
        "implementation_hashes_required": list(
            implementation_hashes
        ),
    }
    runner_path = tmp_path / "runner.json"
    runner_path.write_text(
        json.dumps(runner_contract, sort_keys=True),
        encoding="utf-8",
    )
    source = ParquetBatchSource(
        source_path,
        start=frame.index[0],
        end_exclusive=frame.index[-1] + pd.Timedelta(minutes=1),
        batch_rows=2,
        expected_sha256=_sha(source_path),
    )
    return (
        audit,
        runner_contract,
        source,
        model_path,
        _sha(audit_path),
        _sha(runner_path),
        implementation_hashes,
    )


def _event_schedule(
    frame: pd.DataFrame,
) -> dict[int, tuple[BreakOfStructureState, ...]]:
    schedule = {}
    classes = (
        "confirmed_bos",
        "wick_only_no_close",
        "broken_or_opposed",
    )
    ordinal = 0
    for direction in (Direction.LONG, Direction.SHORT):
        for case_class in classes:
            asof = frame.index[ordinal] + pd.Timedelta(minutes=1)
            schedule[ordinal] = (
                _bos(
                    asof=asof,
                    direction=direction,
                    case_class=case_class,
                    suffix=f"{direction.value}-{case_class}",
                ),
            )
            ordinal += 1
    return schedule


def _build_fixture_runner(
    tmp_path: Path,
    *,
    output_name: str,
    frame: pd.DataFrame,
    schedule: dict[int, tuple[BreakOfStructureState, ...]],
    checkpoint_source_rows: int = 2,
    calendar_years: tuple[int, ...] = (2023,),
    authorization: str = "synthetic integration tests only",
    status: str = "synthetic_test_contract",
) -> _FixtureRunner:
    (
        audit,
        runner_contract,
        source,
        model_path,
        audit_hash,
        runner_hash,
        implementation_hashes,
    ) = _integration_contracts(
        tmp_path / f"contracts-{output_name}",
        frame=frame,
        calendar_years=calendar_years,
        authorization=authorization,
        status=status,
    )
    return _FixtureRunner(
        audit_contract=audit,
        runner_contract=runner_contract,
        source=source,
        model_config_path=model_path,
        output_root=tmp_path / output_name,
        audit_contract_sha256=audit_hash,
        runner_contract_sha256=runner_hash,
        implementation_hashes=implementation_hashes,
        maximum_history=64,
        checkpoint_source_rows=checkpoint_source_rows,
        event_schedule=schedule,
    )


def _clone_fixture_runner(
    base: _FixtureRunner,
    *,
    output_root: Path,
    schedule: dict[int, tuple[BreakOfStructureState, ...]],
) -> _FixtureRunner:
    return _FixtureRunner(
        audit_contract=base.audit_contract,
        runner_contract=base.runner_contract,
        source=base.source,
        model_config_path=base.model_config_path,
        output_root=output_root,
        audit_contract_sha256=base.audit_contract_sha256,
        runner_contract_sha256=base.runner_contract_sha256,
        implementation_hashes=base.implementation_hashes,
        maximum_history=base.maximum_history,
        checkpoint_source_rows=base.checkpoint_source_rows,
        event_schedule=schedule,
    )


def _run(
    replay: SemanticDiscoveryReplay,
    chunks: list[SourceChunk],
) -> list:
    observations = []
    for receipt in iter_provenanced_completed_bars(
        chunks,
        prior_source_bar=replay.last_source_bar,
        prior_source_ordinal=replay.last_source_row_ordinal,
    ):
        observations.append(replay.on_receipt(receipt))
    return observations


def _manifest(
    replay: SemanticDiscoveryReplay,
    *,
    source_start: pd.Timestamp,
    source_end: pd.Timestamp,
) -> dict:
    assert replay.selector is not None
    return replay.selector.manifest(
        audit_contract_sha256="b" * 64,
        runner_contract_sha256="c" * 64,
        engine_output_root="/synthetic/engine-output",
        source_sha256="d" * 64,
        source_start=source_start,
        source_end_exclusive=source_end,
        source_rows=replay.source_rows_admitted,
        produced_bars=replay.produced_bars,
        source_prefix_root=replay.source_prefix_root,
        produced_bar_prefix_root=replay.produced_bar_prefix_root,
        implementation_hashes={"runner": "e" * 64},
        iterator_bindings={
            "maximum_history": 64,
            "maximum_no_trade_gap_minutes": 5,
            "allow_data_gap_reset": True,
            "source_batch_rows": 2,
        },
    )


def test_optimized_receipt_hashes_and_chains_are_legacy_exact() -> None:
    bars = (
        discovery_module.Bar(
            start=pd.Timestamp(
                "2023-11-05 01:30:00-04:00"
            ),
            open=-0.0,
            high=1.0e-12,
            low=-1.0e-12,
            close=0.0,
            volume=1.0e12,
            symbol="NQ-合约",
            instrument_id=2**31,
        ),
        discovery_module.Bar(
            start=pd.Timestamp(
                "2023-11-05 01:31:00-04:00"
            ),
            open=1.25,
            high=1.5,
            low=1.0,
            close=1.25,
            volume=0.0,
            symbol="NQ-合约",
            instrument_id=2**31,
            synthetic_no_trade=True,
        ),
        discovery_module.Bar(
            start=pd.Timestamp(
                "2023-11-05 01:32:00-04:00"
            ),
            open=2.0,
            high=3.0,
            low=1.0,
            close=2.5,
            volume=9.0,
            symbol="NQ-合约",
            instrument_id=2**31,
            data_gap_before_minutes=17,
        ),
    )
    for bar in bars:
        assert _bar_sha256(bar) == content_hash(bar)
    source = bars[0]
    source_mapping = {
        "source_row_ordinal": 2**40,
        "start": source.start,
        "open": source.open,
        "high": source.high,
        "low": source.low,
        "close": source.close,
        "volume": source.volume,
        "symbol": source.symbol,
        "instrument_id": source.instrument_id,
    }
    assert source_row_sha256(source, 2**40) == content_hash(
        source_mapping
    )
    primitive = {
        "clock": source.start.isoformat(),
        "unicode": "合约",
        "negative_zero": -0.0,
        "enabled": True,
        "missing": None,
    }
    assert _chain_primitive_mapping("0" * 64, primitive) == _chain(
        "0" * 64,
        primitive,
    )


def test_optimized_source_bar_construction_is_legacy_exact() -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
            "2023-01-03 18:02",
        ]
    )
    frame = frame[
        [
            "instrument_id",
            "symbol",
            "volume",
            "close",
            "low",
            "high",
            "open",
        ]
    ]
    assert _source_bars(frame) == tuple(
        _source_bar(frame, position)
        for position in range(len(frame))
    )


def test_integrated_fast_commitments_match_legacy_projection() -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:02",
            "2023-01-03 18:03",
        ]
    )
    receipts = tuple(
        iter_provenanced_completed_bars([_chunk(frame)])
    )
    source_root = "0" * 64
    produced_root = "0" * 64
    source_rows = 0
    for produced_number, receipt in enumerate(receipts, start=1):
        assert receipt.bar_sha256 == content_hash(receipt.bar)
        if receipt.real_source_bar:
            source_root = _chain(
                source_root,
                {
                    "source_row_ordinal": receipt.source_row_ordinal,
                    "source_row_start": receipt.source_row_start,
                    "source_row_sha256": receipt.source_row_sha256,
                },
            )
            source_rows += 1
        produced_root = _chain(
            produced_root,
            {
                "produced_bar_number": produced_number,
                "provenance": receipt.provenance,
                "source_row_start": receipt.source_row_start,
                "source_row_ordinal": receipt.source_row_ordinal,
                "source_row_sha256": receipt.source_row_sha256,
                "bar_sha256": receipt.bar_sha256,
                "synthetic_no_trade": receipt.bar.synthetic_no_trade,
                "data_gap_before_minutes": (
                    receipt.bar.data_gap_before_minutes
                ),
            },
        )
    replay = _replay(selector=False)
    for receipt in receipts:
        replay.on_receipt(receipt)
    assert replay.source_rows_admitted == source_rows
    assert replay.source_prefix_root == source_root
    assert replay.produced_bar_prefix_root == produced_root


def test_densification_provenance_and_synthetic_anchor_rejection() -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:02",
        ]
    )
    receipts = list(
        iter_provenanced_completed_bars([_chunk(frame)])
    )
    assert [item.provenance for item in receipts] == [
        "real_source_bar",
        "synthetic_backfill",
        "real_source_bar",
    ]
    assert receipts[1].bar.start == pd.Timestamp(
        "2023-01-03 18:01",
        tz="America/New_York",
    )
    assert receipts[1].source_row_start == receipts[2].bar.start
    assert all(
        item.max_source_row_admitted < item.bar.end
        for item in (receipts[0], receipts[2])
    )

    replay = _replay(selector=False)
    observations = [
        replay.on_receipt(receipt) for receipt in receipts
    ]
    assert replay.synthetic_bars == 1
    assert replay.produced_bars == 3
    assert len(replay.reader.window(Timeframe.M1, 10)) == 3
    selector = BoundedSemanticSelector(_audit_contract())
    with pytest.raises(ValueError, match="synthetic transitions"):
        selector.observe(
            observations[1],
            receipts[1],
            histories={},
            source_prefix_root="0" * 64,
            produced_bar_prefix_root="0" * 64,
            source_rows_admitted=1,
            produced_bars=2,
            reset_epoch=0,
        )


def test_chunk_boundaries_do_not_change_provenance_or_roots() -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:02",
            "2023-01-03 18:03",
            "2023-01-03 18:05",
        ]
    )
    one = _replay()
    _run(one, [_chunk(frame)])
    many = _replay()
    _run(
        many,
        [
            _chunk(frame.iloc[:2], ordinal_start=0),
            _chunk(frame.iloc[2:], ordinal_start=2),
        ],
    )
    assert many.source_prefix_root == one.source_prefix_root
    assert (
        many.produced_bar_prefix_root
        == one.produced_bar_prefix_root
    )
    assert content_hash(many.last_observation) == content_hash(
        one.last_observation
    )


def test_checkpoint_resume_matches_continuous_manifest_bitwise(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
            "2023-01-03 18:03",
            "2023-01-03 18:04",
            "2023-01-03 18:06",
            "2023-01-03 18:07",
        ]
    )
    start = frame.index[0]
    end = frame.index[-1] + pd.Timedelta(minutes=1)
    continuous = _replay()
    _run(continuous, [_chunk(frame)])
    continuous_manifest = _manifest(
        continuous,
        source_start=start,
        source_end=end,
    )

    interrupted = _replay()
    first = _chunk(frame.iloc[:3], ordinal_start=0)
    _run(interrupted, [first])
    store = DiscoveryCheckpointStore(tmp_path / "checkpoint")
    bindings = {"run": "synthetic", "gap_cap": 5}
    state = {
        "replay": interrupted,
        "committed_cases": set(),
        "selection_manifest_sha256": None,
    }
    store.save(state, bindings=bindings, phase="pass1")
    resumed_state = store.load(
        expected_bindings=bindings,
        phase="pass1",
    )
    resumed = resumed_state["replay"]
    _run(
        resumed,
        [_chunk(frame.iloc[3:], ordinal_start=3)],
    )
    resumed_manifest = _manifest(
        resumed,
        source_start=start,
        source_end=end,
    )
    assert json.dumps(
        to_primitive(continuous_manifest),
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        to_primitive(resumed_manifest),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_checkpoint_rejects_binding_and_state_hash_changes(
    tmp_path: Path,
) -> None:
    replay = _replay()
    frame = _frame(["2023-01-03 18:00"])
    _run(replay, [_chunk(frame)])
    store = DiscoveryCheckpointStore(tmp_path / "checkpoint")
    state = {
        "replay": replay,
        "committed_cases": set(),
        "selection_manifest_sha256": None,
    }
    store.save(state, bindings={"source": "one"}, phase="pass1")
    with pytest.raises(ValueError, match="bindings changed"):
        store.load(
            expected_bindings={"source": "two"},
            phase="pass1",
        )
    manifest = json.loads(store.manifest_path.read_text())
    state_path = store.root / manifest["state_file"]
    state_path.write_bytes(state_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest is invalid"):
        store.load(
            expected_bindings={"source": "one"},
            phase="pass1",
        )


def test_unknown_gap_resets_and_resume_preserves_reset_epoch(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:10",
            "2023-01-03 18:11",
        ]
    )
    continuous = _replay(selector=False)
    _run(continuous, [_chunk(frame)])
    assert continuous.synthetic_bars == 0
    assert continuous.reset_epoch == 1

    interrupted = _replay(selector=False)
    _run(interrupted, [_chunk(frame.iloc[:2], ordinal_start=0)])
    store = DiscoveryCheckpointStore(tmp_path / "checkpoint")
    state = {
        "replay": interrupted,
        "committed_cases": set(),
        "selection_manifest_sha256": None,
    }
    bindings = {"allow_data_gap_reset": True, "gap_cap": 5}
    store.save(state, bindings=bindings, phase="pass1")
    resumed = store.load(
        expected_bindings=bindings,
        phase="pass1",
    )["replay"]
    _run(resumed, [_chunk(frame.iloc[2:], ordinal_start=2)])
    assert resumed.reset_epoch == continuous.reset_epoch
    assert resumed.source_prefix_root == continuous.source_prefix_root
    assert (
        resumed.produced_bar_prefix_root
        == continuous.produced_bar_prefix_root
    )


def test_parquet_source_rejects_future_rows_and_is_bounded(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
            "2023-01-03 18:02",
            "2023-01-03 18:03",
        ]
    )
    source_path = tmp_path / "source.parquet"
    frame.to_parquet(source_path)
    cutoff = pd.Timestamp(
        "2023-01-03 18:03",
        tz="America/New_York",
    )
    source = ParquetBatchSource(
        source_path,
        start=frame.index[0],
        end_exclusive=cutoff,
        batch_rows=2,
    )
    chunks = list(source.iter_chunks())
    assert chunks
    assert max(len(chunk.frame) for chunk in chunks) <= 2
    assert max(
        timestamp
        for chunk in chunks
        for timestamp in chunk.frame.index
    ) < cutoff
    assert source.count_rows() == 3


def _packet_producer(authority: Path, blind: Path) -> None:
    authority.mkdir(parents=True)
    blind.mkdir(parents=True)
    (authority / "authority.json").write_text("{}", encoding="utf-8")
    (blind / "case.png").write_bytes(b"png")
    (blind / "blind_raw_evidence.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (blind / "blind_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (blind / "review_template.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (blind / "BLIND_PACKET.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_packet_transaction_adopts_complete_orphan_and_never_overwrites(
    tmp_path: Path,
) -> None:
    transaction = CasePacketTransaction(tmp_path / "packets")
    metadata = {
        "case_clock": "2023-01-03T18:01:00-05:00",
        "prefix_commitment": "f" * 64,
    }
    final, marker_hash = transaction.publish(
        semantic_event_id_value="event-1",
        completion_metadata=metadata,
        producer=_packet_producer,
    )
    assert final.is_dir()
    assert len(marker_hash) == 64

    def must_not_run(authority: Path, blind: Path) -> None:
        raise AssertionError("existing transaction was overwritten")

    same, same_hash = transaction.publish(
        semantic_event_id_value="event-1",
        completion_metadata=metadata,
        producer=must_not_run,
    )
    assert same == final
    assert same_hash == marker_hash

    staging = transaction.staging_root / final.name
    os.replace(final, staging)
    adopted, adopted_hash = transaction.publish(
        semantic_event_id_value="event-1",
        completion_metadata=metadata,
        producer=must_not_run,
    )
    assert adopted == final
    assert adopted_hash == marker_hash
    with pytest.raises(ValueError, match="identity differs"):
        transaction.publish(
            semantic_event_id_value="event-1",
            completion_metadata={**metadata, "reset_epoch": 1},
            producer=must_not_run,
        )


def test_packet_partial_orphan_fails_closed(
    tmp_path: Path,
) -> None:
    transaction = CasePacketTransaction(tmp_path / "packets")
    case_id = transaction.case_id("event-partial")
    partial = transaction.staging_root / case_id
    partial.mkdir(parents=True)
    (partial / "authority").mkdir()
    with pytest.raises(ValueError, match="partial"):
        transaction.publish(
            semantic_event_id_value="event-partial",
            completion_metadata={"prefix": "0" * 64},
            producer=_packet_producer,
        )
    assert partial.is_dir()
    assert not (transaction.cases_root / case_id).exists()


def test_full_pass1_continuous_and_arbitrary_resume_are_bitwise_identical(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:02",
            "2023-01-03 18:03",
            "2023-01-03 18:04",
            "2023-01-03 18:05",
            "2023-01-03 18:06",
        ]
    )
    schedule = _event_schedule(frame)
    continuous = _build_fixture_runner(
        tmp_path,
        output_name="continuous",
        frame=frame,
        schedule=schedule,
    )
    continuous_path = continuous.run_pass1()
    resumed = _clone_fixture_runner(
        continuous,
        output_root=tmp_path / "resumed",
        schedule=schedule,
    )
    with pytest.raises(RuntimeError, match="intentional"):
        resumed.run_pass1(
            diagnostic_stop_after_source_rows=3,
        )
    resumed_path = resumed.run_pass1(resume=True)
    continuous_manifest = json.loads(
        continuous_path.read_text(encoding="utf-8")
    )
    resumed_manifest = json.loads(
        resumed_path.read_text(encoding="utf-8")
    )
    assert continuous_manifest.pop("engine_output_root")
    assert resumed_manifest.pop("engine_output_root")
    assert continuous_manifest == resumed_manifest
    manifest = json.loads(continuous_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["expected_bucket_count"] == 6
    assert manifest["selected_case_count"] == 6
    assert manifest["missing_buckets"] == []
    assert any(
        case["produced_bars"] > case["source_rows_admitted"]
        for case in manifest["cases"]
    )
    progress = json.loads(
        (continuous.output_root / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["resume_supported"] is True
    assert len(progress["durable_checkpoint_state_sha256"]) == 64
    assert progress["selection_progress"]["missing_buckets"] == []
    assert progress["causal_clock"] == (
        frame.index[-1] + pd.Timedelta(minutes=1)
    ).isoformat()


def test_pass1_shortage_is_unavailable_and_pass2_never_reads_source(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
        ]
    )
    runner = _build_fixture_runner(
        tmp_path,
        output_name="shortage",
        frame=frame,
        schedule={},
    )
    selection_path = runner.run_pass1()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["status"] == "unavailable"
    assert len(selection["missing_buckets"]) == 6
    original_count = runner.source.count_rows

    def forbidden_count():
        raise AssertionError("pass2 read source after shortage")

    runner.source.count_rows = forbidden_count
    with pytest.raises(ValueError, match="binding changed: status"):
        runner.run_pass2()
    runner.source.count_rows = original_count


def test_selection_tamper_matrix_is_rejected_before_pass2(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
            "2023-01-03 18:02",
            "2023-01-03 18:03",
            "2023-01-03 18:04",
            "2023-01-03 18:05",
        ]
    )
    runner = _build_fixture_runner(
        tmp_path,
        output_name="tamper",
        frame=frame,
        schedule=_event_schedule(frame),
    )
    selection = json.loads(
        runner.run_pass1().read_text(encoding="utf-8")
    )
    mutations = []
    changed_implementation = json.loads(json.dumps(selection))
    changed_implementation["implementation_hashes"][
        "structure_sha256"
    ] = "f" * 64
    mutations.append(changed_implementation)
    wrong_interval = json.loads(json.dumps(selection))
    wrong_interval["source_end_exclusive"] = (
        frame.index[-1] + pd.Timedelta(minutes=2)
    ).isoformat()
    mutations.append(wrong_interval)
    duplicate = json.loads(json.dumps(selection))
    duplicate["cases"][1]["semantic_event_id"] = duplicate["cases"][0][
        "semantic_event_id"
    ]
    mutations.append(duplicate)
    bad_score = json.loads(json.dumps(selection))
    bad_score["cases"][0]["selection_score"] = "0" * 64
    mutations.append(bad_score)
    missing_case = json.loads(json.dumps(selection))
    missing_case["cases"].pop()
    mutations.append(missing_case)
    synthetic_case = json.loads(json.dumps(selection))
    synthetic_case["cases"][0]["case_bar_synthetic"] = True
    mutations.append(synthetic_case)
    extra_top_field = json.loads(json.dumps(selection))
    extra_top_field["unregistered"] = True
    mutations.append(extra_top_field)
    extra_case_field = json.loads(json.dumps(selection))
    extra_case_field["cases"][0]["unregistered"] = True
    mutations.append(extra_case_field)
    for mutation in mutations:
        with pytest.raises(ValueError):
            runner._validate_selection_authority(mutation)


def test_min_hash_selector_matches_fixed_selector_for_all_three_classes(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            f"2023-01-03 18:{minute:02d}"
            for minute in range(18)
        ]
    )
    contract = {
        **_audit_contract(),
        "selection": {
            "timeframes": ["1m"],
            "directions": ["long", "short"],
            "case_classes": [
                "confirmed_bos",
                "wick_only_no_close",
                "broken_or_opposed",
            ],
            "calendar_years": [2023],
            "cases_per_bucket": 1,
        },
    }
    bounded = BoundedSemanticSelector(contract)
    fixed = FixedSemanticCaseSelector(contract)
    receipts = list(
        iter_provenanced_completed_bars([_chunk(frame)])
    )
    classes = (
        "confirmed_bos",
        "wick_only_no_close",
        "broken_or_opposed",
    )
    for index, receipt in enumerate(receipts):
        direction = (
            Direction.LONG if (index // 9) == 0 else Direction.SHORT
        )
        case_class = classes[(index // 3) % 3]
        replay = _replay(selector=False)
        current = replay.on_receipt(receipt)
        item = _bos(
            asof=current.asof,
            direction=direction,
            case_class=case_class,
            suffix=f"{direction.value}-{case_class}-{index}",
        )
        swing, structure = _semantic_fixture_objects(item)
        current = replace(
            current,
            frames={
                **current.frames,
                Timeframe.M1: replace(
                    current.frame(Timeframe.M1),
                    swings=(swing,),
                    structures=(
                        ()
                        if structure is None
                        else (structure,)
                    ),
                    structure_breaks=(item,),
                ),
            },
        )
        fixed.observe(current)
        bounded.observe(
            current,
            receipt,
            histories=_complete_histories(current.asof),
            source_prefix_root=f"{index + 1:064x}",
            produced_bar_prefix_root=f"{index + 101:064x}",
            source_rows_admitted=index + 1,
            produced_bars=index + 1,
            reset_epoch=0,
        )
    fixed_ids = {
        key: [case.semantic_event_id for case in values]
        for key, values in fixed._buckets.items()
    }
    bounded_ids = {
        key: [case["semantic_event_id"] for case in values]
        for key, values in bounded._buckets.items()
    }
    assert bounded_ids == fixed_ids
    assert len(bounded_ids) == 6


def test_selector_excludes_post_reset_incomplete_context() -> None:
    frame = _frame(
        [
            f"2023-01-03 18:{minute:02d}"
            for minute in range(8)
        ]
    )
    receipt = list(
        iter_provenanced_completed_bars([_chunk(frame)])
    )[-1]
    replay = _replay(selector=False)
    current = replay.on_receipt(receipt)
    item = _bos(
        asof=current.asof,
        direction=Direction.LONG,
        case_class="confirmed_bos",
        suffix="context-complete",
    )
    swing, structure = _semantic_fixture_objects(item)
    current = replace(
        current,
        frames={
            **current.frames,
            Timeframe.M1: replace(
                current.frame(Timeframe.M1),
                swings=(swing,),
                structures=(structure,),
                structure_breaks=(item,),
            ),
        },
    )
    contract = {
        **_audit_contract(),
        "selection": {
            "timeframes": ["1m"],
            "directions": ["long"],
            "case_classes": ["confirmed_bos"],
            "calendar_years": [2023],
            "cases_per_bucket": 1,
        },
    }
    complete = _complete_histories(current.asof)
    incomplete = {
        **complete,
        Timeframe.H4: (),
    }
    excluded = BoundedSemanticSelector(contract)
    excluded.observe(
        current,
        receipt,
        histories=incomplete,
        source_prefix_root="1" * 64,
        produced_bar_prefix_root="2" * 64,
        source_rows_admitted=8,
        produced_bars=8,
        reset_epoch=1,
    )
    assert excluded._buckets == {}
    included = BoundedSemanticSelector(contract)
    included.observe(
        current,
        receipt,
        histories=complete,
        source_prefix_root="1" * 64,
        produced_bar_prefix_root="2" * 64,
        source_rows_admitted=8,
        produced_bars=8,
        reset_epoch=1,
    )
    assert sum(len(values) for values in included._buckets.values()) == 1


def test_full_pass2_interruption_resume_and_global_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
            "2023-01-03 18:02",
            "2023-01-03 18:03",
            "2023-01-03 18:04",
            "2023-01-03 18:05",
        ]
    )
    schedule = _event_schedule(frame)
    continuous = _build_fixture_runner(
        tmp_path,
        output_name="pass2-continuous",
        frame=frame,
        schedule=schedule,
    )
    continuous.run_pass1()
    resumed = _clone_fixture_runner(
        continuous,
        output_root=tmp_path / "pass2-resumed",
        schedule=schedule,
    )
    resumed.run_pass1()

    def fake_materialize(**kwargs):
        _packet_producer(
            Path(kwargs["authority_root"]),
            Path(kwargs["blind_root"]),
        )

    monkeypatch.setattr(
        discovery_module,
        "materialize_blind_unit",
        fake_materialize,
    )
    continuous_global = continuous.run_pass2()
    with pytest.raises(RuntimeError, match="intentional"):
        resumed.run_pass2(
            diagnostic_stop_after_source_rows=3,
        )
    interrupted_progress = json.loads(
        (resumed.output_root / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert interrupted_progress["status"] == "failed"
    assert interrupted_progress["committed_cases"] == 3
    resumed_global = resumed.run_pass2(resume=True)
    payload = json.loads(continuous_global.read_text(encoding="utf-8"))
    resumed_payload = json.loads(
        resumed_global.read_text(encoding="utf-8")
    )
    root_specific_fields = {
        "engine_output_root",
        "selection_manifest_sha256",
        "final_checkpoint_state_sha256",
        "case_commit_hashes",
    }
    assert {
        key: value
        for key, value in payload.items()
        if key not in root_specific_fields
    } == {
        key: value
        for key, value in resumed_payload.items()
        if key not in root_specific_fields
    }
    assert payload["engine_output_root"] != resumed_payload[
        "engine_output_root"
    ]
    assert payload["selection_manifest_sha256"] != resumed_payload[
        "selection_manifest_sha256"
    ]
    assert set(payload["case_commit_hashes"]) == set(
        resumed_payload["case_commit_hashes"]
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value) is not None
        for manifest in (payload, resumed_payload)
        for value in manifest["case_commit_hashes"].values()
    )
    assert all(
        payload["case_commit_hashes"][event_id]
        != resumed_payload["case_commit_hashes"][event_id]
        for event_id in payload["case_commit_hashes"]
    )
    assert payload["case_count"] == 6
    assert len(payload["case_commit_hashes"]) == 6
    assert len(payload["final_checkpoint_state_sha256"]) == 64
    assert payload["final_causal_clock"] == (
        frame.index[-1] + pd.Timedelta(minutes=1)
    ).isoformat()


def test_pass2_publishes_all_same_clock_cases_before_next_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(
        [
            f"2023-01-03 18:{minute:02d}"
            for minute in range(6)
        ]
    )
    asof = frame.index[0] + pd.Timedelta(minutes=1)
    same_clock = tuple(
        _bos(
            asof=asof,
            direction=direction,
            case_class=case_class,
            suffix=f"same-{direction.value}-{case_class}",
        )
        for direction in (Direction.LONG, Direction.SHORT)
        for case_class in (
            "confirmed_bos",
            "wick_only_no_close",
            "broken_or_opposed",
        )
    )
    runner = _build_fixture_runner(
        tmp_path,
        output_name="pass2-same-clock",
        frame=frame,
        schedule={0: same_clock},
    )
    selection_path = runner.run_pass1()
    selection = json.loads(
        selection_path.read_text(encoding="utf-8")
    )
    assert selection["selected_case_count"] == 6
    published: list[str] = []

    def fake_materialize(**kwargs):
        published.append(
            str(kwargs["selected_case"]["semantic_event_id"])
        )
        _packet_producer(
            Path(kwargs["authority_root"]),
            Path(kwargs["blind_root"]),
        )

    monkeypatch.setattr(
        discovery_module,
        "materialize_blind_unit",
        fake_materialize,
    )
    original_receipts = runner._receipts

    def guarded_receipts(replay):
        for receipt in original_receipts(replay):
            if receipt.source_row_ordinal > 0:
                assert len(published) == 6
            yield receipt

    monkeypatch.setattr(runner, "_receipts", guarded_receipts)
    manifest_path = runner.run_pass2()
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    assert len(published) == 6
    assert manifest["case_count"] == 6


@pytest.mark.historical_frozen
def test_api_and_cli_execution_locks_are_both_fail_closed(
    tmp_path: Path,
) -> None:
    frame = _frame(["2023-01-03 18:00"])
    locked = _build_fixture_runner(
        tmp_path,
        output_name="api-locked",
        frame=frame,
        schedule={},
        authorization=(
            "implementation and synthetic tests only; real discovery "
            "execution is forbidden until a later governance release"
        ),
        status="implementation_contract_frozen",
    )
    with pytest.raises(PermissionError, match="not authorized"):
        locked.run_pass1()
    assert not locked.output_root.exists()
    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_v3_exp001_semantic_discovery.py",
            "pass1",
            "--output",
            str(tmp_path / "cli-locked"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "PermissionError" in process.stderr
    assert not (tmp_path / "cli-locked").exists()


def test_checkpoint_invalid_entries_and_manifest_atomic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = DiscoveryCheckpointStore(tmp_path / "invalid-checkpoint")
    invalid.root.mkdir()
    invalid.manifest_path.write_text("{broken", encoding="utf-8")
    assert invalid.status == "invalid"
    with pytest.raises(ValueError, match="refusing overwrite"):
        invalid.require_absent()

    destination = tmp_path / "authority.json"
    original_link = os.link

    def fail_link(source, target):
        raise OSError("fault injection before publish")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="fault injection"):
        discovery_module._write_new_json(
            destination,
            {"complete": True},
        )
    assert not destination.exists()
    monkeypatch.setattr(os, "link", original_link)
    discovery_module._write_new_json(
        destination,
        {"complete": True},
    )
    assert json.loads(destination.read_text()) == {"complete": True}


def test_packet_extra_file_and_symlink_are_rejected(
    tmp_path: Path,
) -> None:
    transaction = CasePacketTransaction(tmp_path / "packets")
    final, _ = transaction.publish(
        semantic_event_id_value="extra-event",
        completion_metadata={"prefix": "0" * 64},
        producer=_packet_producer,
    )
    (final / "blind" / "unbound.json").write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unbound entries"):
        transaction.publish(
            semantic_event_id_value="extra-event",
            completion_metadata={"prefix": "0" * 64},
            producer=_packet_producer,
        )
    symlink_case = transaction.case_id("symlink-event")
    transaction.cases_root.mkdir(exist_ok=True)
    (transaction.cases_root / symlink_case).symlink_to(final)
    with pytest.raises(ValueError, match="symlink"):
        transaction.publish(
            semantic_event_id_value="symlink-event",
            completion_metadata={"prefix": "1" * 64},
            producer=_packet_producer,
        )


def test_observer_side_failure_never_advances_durable_checkpoint(
    tmp_path: Path,
) -> None:
    frame = _frame(
        [
            "2023-01-03 18:00",
            "2023-01-03 18:01",
        ]
    )
    future_clock = frame.index[1] + pd.Timedelta(minutes=2)
    invalid_future = _bos(
        asof=future_clock,
        direction=Direction.LONG,
        case_class="confirmed_bos",
        suffix="future-fault",
    )
    runner = _build_fixture_runner(
        tmp_path,
        output_name="observer-fault",
        frame=frame,
        schedule={1: (invalid_future,)},
        checkpoint_source_rows=1,
    )
    with pytest.raises(ValueError, match="future-known"):
        runner.run_pass1()
    checkpoint_manifest = json.loads(
        (
            runner.output_root
            / "_checkpoint"
            / "pass1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    progress = json.loads(
        (runner.output_root / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint_manifest["source_rows_admitted"] == 1
    assert progress["source_rows_admitted"] == 1
    assert progress["status"] == "failed"
    assert (
        progress["durable_checkpoint_state_sha256"]
        == checkpoint_manifest["state_sha256"]
    )


def test_constructor_rejects_model_and_implementation_binding_mismatch(
    tmp_path: Path,
) -> None:
    frame = _frame(["2023-01-03 18:00"])
    base = _build_fixture_runner(
        tmp_path,
        output_name="binding-base",
        frame=frame,
        schedule={},
    )
    original_model = base.model_config_path.read_bytes()
    base.model_config_path.write_text(
        '{"observer":{"memory_events":99}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model config"):
        _clone_fixture_runner(
            base,
            output_root=tmp_path / "model-mismatch",
            schedule={},
        )
    base.model_config_path.write_bytes(original_model)
    broken_hashes = dict(base.implementation_hashes)
    broken_hashes.pop("structure_sha256")
    with pytest.raises(ValueError, match="hash family"):
        _FixtureRunner(
            audit_contract=base.audit_contract,
            runner_contract=base.runner_contract,
            source=base.source,
            model_config_path=base.model_config_path,
            output_root=tmp_path / "implementation-mismatch",
            audit_contract_sha256=base.audit_contract_sha256,
            runner_contract_sha256=base.runner_contract_sha256,
            implementation_hashes=broken_hashes,
            maximum_history=64,
            checkpoint_source_rows=2,
            event_schedule={},
        )


@pytest.mark.parametrize(
    "pollution",
    ("extra_file", "extra_directory", "tree_symlink"),
)
def test_first_packet_publication_rejects_polluted_producer(
    tmp_path: Path,
    pollution: str,
) -> None:
    transaction = CasePacketTransaction(
        tmp_path / f"packets-{pollution}"
    )
    event_id = f"polluted-{pollution}"

    def polluted_producer(authority: Path, blind: Path) -> None:
        _packet_producer(authority, blind)
        if pollution == "extra_file":
            (blind / "unbound.json").write_text(
                "{}",
                encoding="utf-8",
            )
        elif pollution == "extra_directory":
            (blind / "unbound-directory").mkdir()
        else:
            (blind / "unbound-link").symlink_to(
                "../authority/authority.json"
            )

    expected = (
        "symlink" if pollution == "tree_symlink" else "unbound entries"
    )
    with pytest.raises(ValueError, match=expected):
        transaction.publish(
            semantic_event_id_value=event_id,
            completion_metadata={"prefix": "8" * 64},
            producer=polluted_producer,
        )
    case_id = transaction.case_id(event_id)
    assert not (transaction.cases_root / case_id).exists()
    assert (transaction.staging_root / case_id).is_dir()


def test_selection_rejects_cross_year_field_swap_with_balanced_buckets(
    tmp_path: Path,
) -> None:
    starts = [
        f"2022-12-29 18:{minute:02d}" for minute in range(6)
    ] + [
        f"2023-01-03 18:{minute:02d}" for minute in range(6)
    ]
    frame = _frame(starts)
    schedule: dict[int, tuple[BreakOfStructureState, ...]] = {}
    classes = (
        "confirmed_bos",
        "wick_only_no_close",
        "broken_or_opposed",
    )
    for offset in (0, 6):
        ordinal = offset
        for direction in (Direction.LONG, Direction.SHORT):
            for case_class in classes:
                asof = frame.index[ordinal] + pd.Timedelta(minutes=1)
                schedule[ordinal] = (
                    _bos(
                        asof=asof,
                        direction=direction,
                        case_class=case_class,
                        suffix=(
                            f"{asof.year}-{direction.value}-{case_class}"
                        ),
                    ),
                )
                ordinal += 1
    runner = _build_fixture_runner(
        tmp_path,
        output_name="two-year",
        frame=frame,
        schedule=schedule,
        calendar_years=(2022, 2023),
    )
    selection = json.loads(
        runner.run_pass1().read_text(encoding="utf-8")
    )
    assert selection["expected_bucket_count"] == 12
    tampered = json.loads(json.dumps(selection))
    for index in range(0, len(tampered["cases"]), 2):
        first = tampered["cases"][index]
        second = tampered["cases"][index + 1]
        first["calendar_year"], second["calendar_year"] = (
            second["calendar_year"],
            first["calendar_year"],
        )
        tampered["cases"][index], tampered["cases"][index + 1] = (
            second,
            first,
        )
    with pytest.raises(
        ValueError,
        match="calendar year differs",
    ):
        runner._validate_selection_authority(tampered)
