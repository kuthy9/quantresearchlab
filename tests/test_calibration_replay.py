from __future__ import annotations

from dataclasses import replace
import pickle

import pandas as pd
import pytest

import smc_trader.calibration_replay as calibration_replay
from smc_trader.calibration_replay import (
    CalibrationSequentialReplay,
    ReplayCheckpointStore,
    iter_after_source_checkpoint,
)
from smc_trader.decision import UtilityDecisionLayer
from smc_trader.engine import ContinuousSMCEngine
from smc_trader.io import iter_completed_bars
from smc_trader.model import Bar, EngineSnapshot, to_primitive
from smc_trader.observation import ExecutionRealityInput
from smc_trader.simulation import SequentialPortfolio, SequentialReplay
from smc_trader.risk import StructuralRiskEngine

from .helpers import flat_account, session_bars
from .test_v4_typed_vertical import _brain, _dfp_fixture


def _execution(asof: pd.Timestamp) -> ExecutionRealityInput:
    return ExecutionRealityInput(
        spread_points=0.0,
        expected_slippage_points=0.0,
        commission_per_contract_per_side=0.0,
        deadline=asof + pd.Timedelta(hours=4),
        source="calibration_equivalence_test",
    )


def _snapshot_payload(snapshot) -> dict:
    return to_primitive(snapshot)


def _on_grid_session_bars(count: int) -> list[Bar]:
    output: list[Bar] = []
    for bar in session_bars(1)[:count]:
        open_price = round(bar.open / 0.25) * 0.25
        close = round(bar.close / 0.25) * 0.25
        output.append(
            replace(
                bar,
                open=open_price,
                high=max(open_price, close) + 0.5,
                low=min(open_price, close) - 0.5,
                close=close,
            )
        )
    return output


def test_lightweight_replay_preserves_causal_decisions_risk_and_execution() -> None:
    bars = _on_grid_session_bars(80)
    standard = SequentialReplay(
        engine=ContinuousSMCEngine.from_config("configs/model.json")
    )
    lightweight = CalibrationSequentialReplay(
        engine=ContinuousSMCEngine.from_config("configs/model.json")
    )
    for bar in bars:
        execution = _execution(bar.end)
        expected = standard.on_bar(bar, execution=execution)
        actual = lightweight.on_bar(bar, execution=execution)
        assert _snapshot_payload(actual.snapshot) == _snapshot_payload(
            expected.snapshot
        )
        assert to_primitive(actual.closed_trades) == to_primitive(
            expected.closed_trades
        )
        assert to_primitive(actual.position) == to_primitive(expected.position)
        assert to_primitive(actual.account_state) == to_primitive(
            expected.account_state
        )
        assert to_primitive(actual.belief_position_input) == to_primitive(
            expected.belief_position_input
        )
    assert lightweight.engine.last_snapshot is actual.snapshot


def test_non_simulating_replay_preserves_flat_causal_model_without_portfolio() -> None:
    bars = _on_grid_session_bars(40)
    expected_engine = ContinuousSMCEngine.from_config("configs/model.json")
    replay = CalibrationSequentialReplay(
        engine=ContinuousSMCEngine.from_config("configs/model.json"),
        simulate_execution=False,
    )
    assert replay.portfolio is None
    for bar in bars:
        execution = _execution(bar.end)
        expected = expected_engine.on_bar(
            bar,
            execution=execution,
            account=flat_account(),
        )
        actual = replay.on_bar(bar, execution=execution)
        assert actual.closed_trades == ()
        assert actual.position is None
        assert _snapshot_payload(actual.snapshot) == _snapshot_payload(
            expected
        )


def test_non_simulating_replay_rejects_portfolio_state() -> None:
    with pytest.raises(ValueError, match="portfolio cannot be supplied"):
        CalibrationSequentialReplay(
            portfolio=SequentialPortfolio(),
            simulate_execution=False,
        )


def test_lightweight_replay_passes_open_position_to_brain() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    brain.update(forming)
    belief = brain.update(triggered)
    decision = UtilityDecisionLayer().decide(triggered, belief)
    risk = StructuralRiskEngine().review(decision, triggered)
    approved = EngineSnapshot(triggered, belief, decision, risk)
    assert approved.risk.passed
    standard_portfolio = SequentialPortfolio()
    lightweight_portfolio = SequentialPortfolio()
    standard_portfolio.after_decision(approved)
    lightweight_portfolio.after_decision(approved)
    standard = SequentialReplay(portfolio=standard_portfolio)
    lightweight = CalibrationSequentialReplay(portfolio=lightweight_portfolio)
    bar = Bar(
        approved.observation.asof,
        101.0,
        101.5,
        99.5,
        100.5,
        100.0,
        "NQH5",
        1,
    )
    execution = _execution(bar.end)
    expected = standard.on_bar(bar, execution=execution)
    actual = lightweight.on_bar(bar, execution=execution)
    assert expected.position is not None
    assert actual.position is not None
    assert expected.belief_position_input is not None
    assert actual.belief_position_input == expected.belief_position_input
    assert _snapshot_payload(actual.snapshot) == _snapshot_payload(
        expected.snapshot
    )


def test_lightweight_replay_pickle_resume_is_bitwise_deterministic() -> None:
    bars = _on_grid_session_bars(80)
    uninterrupted = CalibrationSequentialReplay()
    resumed = CalibrationSequentialReplay()
    for index, bar in enumerate(bars):
        execution = _execution(bar.end)
        expected = uninterrupted.on_bar(bar, execution=execution)
        if index == 39:
            resumed = pickle.loads(
                pickle.dumps(resumed, protocol=pickle.HIGHEST_PROTOCOL)
            )
        actual = resumed.on_bar(bar, execution=execution)
        assert _snapshot_payload(actual.snapshot) == _snapshot_payload(
            expected.snapshot
        )
    assert to_primitive(resumed.portfolio.records) == to_primitive(
        uninterrupted.portfolio.records
    )


def test_resume_iterator_preserves_gap_densification_after_checkpoint() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-06 10:00:00-05:00",
            "2025-01-06 10:02:00-05:00",
            "2025-01-06 10:03:00-05:00",
            "2025-01-06 10:05:00-05:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [100.25, 101.25, 102.25, 103.25],
            "volume": [10.0, 11.0, 12.0, 13.0],
            "symbol": ["NQH5"] * 4,
            "instrument_id": [1] * 4,
        },
        index=index,
    )
    complete = list(iter_completed_bars(frame))
    resumed = list(iter_after_source_checkpoint(frame, index[1]))
    expected = [bar for bar in complete if bar.start > index[1]]
    assert [to_primitive(bar) for bar in resumed] == [
        to_primitive(bar) for bar in expected
    ]
    assert [bar.synthetic_no_trade for bar in resumed] == [False, True, False]


def test_lightweight_replay_resets_belief_on_explicit_data_gap(
    monkeypatch,
) -> None:
    replay = CalibrationSequentialReplay()
    reset = replay.engine.brain.reset
    calls = []

    def recording_reset() -> None:
        calls.append(replay.engine.reader.last_asof)
        reset()

    monkeypatch.setattr(replay.engine.brain, "reset", recording_reset)
    first = Bar(
        pd.Timestamp("2025-01-06 10:00:00", tz="America/New_York"),
        100.0,
        100.5,
        99.5,
        100.25,
        10.0,
        "NQH5",
        1,
    )
    after_gap = Bar(
        pd.Timestamp("2025-01-06 10:10:00", tz="America/New_York"),
        101.0,
        101.5,
        100.5,
        101.25,
        10.0,
        "NQH5",
        1,
        data_gap_before_minutes=9,
    )
    replay.on_bar(first, execution=_execution(first.end))
    replay.on_bar(after_gap, execution=_execution(after_gap.end))
    assert len(calls) == 1


def test_checkpoint_store_binds_state_and_rejects_different_run(
    tmp_path,
    monkeypatch,
) -> None:
    store = ReplayCheckpointStore(tmp_path / "checkpoint")
    state = {
        "replay": CalibrationSequentialReplay(),
        "processed_bars": 12,
        "decision_rows": 7,
        "next_shard_index": 1,
        "committed_shards": [
            {"index": 0, "path": "decision_shards/part-00000.parquet"}
        ],
        "last_source_start": pd.Timestamp(
            "2025-01-06 10:00:00",
            tz="America/New_York",
        ),
    }
    bindings = {"run_manifest": "run_manifest.json"}
    store.save(state, bindings=bindings)
    restored = store.load(expected_bindings=bindings)
    assert restored["processed_bars"] == 12
    assert isinstance(restored["replay"], CalibrationSequentialReplay)

    original_atomic_bytes = calibration_replay._atomic_bytes

    def interrupt_manifest_publish(path, payload):
        if path.name == "manifest.json":
            raise RuntimeError("simulated manifest publication failure")
        original_atomic_bytes(path, payload)

    monkeypatch.setattr(
        calibration_replay,
        "_atomic_bytes",
        interrupt_manifest_publish,
    )
    interrupted = dict(state)
    interrupted["processed_bars"] = 13
    with pytest.raises(RuntimeError, match="publication failure"):
        store.save(interrupted, bindings=bindings)
    monkeypatch.setattr(
        calibration_replay,
        "_atomic_bytes",
        original_atomic_bytes,
    )
    assert store.load(expected_bindings=bindings)["processed_bars"] == 12

    with pytest.raises(ValueError, match="bindings differ"):
        store.load(
            expected_bindings={
                "run_manifest": "different-run-manifest.json",
            }
        )
