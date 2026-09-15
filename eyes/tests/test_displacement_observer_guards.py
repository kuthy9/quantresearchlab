"""The displacement Eye's fail-closed guards.

``CausalDisplacementEye`` is the Eye's isolated projection of the frozen 5m
displacement reducer, and it is the only thing standing between a malformed
reader update and the displacement history.  Its happy path is exercised by the
replay suites; these are the five refusals that keep a bad update from becoming
displacement state, each of which was uncovered before this file existed.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from eyes.core.displacement import DisplacementLifecycle
from eyes.core.displacement_observer import (
    CONTRACT_BOUNDARY,
    DATA_GAP_BOUNDARY,
    REGISTERED_SESSION_BOUNDARY,
    CausalDisplacementEye,
    _boundary_reason,
)
from contract.market import Timeframe

from eyes.tests.test_v3_displacement_replay import _m5, _protocol, _send, _update, _warm_eye


def test_a_non_protocol_is_refused_before_any_state_exists() -> None:
    with pytest.raises(TypeError, match="frozen displacement protocol"):
        CausalDisplacementEye({"tick_size": 0.25})


def test_a_contract_change_outranks_a_data_gap_and_a_session_close() -> None:
    """Boundary precedence decides which epoch the history is censored into.

    Reading a contract change as a session reset would carry the previous
    instrument's episodes into the new contract.
    """

    assert (
        _boundary_reason((CONTRACT_BOUNDARY, DATA_GAP_BOUNDARY))
        == CONTRACT_BOUNDARY
    )
    assert (
        _boundary_reason((DATA_GAP_BOUNDARY, "scheduled_market_closure"))
        == DATA_GAP_BOUNDARY
    )
    assert (
        _boundary_reason(("scheduled_weekend_closure",))
        == REGISTERED_SESSION_BOUNDARY
    )
    assert _boundary_reason(()) is None


def test_an_unregistered_reader_anomaly_is_refused() -> None:
    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    candle = _m5(index)

    with pytest.raises(ValueError, match="unknown reader anomalies"):
        eye.on_update(
            _update(
                candle.end,
                m5=(candle,),
                anomalies=("some_unregistered_anomaly",),
            )
        )


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param({"complete": False}, id="incomplete"),
        pytest.param({"timeframe": Timeframe.M15}, id="wrong-timeframe"),
    ),
)
def test_a_malformed_completed_m5_is_refused(mutation: dict) -> None:
    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    candle = _m5(index)
    broken = replace(candle, **mutation)

    with pytest.raises(ValueError, match="invalid ordered completed M5"):
        eye.on_update(_update(candle.end, m5=(broken,)))


def test_an_m5_candle_after_the_update_clock_is_refused() -> None:
    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    candle = _m5(index)

    with pytest.raises(ValueError, match="invalid ordered completed M5"):
        eye.on_update(
            _update(candle.end - pd.Timedelta(minutes=1), m5=(candle,))
        )


def test_out_of_order_m5_candles_are_refused_as_a_batch() -> None:
    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    first, second = _m5(index), _m5(index + 1)

    with pytest.raises(ValueError, match="invalid ordered completed M5"):
        eye.on_update(_update(second.end, m5=(second, first)))


def test_a_state_clock_after_the_update_clock_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No projected state may carry a clock the reader has not reached.

    The tracker is the authority on its own clocks, so this guard catches a
    future clock arriving from a corrupted or hand-built tracker state rather
    than from ordinary replay.
    """

    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    _send(eye, _m5(index))
    candle = _m5(index + 1)

    snapshot = eye.tracker.snapshot()
    assert snapshot is not None
    monkeypatch.setattr(
        type(eye.tracker),
        "on_completed",
        lambda _self, _candle: _FakeUpdate(
            replace(
                snapshot,
                last_updated_at=candle.end + pd.Timedelta(minutes=5),
            )
        ),
    )

    with pytest.raises(ValueError, match="future clock"):
        eye.on_update(_update(candle.end, m5=(candle,)))


def test_a_terminal_displacement_can_never_be_the_current_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``current_*`` describes a live episode; a terminal one is history."""

    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    _send(eye, _m5(index))
    candle = _m5(index + 1)

    snapshot = eye.tracker.snapshot()
    assert snapshot is not None
    monkeypatch.setattr(
        type(eye.tracker),
        "on_completed",
        lambda _self, _candle: _FakeUpdate(
            replace(snapshot, lifecycle=DisplacementLifecycle.EXHAUSTED)
        ),
    )

    with pytest.raises(ValueError, match="terminal displacement cannot be current"):
        eye.on_update(_update(candle.end, m5=(candle,)))


class _FakeUpdate:
    """Minimal stand-in for one tracker result with no transitions."""

    def __init__(self, state) -> None:
        self.state = state
        self.transitions = ()
