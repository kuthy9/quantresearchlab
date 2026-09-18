from __future__ import annotations

from shares.core.timing import NO_TIMINGS, Timings, timed


def test_summary_has_count_total_and_percentiles() -> None:
    timings = Timings()
    for seconds in (0.010, 0.020, 0.030, 0.040):
        timings.record("eye", seconds)
    summary = timings.summary()["eye"]
    assert summary["count"] == 4
    assert abs(summary["total_s"] - 0.1) < 1e-9
    assert summary["max_ms"] == 40.0 and summary["p50_ms"] == 20.0 and summary["p95_ms"] == 40.0
    assert summary["mean_ms"] == 25.0


def test_timed_records_the_block_and_no_timings_ignores_it() -> None:
    timings = Timings()
    with timed(timings, "llm"):
        pass
    assert timings.summary()["llm"]["count"] == 1 and timings.summary()["llm"]["total_s"] >= 0.0
    with timed(NO_TIMINGS, "llm"):
        pass
    NO_TIMINGS.record("eye", 1.0)
    assert NO_TIMINGS.summary() == {}


def test_summary_is_sorted_by_phase_and_empty_when_nothing_recorded() -> None:
    timings = Timings()
    assert timings.summary() == {}
    timings.record("plan", 0.001)
    timings.record("eye", 0.002)
    assert list(timings.summary()) == ["eye", "plan"]
