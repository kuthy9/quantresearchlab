"""Per-phase wall-time accounting for a run.

``Timings.record(phase, seconds)`` accumulates durations; ``summary()``
reports count, total and percentiles per phase.  ``NO_TIMINGS`` is the
default everywhere a component accepts one: it records nothing, so tests and
replay pay nothing.  ``timed`` measures a block with ``perf_counter`` — a
duration, never a clock a component could read as the market's time."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import time
from typing import Any


class Timings:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, phase: str, seconds: float) -> None:
        self._samples.setdefault(phase, []).append(float(seconds))

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for phase in sorted(self._samples):
            samples = sorted(self._samples[phase])
            count = len(samples)
            out[phase] = {
                "count": count,
                "total_s": round(sum(samples), 6),
                "mean_ms": _ms(sum(samples) / count),
                "p50_ms": _ms(_rank(samples, 0.50)),
                "p95_ms": _ms(_rank(samples, 0.95)),
                "max_ms": _ms(samples[-1]),
            }
        return out


class _NoTimings(Timings):
    def record(self, phase: str, seconds: float) -> None:
        return None


NO_TIMINGS: Timings = _NoTimings()


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def _rank(samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    index = max(0, min(len(samples) - 1, int(round(fraction * len(samples) + 0.5)) - 1))
    return samples[index]


@contextmanager
def timed(timings: Timings, phase: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings.record(phase, time.perf_counter() - started)


__all__ = ["NO_TIMINGS", "Timings", "timed"]
