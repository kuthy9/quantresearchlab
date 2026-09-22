"""Summarize one or more Brain runs from their journals.

    .venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/<run_id> [--run-dir …] [--write] [--until <UTC time>]

Everything is read from the journal (``llm_call``, ``state``, ``tick``,
``opportunity``, ``trade``, ``incident``, ``sleep`` records) and from
``run.json``; ``--write`` saves ``summary.json`` beside ``run.json``.  With
several runs the sections print side by side.  Sharp-move coverage needs
the tape the run names (``--no-coverage`` skips it).  ``--until`` bounds
every count to the records at or before that time, so runs of different
lengths compare over a common window (``--write`` is then skipped).

Sections: ``run``, ``llm`` (calls, tokens, cost at DeepSeek's published
rates, latency, what triggered each call), ``controller`` (decisions,
episodes, sleeps, awake bars), ``coverage`` (sharp 15-bar moves and how
many had a call within [-3, +10] minutes), ``brain`` (opportunities
proposed and survived, confidence, rejections), ``risk`` (vetoes, their
repeats after the LLM saw them, re-analyses), ``orders`` (the lifecycle
counts, cancel reasons, bars to fill, closed trades), ``invariants``
(double entries, stale snapshots — all must be zero), ``account`` and
``timings``."""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import pandas as pd

from brain.core.journal import JournalReader
from brain.scripts._run_identity import MARKET_TIMEZONE, ROOT
from contract.market.primitives import Bar
from shares.core.io import iter_completed_bars, load_ohlcv

DEFAULT_PRICING = "brain/configs/llm_pricing.json"
# The audit's rule (brain/docs/evidence/2026-09-16_controller_eye_audit_2022-01-03.md):
# a bar qualifies when the range of it and the next 14 bars exceeds 2.5 × the
# simple 14-bar ATR; a *move* starts at the first qualifying bar more than 15
# minutes after the previous start; it is covered when an LLM call lands
# within [-3, +10] minutes of the move's bar; RTH is 09:00–16:00 New York.
SHARP_MOVE_BARS = 15
SHARP_MOVE_ATR_MULTIPLE = 2.5
ATR_PERIOD = 14
MOVE_SPACING = pd.Timedelta(minutes=15)
COVER_BEFORE = pd.Timedelta(minutes=3)
COVER_AFTER = pd.Timedelta(minutes=10)
RTH_HOURS = range(9, 16)
OPPORTUNITY_KEYS = ("direction", "entry_object_id", "invalidation_object_id", "target_object_id")


def load_pricing(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported llm_pricing schema_version")
    return payload


def cost_usd(usage: Mapping[str, int], rates: Mapping[str, float]) -> float:
    hit = int(usage.get("prompt_cache_hit_tokens", 0))
    miss = int(usage.get("prompt_cache_miss_tokens", 0))
    if hit == 0 and miss == 0:
        miss = int(usage.get("prompt_tokens", 0))
    out = int(usage.get("completion_tokens", 0))
    return (hit * rates["input_cache_hit"] + miss * rates["input_cache_miss"] + out * rates["output"]) / 1_000_000.0


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    rank = lambda f: ordered[max(0, min(len(ordered) - 1, int(round(f * len(ordered) + 0.5)) - 1))]  # noqa: E731
    return {"p50": rank(0.5), "p95": rank(0.95), "max": ordered[-1]}


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _parse(text: str | None) -> Mapping[str, Any] | None:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _opportunity_key(opportunity: Mapping[str, Any] | None) -> tuple | None:
    if not opportunity or opportunity.get("state") in (None, "NONE"):
        return None
    return (opportunity.get("state"),) + tuple(opportunity.get(key) for key in OPPORTUNITY_KEYS)


def _veto_key(veto: Mapping[str, Any]) -> tuple:
    return ("ACTIONABLE",) + tuple(veto.get(key) for key in OPPORTUNITY_KEYS)


# --------------------------------------------------------------- coverage


def sharp_move_coverage(bars: Sequence[Bar], *, call_times: Sequence[pd.Timestamp]) -> dict[str, Any]:
    """Bars that start a sharp move and how many had an LLM call near them."""
    highs = [float(bar.high) for bar in bars]
    lows = [float(bar.low) for bar in bars]
    closes = [float(bar.close) for bar in bars]
    ends = [pd.Timestamp(bar.start).tz_convert("UTC") + pd.Timedelta(minutes=1) for bar in bars]
    true_ranges = [highs[0] - lows[0]] + [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(bars))
    ]
    calls = sorted(pd.Timestamp(item).tz_convert("UTC") for item in call_times)
    sharp = covered = rth_sharp = rth_covered = 0
    last_start: pd.Timestamp | None = None
    for t in range(ATR_PERIOD - 1, len(bars) - SHARP_MOVE_BARS + 1):
        atr = sum(true_ranges[t - ATR_PERIOD + 1: t + 1]) / ATR_PERIOD
        if atr <= 0.0:
            continue
        window_range = max(highs[t: t + SHARP_MOVE_BARS]) - min(lows[t: t + SHARP_MOVE_BARS])
        if window_range <= SHARP_MOVE_ATR_MULTIPLE * atr:
            continue
        if last_start is not None and ends[t] - last_start <= MOVE_SPACING:
            continue
        last_start = ends[t]
        sharp += 1
        lo, hi = ends[t] - COVER_BEFORE, ends[t] + COVER_AFTER
        index = bisect.bisect_left(calls, lo)
        hit = index < len(calls) and calls[index] <= hi
        covered += int(hit)
        if ends[t].tz_convert(MARKET_TIMEZONE).hour in RTH_HOURS:
            rth_sharp += 1
            rth_covered += int(hit)
    return {
        "rule": f"{SHARP_MOVE_BARS}-bar range > {SHARP_MOVE_ATR_MULTIPLE} x ATR({ATR_PERIOD}), moves {MOVE_SPACING.seconds // 60} min apart; covered by a call within [-{COVER_BEFORE.seconds // 60}, +{COVER_AFTER.seconds // 60}] min; RTH 09-16 NY",
        "bars": len(bars), "sharp_moves": sharp, "covered": covered, "rth_sharp_moves": rth_sharp, "rth_covered": rth_covered,
    }


def direction_accuracy(
    readings: Sequence[tuple[pd.Timestamp, str]], bars: Sequence[Bar], *, horizon_minutes: int = 60
) -> dict[str, Any]:
    """How often a stated direction (``LONG`` / ``SHORT`` at a bar's end)
    matched the sign of the close ``horizon_minutes`` later (2026-09-18).
    Readings without both closes, or with an unchanged close, are not
    counted."""
    close_at = {pd.Timestamp(bar.start).tz_convert("UTC") + pd.Timedelta(minutes=1): float(bar.close) for bar in bars}
    counted = agreed = 0
    for when, direction in readings:
        now = pd.Timestamp(when).tz_convert("UTC")
        first, later = close_at.get(now), close_at.get(now + pd.Timedelta(minutes=horizon_minutes))
        if first is None or later is None or later == first:
            continue
        counted += 1
        agreed += int((later > first) == (direction == "LONG"))
    return {"readings": counted, "agreed": agreed, "accuracy": None if not counted else round(agreed / counted, 4)}


def missed_trends(expiries: Sequence[Mapping[str, Any]], bars: Sequence[Bar]) -> dict[str, int]:
    """Expired entries the tape ran away from (2026-09-19): between the
    submission and the expiry, price never touched the limit and travelled
    at least one R (the plan's stop distance) in the thesis direction from
    where it was when the order went in (the first bar's open).  Each
    expiry is ``{direction, limit_price, stop_price, submitted_at,
    expired_at}``."""
    starts = [pd.Timestamp(bar.start).tz_convert("UTC") for bar in bars]
    missed = 0
    for expiry in expiries:
        limit, stop = float(expiry["limit_price"]), float(expiry["stop_price"])
        first = bisect.bisect_left(starts, pd.Timestamp(expiry["submitted_at"]).tz_convert("UTC"))
        last = bisect.bisect_right(starts, pd.Timestamp(expiry["expired_at"]).tz_convert("UTC"))
        window = bars[first:last]
        if not window:
            continue
        origin = float(window[0].open)
        if expiry["direction"] == "LONG":
            touched = any(float(bar.low) <= limit for bar in window)
            travelled = max(float(bar.high) for bar in window) - origin
        else:
            touched = any(float(bar.high) >= limit for bar in window)
            travelled = origin - min(float(bar.low) for bar in window)
        if not touched and travelled >= abs(limit - stop):
            missed += 1
    return {"expired": len(expiries), "missed": missed}


def _location(price: float, window: Sequence[Bar], direction: str) -> float | None:
    """Where ``price`` sits in the window's range, seen from the trade: 0 is
    the best price of the window (its low for a LONG, its high for a SHORT),
    1 its worst or beyond."""
    if not window:
        return None
    high, low = max(float(bar.high) for bar in window), min(float(bar.low) for bar in window)
    if high <= low:
        return None
    raw = (price - low) / (high - low) if direction == "LONG" else (high - price) / (high - low)
    return min(1.0, max(0.0, raw))


def entry_quality(fills: Sequence[Mapping[str, Any]], bars: Sequence[Bar], *, submitted: int) -> dict[str, Any]:
    """The entries as the tape saw them (2026-09-20): each fill's location in
    the range of the 60 and the 240 one-minute bars before its bar (0 = the
    window's best price for the trade, 1 = its worst), its wait from
    submission in minutes, its maximum favourable and adverse excursions
    over the next 60 bars in R (the plan's stop distance), and whether the
    close 60 minutes later was on its side; since 2026-09-21 also the
    median stop distance in points (the distance the gate sized on) and
    the median contracts.  Each fill is ``{direction, fill_price,
    limit_price, stop_price, quantity, submitted_at, filled_at}``;
    ``chased`` counts fills at or past 0.8 of the 240-bar window."""
    starts = [pd.Timestamp(bar.start).tz_convert("UTC") for bar in bars]
    close_at = {start + pd.Timedelta(minutes=1): float(bar.close) for start, bar in zip(starts, bars)}
    waits: list[float] = []
    loc_60: list[float] = []
    loc_240: list[float] = []
    mfe: list[float] = []
    mae: list[float] = []
    risks: list[float] = []
    quantities: list[float] = []
    chased = right = 0
    for fill in fills:
        direction = str(fill["direction"])
        price = float(fill["fill_price"])
        filled_at = pd.Timestamp(fill["filled_at"]).tz_convert("UTC")
        waits.append((filled_at - pd.Timestamp(fill["submitted_at"]).tz_convert("UTC")).total_seconds() / 60.0)
        index = max(0, bisect.bisect_left(starts, filled_at) - 1)  # the bar that ends at the fill's known_at
        for horizon, sink in ((60, loc_60), (240, loc_240)):
            location = _location(price, bars[max(0, index - horizon):index], direction)
            if location is not None:
                sink.append(location)
                if horizon == 240 and location >= 0.8:
                    chased += 1
        risk = abs(float(fill["limit_price"]) - float(fill["stop_price"]))
        risks.append(risk)
        if fill.get("quantity") is not None:
            quantities.append(float(fill["quantity"]))
        after = bars[index + 1:index + 61]
        if after and risk > 0.0:
            high, low = max(float(bar.high) for bar in after), min(float(bar.low) for bar in after)
            mfe.append(((high - price) if direction == "LONG" else (price - low)) / risk)
            mae.append(((price - low) if direction == "LONG" else (high - price)) / risk)
        later = close_at.get(filled_at + pd.Timedelta(minutes=60))
        if later is not None and later != price:
            right += int((later > price) == (direction == "LONG"))
    return {
        "fills": len(fills), "fill_rate": None if not submitted else round(len(fills) / submitted, 4),
        "median_wait_minutes": _median(waits), "median_location_60m": _median(loc_60), "median_location_240m": _median(loc_240),
        "chased": chased, "right_60m": right, "median_mfe_r": _median(mfe), "median_mae_r": _median(mae),
        "median_risk_points": _median(risks), "median_quantity": _median(quantities),
    }


# --------------------------------------------------------------- summary


def summarize(
    run_dir: Path, *, pricing: Mapping[str, Any], bars: Sequence[Bar] | None = None, until: pd.Timestamp | None = None
) -> dict[str, Any]:
    """``until`` bounds every count to records at or before that ``known_at``
    (for comparing runs over a common window); the account and the timings,
    which only exist for the whole run, are then omitted."""
    reader = JournalReader(run_dir)
    run = dict(reader.run())
    episodes = reader.episode_ids()
    limit = None if until is None else pd.Timestamp(until).tz_convert("UTC")
    if limit is not None and bars is not None:
        bars = [bar for bar in bars if pd.Timestamp(bar.start).tz_convert("UTC") + pd.Timedelta(minutes=1) <= limit]

    calls = replies = repairs = 0
    usage: Counter[str] = Counter()
    latencies: list[float] = []
    input_chars: list[int] = []
    calls_by_trigger: Counter[str] = Counter()
    trigger_reasons: Counter[str] = Counter()
    wake_kinds: Counter[str] = Counter()
    unchanged_calls = 0
    confidence: Counter[str] = Counter()
    proposed_by_state: Counter[str] = Counter()
    incidents: Counter[str] = Counter()
    call_times: list[pd.Timestamp] = []
    calls_with_feedback = kept_after_veto = reanalysed_after_veto = 0

    survived_by_state: Counter[str] = Counter()
    distinct_opportunities: set[tuple] = set()
    geometry_errors = 0
    rejections: Counter[str] = Counter()
    understanding_changes = 0
    awake_bars: list[int] = []
    sleeps_by_reason: Counter[str] = Counter()
    active_at_end = 0

    vetoes = reproposals = stale = 0
    veto_signatures: set[tuple[str, str]] = set()
    vetoes_by_code: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    cancel_reasons: Counter[str] = Counter()
    refusals_by_reason: Counter[str] = Counter()
    exit_roles: Counter[str] = Counter()
    bars_to_fill: list[float] = []
    closed_trades: list[dict[str, Any]] = []
    pending_entries: dict[str, dict[str, Any]] = {}  # signature → the submitted entry, for missed_trends
    expiries: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []  # every entry fill with its submission, for entry_quality
    actionable_readings: list[tuple[pd.Timestamp, str]] = []  # the ACTIONABLE replies' direction at their call
    double_entry = positions_over_limit = position_without_fill = 0
    max_positions = int(run.get("max_open_positions", 3) or 3)
    # The stated direction per state revision (the bias since 2026-09-18,
    # the opportunity's direction for runs before it), and the bias changes.
    readings: list[tuple[pd.Timestamp, str]] = []
    bias_changes = neutral_revisions = 0

    for episode_id in episodes:
        records = reader.records(episode_id)
        if limit is not None:
            records = tuple(r for r in records if r.known_at <= limit)
            if not records:
                continue
        revisions = sum(1 for r in records if r.record in ("state", "tick"))
        awake_bars.append(revisions)
        slept = False
        previous_understanding: str | None = None
        previous_bias: str | None = None
        wake_kind_by_id: dict[str, str] = {}
        entry_open = False  # an entry order working (WORKING / PARTIAL)
        filled_seen = False
        submitted_at: pd.Timestamp | None = None
        # Per signature: the direction, the entry price and the filled quantity of every open intent
        # (several positions may be open at once since Risk v2).
        intents: dict[str, dict[str, Any]] = {}
        for record in records:
            payload = record.payload
            if record.record == "llm_call":
                calls += 1
                call_times.append(record.known_at)
                llm_input = payload.get("input") or {}
                trigger = llm_input.get("trigger") or {}
                kind = str(trigger.get("kind", "?"))
                calls_by_trigger[kind] += 1
                input_chars.append(len(json.dumps(llm_input, sort_keys=True, ensure_ascii=False, separators=(",", ":"))))
                kinds_by_id = {item["evidence_id"]: item["kind"] for item in llm_input.get("new_evidence", ())}
                for reason in trigger.get("reasons", ()):
                    trigger_reasons[kinds_by_id.get(reason, "relation_change")] += 1
                    if kind == "WAKE":
                        wake_kinds[kinds_by_id.get(reason, "?")] += 1
                if payload.get("repaired"):
                    repairs += 1
                rejected = payload.get("rejected_reply")
                if rejected and rejected.get("usage"):
                    usage.update({k: int(v) for k, v in rejected["usage"].items() if isinstance(v, int)})
                reply = payload.get("reply")
                parsed = None
                if reply is not None:
                    replies += 1
                    usage.update({k: int(v) for k, v in (reply.get("usage") or {}).items() if isinstance(v, int)})
                    latencies.append(float(reply.get("latency_ms", 0)))
                    parsed = _parse(reply.get("content"))
                prior = llm_input.get("prior_state")
                if parsed is not None:
                    confidence[str(parsed.get("reasoning_confidence"))] += 1
                    proposed = parsed.get("opportunity") or {}
                    proposed_by_state[str(proposed.get("state"))] += 1
                    if proposed.get("state") == "ACTIONABLE" and proposed.get("direction") in ("LONG", "SHORT"):
                        actionable_readings.append((record.known_at, str(proposed["direction"])))
                    if prior is not None:
                        same_opportunity = _opportunity_key(proposed) == _opportunity_key(prior.get("opportunity"))
                        same_watch = [w.get("object_id") for w in parsed.get("watch_next", ())] == [w.get("object_id") for w in prior.get("watch_next", ())]
                        if parsed.get("understanding_holds") is True and same_opportunity and same_watch:
                            unchanged_calls += 1
                if prior is not None:
                    last_veto = (prior.get("execution") or {}).get("last_veto")
                    if last_veto:
                        calls_with_feedback += 1
                        if parsed is not None:
                            if _opportunity_key(parsed.get("opportunity")) == _veto_key(last_veto):
                                kept_after_veto += 1
                            else:
                                reanalysed_after_veto += 1
            elif record.record == "incident":
                incidents[str(payload.get("kind"))] += 1
            elif record.record == "state":
                state = payload.get("state") or {}
                for rejection in payload.get("rejections", ()):
                    rejections[str(rejection).split(":", 1)[0]] += 1
                understanding = state.get("market_understanding")
                if previous_understanding is not None and understanding != previous_understanding:
                    understanding_changes += 1
                previous_understanding = understanding
                bias = state.get("bias") or {}
                bias_direction = bias.get("direction")
                if bias_direction is not None:
                    if previous_bias is not None and bias_direction != previous_bias:
                        bias_changes += 1
                    previous_bias = bias_direction
                    if bias_direction == "NEUTRAL":
                        neutral_revisions += 1
                stated = bias_direction if bias_direction in ("LONG", "SHORT") else None
                if stated is None and bias_direction is None:
                    opportunity = state.get("opportunity") or {}
                    if opportunity.get("state") in ("DEVELOPING", "ACTIONABLE"):
                        stated = opportunity.get("direction")
                if stated in ("LONG", "SHORT"):
                    readings.append((record.known_at, stated))
            elif record.record == "opportunity":
                opportunity = payload.get("opportunity") or {}
                survived_by_state[str(opportunity.get("state"))] += 1
                distinct_opportunities.add((episode_id,) + tuple(opportunity.get(key) for key in OPPORTUNITY_KEYS))
                if "error" in (payload.get("geometry") or {}):
                    geometry_errors += 1
            elif record.record == "sleep":
                slept = True
                sleeps_by_reason[str(payload.get("reason"))] += 1
            elif record.record == "trade":
                kind = str(payload.get("kind"))
                order_counts[kind] += 1
                known_at = record.known_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                if kind == "veto":
                    vetoes += 1
                    key = (episode_id, str(payload.get("signature")))
                    if key in veto_signatures:
                        reproposals += 1
                    veto_signatures.add(key)
                    for code in (payload.get("verdict") or {}).get("vetoes", ()):
                        vetoes_by_code[str(code)] += 1
                    if payload.get("account_asof") not in (None, known_at):
                        stale += 1
                elif kind == "submitted":
                    if entry_open:
                        double_entry += 1
                    if int(payload.get("open_positions", 0) or 0) >= max_positions:
                        positions_over_limit += 1
                    entry_open, filled_seen = True, False
                    submitted_at = record.known_at
                    intents[str(payload.get("signature"))] = {"direction": str((payload.get("plan") or {}).get("direction")), "entry_price": None, "quantity": 0}
                    verdict = payload.get("verdict") or {}
                    if verdict.get("limit_price") is not None and verdict.get("stop_price") is not None:
                        pending_entries[str(payload.get("signature"))] = {
                            "direction": str((payload.get("plan") or {}).get("direction")), "limit_price": float(verdict["limit_price"]),
                            "stop_price": float(verdict["stop_price"]), "quantity": int(verdict.get("quantity") or 0), "submitted_at": record.known_at,
                        }
                    if (payload.get("account") or {}).get("asof") not in (None, known_at):
                        stale += 1
                elif kind == "cancel_requested":
                    cancel_reasons[str(payload.get("reason"))] += 1
                elif kind == "thesis_refused":
                    refusals_by_reason[str(payload.get("reason"))] += 1
                elif kind in ("partial", "filled"):
                    fill = payload.get("fill") or {}
                    order = payload.get("order") or {}
                    if not filled_seen and submitted_at is not None:
                        bars_to_fill.append((record.known_at - submitted_at).total_seconds() / 60.0)
                    filled_seen = True
                    intent = intents.setdefault(str(payload.get("signature")), {"direction": None, "entry_price": None, "quantity": 0})
                    intent["entry_price"] = order.get("average_fill_price", fill.get("price"))
                    intent["quantity"] = int(order.get("filled_quantity", 0) or 0)
                    if kind == "filled":
                        entry_open = False
                        pending = pending_entries.get(str(payload.get("signature")))
                        if pending is not None and fill.get("price") is not None:
                            fills.append({**pending, "fill_price": float(fill["price"]), "filled_at": record.known_at})
                elif kind == "position_opened":
                    if not filled_seen:
                        position_without_fill += 1
                elif kind in ("cancelled", "expired", "rejected"):
                    order = payload.get("order") or {}
                    if (payload.get("order") or {}).get("role") == "entry":
                        entry_open = False
                        if int(order.get("filled_quantity", 0) or 0) == 0:
                            intents.pop(str(payload.get("signature")), None)
                            pending = pending_entries.pop(str(payload.get("signature")), None)
                            if kind == "expired" and pending is not None:
                                expiries.append({**pending, "expired_at": record.known_at})
                        else:
                            pending_entries.pop(str(payload.get("signature")), None)
                elif kind == "position_closed":
                    role = str(payload.get("exit_role"))
                    exit_roles[role] += 1
                    exit_price = payload.get("exit_price")
                    intent = intents.pop(str(payload.get("signature")), {"direction": None, "entry_price": None, "quantity": 0})
                    pnl = None
                    if intent["entry_price"] is not None and exit_price is not None and intent["direction"] in ("LONG", "SHORT"):
                        sign = 1.0 if intent["direction"] == "LONG" else -1.0
                        pnl = sign * (float(exit_price) - float(intent["entry_price"])) * intent["quantity"]
                    closed_trades.append({
                        "episode_id": episode_id, "direction": intent["direction"], "quantity": intent["quantity"], "entry_price": intent["entry_price"],
                        "exit_price": exit_price, "exit_role": role, "closed_at": known_at, "pnl_points": pnl,
                        "thesis_id": payload.get("thesis_id"),
                    })
        if not slept:
            active_at_end += 1

    model = str(run.get("model", ""))
    model_id = model.split(":", 1)[-1].split("@", 1)[0]
    rates = (pricing.get("per_million_tokens") or {}).get(model_id)
    point_value = float(run.get("point_value", 20.0))
    account = None if limit is not None else run.get("simulated_account")
    sim_equity = run.get("sim_equity")
    positions = list((account or {}).get("positions", ()))
    unrealized = None
    if bars and positions:
        last_close = float(bars[-1].close)
        unrealized = sum((last_close - float(p["average_price"])) * int(p["quantity"]) * point_value for p in positions)
    total_awake = sum(awake_bars)
    summary = {
        "run": {
            "run_id": run.get("run_id"), "window": run.get("window"), "model": model, "reasoning_effort": run.get("reasoning_effort"),
            "broker": run.get("broker"), "client": run.get("client"), "started_at": run.get("started_at"), "finished_at": run.get("finished_at"),
            "minutes": run.get("minutes"), "bars_emitted": run.get("bars_emitted"), "max_llm_calls": run.get("max_llm_calls"),
            "system_prompt_sha256": run.get("system_prompt_sha256"), "sleep_controller_sha256": run.get("sleep_controller_sha256"),
            "main_brain_config_sha256": run.get("main_brain_config_sha256"), "risk_config_sha256": run.get("risk_config_sha256"),
            "simulator_config_sha256": run.get("simulator_config_sha256"), "git_revision": run.get("git_revision"),
            "until": None if limit is None else limit.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "llm": {
            "calls": calls, "replies": replies, "incidents": sum(incidents.values()), "incidents_by_kind": dict(sorted(incidents.items())),
            "repairs": repairs, "prompt_tokens": usage.get("prompt_tokens", 0), "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
            "cache_miss_tokens": usage.get("prompt_cache_miss_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
            "cost_usd_peak": None if rates is None else round(cost_usd(usage, rates["peak"]), 6),
            "cost_usd_off_peak": None if rates is None else round(cost_usd(usage, rates["off_peak"]), 6),
            "latency_ms": _percentiles(latencies), "input_chars_median": _median(input_chars),
            "calls_by_trigger": dict(sorted(calls_by_trigger.items())), "trigger_reasons": dict(sorted(trigger_reasons.items())),
            "wake_kinds": dict(sorted(wake_kinds.items())), "unchanged_calls": unchanged_calls, "confidence": dict(sorted(confidence.items())),
        },
        "controller": {
            "decisions": run.get("decisions"), "episodes": len(episodes), "sleeps_by_reason": dict(sorted(sleeps_by_reason.items())),
            "active_at_end": active_at_end, "awake_bars_total": total_awake, "awake_bars_median": _median(awake_bars),
            "calls_per_awake_bar": None if not total_awake else round(calls / total_awake, 3),
        },
        "coverage": None if bars is None else sharp_move_coverage(bars, call_times=call_times),
        "brain": {
            "proposed_by_state": dict(sorted(proposed_by_state.items())), "survived_by_state": dict(sorted(survived_by_state.items())),
            "distinct_opportunities": len(distinct_opportunities), "geometry_errors": geometry_errors,
            "rejections_by_kind": dict(sorted(rejections.items())), "understanding_changes": understanding_changes,
            "actionable_direction_accuracy_60m": None if bars is None else direction_accuracy(actionable_readings, bars),
        },
        "risk": {
            "vetoes": vetoes, "veto_signatures": len(veto_signatures), "reproposals_after_veto": reproposals,
            "veto_repeat_rate": None if not vetoes else round(reproposals / vetoes, 4), "vetoes_by_code": dict(sorted(vetoes_by_code.items())),
            "calls_with_veto_feedback": calls_with_feedback, "kept_after_veto": kept_after_veto, "reanalysed_after_veto": reanalysed_after_veto,
            "veto_bars": (run.get("machine_stats") or {}).get("veto_bars"), "stale_snapshots": stale,
            "thesis_refused": sum(refusals_by_reason.values()), "refusals_by_reason": dict(sorted(refusals_by_reason.items())),
            "daily_stop_vetoes": vetoes_by_code.get("daily_stop", 0), "halted": run.get("halted"),
        },
        "orders": {
            **{kind: order_counts.get(kind, 0) for kind in ("submitted", "working", "partial", "filled", "cancel_requested", "cancelled", "expired", "rejected", "position_opened", "position_closed", "exit_leg_lost", "invalidation_close", "bias_reversed", "event_sleep", "flattened", "halted")},
            "missed_trends": None if bars is None else missed_trends(expiries, bars),
            "entry_quality": None if bars is None else entry_quality(fills, bars, submitted=order_counts.get("submitted", 0)),
            "cancel_reasons": dict(sorted(cancel_reasons.items())), "replacements": cancel_reasons.get("signature_changed", 0),
            "exit_roles": dict(sorted(exit_roles.items())), "bars_to_fill": {"median": _median(bars_to_fill), "max": max(bars_to_fill) if bars_to_fill else None},
            "closed_trades": closed_trades,
            "realized_points": None if not closed_trades or any(t["pnl_points"] is None for t in closed_trades) else round(sum(t["pnl_points"] for t in closed_trades), 2),
        },
        "bias": {
            "changes": bias_changes, "neutral_revisions": neutral_revisions,
            "opportunities_against_bias": rejections.get("opportunity_against_bias", 0) + rejections.get("opportunity_scale_above_bias", 0),
            "direction_accuracy_60m": None if bars is None else direction_accuracy(readings, bars),
        },
        "invariants": {"double_entry": double_entry, "positions_over_limit": positions_over_limit, "position_without_fill": position_without_fill, "stale_snapshots": stale},
        "account": None if account is None else {
            "cash": account.get("cash"), "realized_pnl": None if sim_equity is None else round(float(account["cash"]) - float(sim_equity), 2),
            "positions": positions, "unrealized_pnl": unrealized, "orders": account.get("orders"), "fills": account.get("fills"),
        },
        "timings": None if limit is not None else run.get("timings"),
    }
    return summary


# --------------------------------------------------------------- rendering


def _flatten(summary: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value and name != "account.orders" and not key.endswith(("_by_kind", "_by_state", "_by_trigger", "_by_code", "_by_reason", "_reasons", "_kinds", "confidence", "exit_roles")):
            rows.update(_flatten(value, f"{name}."))
        else:
            rows[name] = value
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1000 else f"{value:,.2f}"
    if isinstance(value, (list, tuple, Mapping)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) <= 60 else text[:57] + "..."
    return str(value)


def render(summaries: Sequence[Mapping[str, Any]]) -> str:
    flattened = [_flatten(dict(summary)) for summary in summaries]
    keys: list[str] = []
    for rows in flattened:
        for key in rows:
            if key not in keys and not key.startswith("run.window") and key not in ("orders.closed_trades", "account.positions", "timings", "coverage.rule"):
                keys.append(key)
    headers = [str((summary.get("run") or {}).get("run_id") or f"run {i + 1}")[:16] for i, summary in enumerate(summaries)]
    width = max(len(key) for key in keys) if keys else 10
    lines = [f"{'metric':<{width}}  " + "  ".join(f"{h:>18}" for h in headers)]
    for key in keys:
        lines.append(f"{key:<{width}}  " + "  ".join(f"{_cell(rows.get(key)):>18}" for rows in flattened))
    return "\n".join(lines)


def load_bars_for(run: Mapping[str, Any]) -> list[Bar] | None:
    window = run.get("window")
    if not window:
        return None
    source = Path(window["source"])
    source = source if source.is_absolute() else ROOT / source
    if not source.exists():
        return None
    start = pd.Timestamp(window["emit_start"]) - pd.Timedelta(hours=1)
    loaded = load_ohlcv(source, start=start.strftime("%Y-%m-%d %H:%M"), end=window["end"])
    return list(iter_completed_bars(loaded.frame))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--pricing", default=DEFAULT_PRICING)
    parser.add_argument("--write", action="store_true", help="save summary.json beside run.json")
    parser.add_argument("--no-coverage", action="store_true", help="skip the sharp-move coverage (no tape read)")
    parser.add_argument("--until", default=None, help="count only records at or before this UTC time (a common window across runs)")
    args = parser.parse_args(argv)
    until = None if args.until is None else pd.Timestamp(args.until, tz="UTC")
    pricing = load_pricing(ROOT / args.pricing if not Path(args.pricing).is_absolute() else Path(args.pricing))
    summaries = []
    for item in args.run_dir:
        run_dir = Path(item)
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        bars = None if args.no_coverage else load_bars_for(JournalReader(run_dir).run())
        summary = summarize(run_dir, pricing=pricing, bars=bars, until=until)
        if args.write and until is None:
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        summaries.append(summary)
    print(render(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
