# Entry model — the ten-window benchmark (2022)

*2026-09-20 / 21. Ten windows across regimes
(`brain/configs/benchmark_windows.json`, market time, the Eye warmed seven
days before each), run by `brain/scripts/run_benchmark.py` with DeepSeek
`deepseek-flash@high`, `--broker sim`, `--max-llm-calls 400`, three at a
time. First pass under the label `entry-model` (the code of the frozen-window
receipt [2026-09-20_entry_model_frozen_window_2022-01-03.md](2026-09-20_entry_model_frozen_window_2022-01-03.md));
second pass under `entry-model-pools` (spec §1.7) cut short by the API; third pass under `entry-model-pools-2`, complete.*

## 1. First pass — every window, the entry statistics

| window (ET) | regime | calls / USD | direction acc. (n) | ACTIONABLE acc. (n) | incoherent | submitted / filled / expired | median wait | location 60-bar / 240-bar | chased | right 60m | MFE / MAE (R) | exits | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-24 12:00–16:00 | extreme reversal | 56 / 1.49 | 0.25 (36) | 0.33 (15) | 1 | 3 / 3 / 0 | 10 | 0.42 / 0.22 | 0 | 0 of 3 | 0.35 / 5.23 | 3 stops | −171.0 |
| 02-24 08:30–12:30 | V reversal | 54 / 1.25 | 0.11 (44) | 0.08 (25) | 3 | 4 / 3 / 0 | 11 | 0.00 / 0.00 | 0 | 1 of 3 | 1.70 / 3.90 | 2 stops, 1 open | −100.25 |
| 05-10 09:30–13:00 | two-sided chop | 51 / 1.32 | 0.19 (31) | 0.08 (13) | 0 | 3 / 3 / 0 | 1 | 0.03 / 0.15 | 0 | 0 of 3 | 2.02 / 2.22 | target, stop, invalidation | −16.25 |
| 06-27 09:30–13:00 | low conviction | 46 / 1.18 | 0.37 (35) | 0.31 (13) | 0 | 4 / 3 / 0 | 2 | 0.28 / 0.41 | 0 | 0 of 3 | 0.17 / 1.39 | 2 stops, invalidation | −116.5 |
| 07-27 13:30–16:00 | FOMC, strong long | 35 / 0.84 | **0.96** (24) | 0.91 (11) | 0 | 5 / 2 / 0 | 7.5 | 0.59 / 0.56 | 0 | 2 of 2 | 8.30 / 1.69 | stop, **target** | **+105.75** |
| 08-26 09:45–13:00 | strong short trend | 52 / 1.28 | **1.00** (29) | 1.00 (6) | 2 | 4 / 1 / 0 | 2 | 0.73 / 0.73 | 0 | 1 of 1 | 15.5 / 9.65 | stop | −45.0 |
| 09-13 08:15–11:30 | CPI, one-way down | 36 / 1.06 | **0.72** (29) | 0.80 (5) | 2 | 1 / 0 / 0 | — | — | 0 | — | — | — | 0 |
| 10-13 08:15–12:30 | CPI, huge reversal | 55 / 1.39 | 0.12 (41) | 0.30 (10) | 3 | 1 / 1 / 0 | 1 | 0.78 / 0.83 | 1 | 0 of 1 | 2.52 / 6.60 | stop | −43.5 |
| 10-19 09:30–12:30 | mixed, unclean | 42 / 1.13 | 0.32 (28) | 0.54 (13) | 1 | 3 / 2 / 0 | 1 | 0.00 / 0.00 | 0 | 0 of 2 | 4.15 / 12.3 | 2 stops | −74.25 |
| 11-10 08:15–11:30 | CPI, strong long | 36 / 0.87 | **0.73** (22) | 0.57 (7) | 3 | 2 / 2 / 0 | 7.5 | 0.39 / 0.49 | 1 | 0 of 2 | −13.6 / 30.0 | 2 stops | −81.75 |
| **total** | | 463 / 11.79 | | | 15 | 30 / 20 / 0 | | | 2 | 4 of 20 | | | **−542.75** |

Points are the account's (`realized_points`, one or two contracts by the
risk budget); "location" is the fill's place in the range of the 60 and
240 one-minute bars before it, 0 the best price of the window for the
trade, 1 the worst; "chased" is a 240-bar location at or past 0.8; sharp-move
coverage was 9–15 of 12–17 per window (all RTH moves covered).

## 2. What the ten windows say

**The entry model holds across regimes; the direction does not.**

- **No chase, anywhere.** 2 of 20 fills sat past 0.8 of the four-hour
  range; the median fill sat at 0.0–0.6 of the hour's range. On the
  reversal and chop days the shorts were sold at the top of their hour
  (02-24, 10-19: location 0.00) and the longs bought at the bottom (05-10:
  0.03) — the location was right and the side was wrong. Every fill waited
  for the tape (median 1–11 minutes; the nearest 5m zone is usually within
  one ATR, so the wait is short), and no limit expired: on a fast day the
  tape sweeps every pullback object it left behind.
- **Direction is the regime.** The bias was right on the four trend days
  (07-27 0.96, 08-26 1.00, 09-13 0.72, 11-10 0.73) and wrong on the three
  reversal days (01-24 0.25, 02-24 0.11, 10-13 0.12) and the chop (0.19,
  0.37, 0.32). On 01-24 and 02-24 the bias never changed inside the window:
  it followed the trend that brought price there and stayed short through
  a 500-point reversal. The receipt of the direction fix said the reading
  is the model's; here it is the model's on every reversal.
- **A right thesis was expressed on one of the four trend days.** 07-27
  filled twice (the first stopped on a 9-point stop, then a LONG through
  `FVG_5m_11` after a 5-minute wait ran to its target, +44.25 a contract).
  On the other three the right thesis was blocked twice over:
  - **by geometry the model could not see** — seven of the fifteen
    `opportunity_incoherent` rejections (08-26 ×2, 09-13 ×2, 11-10 ×3)
    name a target the model could not place: a liquidity pool outside the
    4-ATR `price_relations` window, a sell-side pool price had already
    fallen through (09-13: `SSL_15m_1` at 12836 against a 12494 entry) or a
    buy-side pool it had risen through (11-10: 10817 against 11195). The
    prompt said pools are the draw; the input showed the model 19 of the
    67 pools it could name. Fixed as spec §1.7: every pool is placed
    whatever its distance.
  - **by stops sized by the object, not the day** — 08-26 SHORT with a
    15-point stop at the open (MFE 15.5 R inside the hour, stopped in two
    minutes), 11-10 SHORT with 8.75 (against a 5 % day), 10-19 with 6.5
    (MAE 21 R), 07-27's first LONG with 9 at FOMC (the market then ran 10.7
    R). The invalidation object is a 5m swing or zone a few points away
    while the one-minute ATR is 10–20 points. Not changed here: a floor
    would be a threshold; the structural answer is an invalidation judged
    on the governing scale's volatility, and it is the next design.
- **The CPI inputs run the reasoning past its budget.** `evidence_expired`
  17 on 09-13 and 23 on 11-10: every one follows a `finish_reason=length`
  incident at `max_tokens` 32 768, the pending-evidence bound then expiring
  the oldest items. The bound did its job; the budget is the question.

## 3. The amendment, the 402, and the third pass

Spec §1.7 (every pool placed) was implemented after the first pass, with a
unit test. The second pass (`entry-model-pools`, 2026-09-21 01:17Z) was cut
short about an hour in by **HTTP 402** — the account's balance — and is
recorded only as the three windows that finished and replay (01-24, 02-24,
05-10) plus the useless remainder; it is not read. After the top-up the
frozen window and the ten windows ran a third time under
`entry-model-pools-2`, with no 402. The paired table, first pass against
third, per window:

| window | regime | direction acc. | fills / right / incoherent / points — first pass | fills / right / incoherent / points — third pass | what changed |
| --- | --- | --- | --- | --- | --- |
| 01-24 | extreme reversal | 0.25 / 0.25 | 3 / 0 / 1 / −171.0 | 3 / 1 / 0 / −114.5 | the third fill is a LONG at 15:43, 2.6 R in its favour at the close |
| 02-24 | V reversal | 0.11 / 0.12 | 3 / 1 / 3 / −100.25 | 1 / 0 / 1 / −48.0 | one SHORT at 09:31 with a 16-point stop, out in a minute (MAE 23 R) |
| 05-10 | two-sided chop | 0.19 / 0.19 | 3 / 0 / 0 / −16.25 | 1 / 0 / 0 / open | one SHORT after a 65-minute wait at the hour's high, 6.9 R in its favour, open at the close |
| 06-27 | low conviction | 0.37 / 0.36 | 3 / 0 / 0 / −116.5 | 1 / 0 / 0 / −45.75 | |
| 07-27 | FOMC, strong long | 0.96 / 0.96 | 2 / 2 / 0 / **+105.75** | 1 / 1 / 1 / −43.5 | the LONG at `SSL_15m_10` (14:07) ran 14 R and was stopped for one: a 14.5-point stop at FOMC; three replacements, no second fill |
| 08-26 | strong short trend | 1.00 / 1.00 | 1 / 1 / 2 / −45.0 | 1 / 1 / 1 / **+165.0** | the SHORT reached its target (+55 a contract) — the thesis the first pass could not express |
| 09-13 | CPI, one-way down | 0.72 / 0.79 | 0 / — / 2 / 0 | 1 / 1 / 1 / −18.75 | the right SHORT filled at 10:19, ran 20 R, stopped for one: a 6.25-point stop |
| 10-13 | CPI, huge reversal | 0.12 / 0.14 | 1 / 0 / 3 / −43.5 | 1 / 0 / 0 / **−1055.25** | see below; the run halted at 08:32 (21 % drawdown), coverage 2 of 16 |
| 10-19 | mixed, unclean | 0.32 / 0.45 | 2 / 0 / 1 / −74.25 | 2 / 1 / 1 / −61.5 | |
| 11-10 | CPI, strong long | 0.73 / 0.73 | 2 / 0 / 3 / −81.75 | 2 / 0 / 1 / −105.0 | the SHORT at 08:31 (16.5-point stop against a 5 % day), then the LONG stopped |
| **total** | | | 20 / 4 / 15 / −542.75 | 14 / 5 / 6 / −1327.25 (−272.0 without 10-13) | calls 463 → 408, cost 11.79 → 9.93 USD |

The direction accuracies agree pass to pass (the Eye and the tape are the
same; the model reads them the same way), so the pairs compare the entry.

**What the pool fix did.** `opportunity_incoherent` 15 → 6, and of the six,
four are the side rule doing its work (an entry a few ticks past the market,
e.g. 08-26 10:30 `SHORT entry 12980.5 lies below price 12986.75`), one an
invalidation on the wrong side (09-13 10:20), one a target still on the
wrong side (11-10 09:40). The right thesis reached a fill on three of the
four trend days instead of one: 08-26 to its target, 07-27 and 09-13 to
14 R and 20 R of favourable excursion — and both of those were stopped for
one R first, by a 14.5-point and a 6.25-point stop.

**What 10-13 did.** At 08:30 the Brain's LONG limit at `FVG_15m_5` rested
14.5 points under price with a 14.5-point stop and three contracts (the
budget buys five at that stop; leverage capped it at three). The CPI print
at 08:30 sent the 08:31 bar 350 points lower: the limit filled at 10903.25
on the way down, the bias flipped SHORT on the same bar, the machine asked
for the bias-reversal flatten at once, and the market order filled at the
08:32 open, 10551.5 — 351.75 a contract, −1055.25, a 21 % drawdown, the
halt. The stop at 10888.75 would not have saved it: a stop through a gap
fills where the gap ends, and the simulator's stop-at-its-price rule would
only have hidden the loss. Three things compound here and none of them is
the entry model's location rule: a resting order one minute before a
scheduled release, a stop sized by a 5m object against a day whose
one-minute bar is 350 points, and a position sized to that stop. They are
the same stop-size problem as 07-27 and 09-13 seen from the losing side.

## 4. Verdict

The entry model changed where the system enters, and that change is
stable across the ten regimes and both passes: no chase (2 and 4 of 20 and
14 fills past 0.8 of the four-hour range — and the 08-26 "chase" is a short
sold near the low of a trend that kept going to its target), resting limits
that fill, replacement and expiry costing the thesis nothing, the
code-made incoherence gone, and with every pool placed the right thesis
gets filled on trend days. It did not change how often the system is
right — reversal days are read wrong in both passes — and it exposed that
the invalidation, not the entry, now decides the P&L: 14.5, 6.25, 16, 16.5
and 20.5-point stops against days whose hourly ranges are hundreds of
points, sized to full leverage. The next structural work, in order: the
invalidation's scale and size (and the position size that follows it),
scheduled-release awareness for a resting order, the bias on a reversal,
the reasoning budget on fast days.

## 5. Baselines

`regression_baselines.json` is frozen on the ten third-pass windows and on
the frozen window's third-pass run. Nine replayed at once; 10-13 did not
until the replay learned to stop at the drawdown halt as the runner does
(`replay_journal.py`, 2026-09-21) — it then reproduced its 7 calls, 18
revisions and 15 trade records
([2026-09-20_entry_model_frozen_window_2022-01-03.md](2026-09-20_entry_model_frozen_window_2022-01-03.md) §7).

## 6. Commands of this receipt

```bash
.venv/bin/python -m brain.scripts.run_benchmark --client deepseek --broker sim --reasoning-effort high --label entry-model --max-llm-calls 400 --parallel 3
.venv/bin/python -m brain.scripts.run_benchmark --client deepseek --broker sim --reasoning-effort high --label entry-model-pools --max-llm-calls 400 --parallel 3   # cut by HTTP 402
.venv/bin/python -m brain.scripts.run_benchmark --client deepseek --broker sim --reasoning-effort high --label entry-model-pools-2 --max-llm-calls 400 --parallel 3
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/099c0dd4eb573551 --run-dir outputs/brain_journal/acbf9be98fc8bd2a --run-dir outputs/brain_journal/2ce9c1eb2c8f280c --run-dir outputs/brain_journal/2f633233b6b64e12 --run-dir outputs/brain_journal/ca8fd337267ab7a0 --run-dir outputs/brain_journal/b0d4b54b2aabae5e --run-dir outputs/brain_journal/975de0e52a799e25 --run-dir outputs/brain_journal/59056391614a5d05 --run-dir outputs/brain_journal/3dc846f3cd392edb --run-dir outputs/brain_journal/aee84d47a233ef03 --write
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/2a5ec0fa05655173
```
