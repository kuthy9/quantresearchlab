# Entry model — the frozen window (2022-01-02 18:00 → 2022-01-03 17:00)

*2026-09-20. Run `b8870c30ec04d5df` (label `entry-model`), DeepSeek
`deepseek-flash@high`, `--broker sim`, against run X `ab572b09d59bc4d7`
(the direction fix, all three layers) and run B′ `3cc1bb402bb7b964`.
Design: [specs/2026-09-20-entry-model-design.md](../specs/2026-09-20-entry-model-design.md);
plan: [plans/2026-09-20-entry-model.md](../plans/2026-09-20-entry-model.md).*

## 1. What changed between run X and this run

Geometry: a zone that contains price is entered at its midpoint (LONG) or
far edge, never on the wrong side of price; a range is refused as an entry;
`coherence_error` refuses a limit the market is already past. Feedback:
`prior_state.last_update.rejections`. The wait: `order_ttl_bars` counts
bars of the entry object's scale (75 1m bars for a 5m object, 225 for a
15m one); a replaced entry object gives the thesis its expression back; a
plan whose limit the market is past is held (`entry_marketable`). The
prompt's "Expression" section replaced "the object that `contains_price` —
the order fills now". Nothing else moved: the bias section, rule 4b, the
controller, the risk policy (2.5 % daily stop, RR ≥ 2) and the Eye are run
X's.

## 2. The run against X and B′

| | entry model `b8870c30` | run X `ab572b09` | run B′ `3cc1bb40` |
| --- | --- | --- | --- |
| LLM calls / repairs / incidents / cost (peak USD) | 293 / 49 / 10 / 6.31 | 290 / 30 / 5 / 5.91 | 302 / 37 / 2 / 6.83 |
| episodes / sleeps (idle, `continue_active=false`) | 4 / 3 (2, 1) | 11 / 11 (6, 5) | 14 / 14 (5, 9) |
| sharp-move coverage (all / RTH) | 77 of 81 / 25 of 25 | 77 of 81 / 25 of 25 | 80 of 81 / 25 of 25 |
| ACTIONABLE replies / revisions that kept one | 147 / 18 | 46 / 26 | 76 / 33 |
| `opportunity_incoherent` (of which code's own price choice) | 3 (0) | 8 (8) | 16 (16) |
| bias changes / NEUTRAL revisions | 12 / 16 | 21 / 46 | 14 / 21 |
| `direction_accuracy_60m` (state revisions) | 0.496 | 0.465 | 0.502 |
| `actionable_direction_accuracy_60m` | **0.414** | 0.318 | 0.384 |
| orders submitted / filled / expired / replaced | 10 / 6 / 2 / 2 | 9 / 7 / 1 / 1 | 18 / 10 / 7 / 0 |
| refusals: `thesis_engaged` / `entry_marketable` | 16 / 1 | 4 / — | 0 / — |
| vetoes: `reward_risk` / `daily_stop` | 2 / 2 | 13 / 1 | 13 / 0 |
| `missed_trends` (missed of expired) | 1 of 2 | 1 of 1 | 3 of 7 |
| fills: median wait (min) / max | **4.5 / 80** | 2 / 6 | 1 / 9 |
| fills: median location, 60-bar / 240-bar window (0 best … 1 worst) | 0.46 / 0.51 | 0.11 / 0.49 | 0.42 / 0.46 |
| fills: `chased` (240-bar location ≥ 0.8) | 0 | 0 | 2 |
| fills: right an hour later / median MFE / median MAE (R) | 2 of 6 / 0.40 / 2.84 | 4 of 7 / 1.13 / 1.25 | 7 of 10 / 1.46 / 1.71 |
| closed: stop / target / invalidation / bias_reversed | 6 / 0 / 0 / 0 | 5 / 0 / 1 / 1 | 7 / 1 / 2 / — |
| realized (points / USD) | **−167 / −3 340** | −146.5 / −2 930 | −139.5 / −2 790 |

The fills (entry object, its place when the order went in, the wait, and
what the next hour did in R):

| filled | thesis | side | entry object (offset at submit, 1m ATR) | fill | wait | MFE / MAE 60m | closed | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-02 19:25 | T1 | SHORT | `BSL_15m_2` (+1.1 above) | 16397.25 | 80 min | 0.24 / 3.59 R | 19:31 stop | −8.5 |
| 01-03 01:31 | T3 | SHORT | `FVG_5m_20` (+1.9 above) | 16404.00 | 15 min | 2.87 / −0.43 R | 04:59 stop | −7.5 |
| 01-03 07:45 | T4 | LONG | `FVG_5m_32` (−0.6 below) | 16436.50 | 5 min | 0.57 / 2.09 R | 08:34 stop | −11.75 |
| 01-03 09:09 | T5 | SHORT | `FVG_5m_38` (+0.7 above) | 16387.50 | 4 min | 15.9 / 16.2 R | 09:11 stop | −6.0 |
| 01-03 09:46 | T6 | LONG | `FVG_5m_39` (−0.9 below) | 16426.00 | 1 min | 0.11 / 7.44 R | 09:48 stop | −18.0 |
| 01-03 10:26 | T2 | SHORT | `FVG_5m_40` (contains, midpoint) | 16392.00 | 1 min | 0.17 / 1.64 R | 10:29 stop | −31.75 |

Two contracts each; the sixth stop at 10:29 took the session past 2.5 %
and the daily stop vetoed the rest of the day (11:45 SHORT, 16:05 LONG).

## 3. What the entry model did

- **No fill at the market.** Every fill waited for the tape to come to a
  resting limit (80, 15, 5, 4, 1, 1 minutes; run X: 1–6). Two chases were
  refused: at 19:30 the reducer (`SHORT entry 16397.25 lies below price
  16400.75`) and at 04:12 the machine (`entry_marketable`, `FVG_5m_26`).
  `chased` is 0.
- **The code-made incoherence is gone.** Run X's eight and B′'s sixteen
  `opportunity_incoherent` were all the far edge of a containing zone or a
  range's value price. This run has three: one is the new side rule
  working (19:30); the other two are the model naming an invalidation on
  the wrong side of its entry (09:10 an SSL *below* a SHORT's entry; 14:20
  a 5m swing low *above* a LONG's midpoint entry) — a reading error the
  rejection now reports back to it in `last_update.rejections`.
- **The wait is affordable and supervised.** T2's limit at `BSL_15m_5`
  worked the full 225 bars (20:17 → 00:02) while price sat 5–19 ATR under
  it; T4's at `FVG_5m_31` the full 75 (06:21 → 07:37) with price 3–9 ATR
  above. Both expired, both gave the expression back, and both theses were
  expressed again (T2 filled at 10:26, T4 at 07:45 through the newer
  `FVG_5m_32` — the "follow the leg" replacement, cancelled
  `signature_changed` and refunded). Calls did not rise (293 vs 290) though
  the Brain slept three times instead of eleven: a resting order keeps it
  awake, and the reactions it is called on are the same.
- **The Brain proposes more and code keeps the book.** 147 ACTIONABLE
  replies (46 in run X) because a waiting order is restated call after
  call; 16 `thesis_engaged` refusals are the same thesis re-proposed
  through other objects while its expression was working or open, and
  cost nothing.
- **The afternoon LONG was proposed twice and expressed never**: 14:20
  died to the model's own invalidation choice (above), 16:05 to the daily
  stop. No position was open into the 13:45 breakout (nothing was open
  after 10:29).

## 4. What it did not do

- **The day lost 20.5 points more than run X.** Six stops, no target. The
  location numbers say the entries were not chased (median 0.46 of the
  hour's range, 0.51 of the four hours') and the direction numbers say the
  fills were wrong: 2 of 6 were on the right side an hour later (4 of 7 in
  X), median MFE 0.40 R against a median MAE 2.84 R. Three of the six were
  taken in the first hour of RTH — the 09:09 SHORT with a 6-point stop was
  out in two minutes on the open's 95-point swing both ways, the 09:46
  LONG six minutes before the flush, the 10:26 SHORT at the low of the
  day with a 31.75-point stop — and those three (−55.75 a contract) are
  the day. The bias was SHORT/1H at the low and LONG/15m before the flush;
  the entry model does not read direction, and run X's receipt already
  found the reading wrong more often than right at the segment level on
  this day.
- **A right trade with no way to keep it.** T3 SHORT (01:31) ran 2.87 R
  in its favour within the hour and never touched its target
  (`SSL_15m_2`, five ATR away); it was stopped three and a half hours
  later for −7.5. There is no partial, no trailing, no time stop — trade
  management is not in this build and is the next structural gap after
  direction.
- **The daily stop still decides the afternoon.** 2.5 % of equity is six
  small stops at two contracts; the policy is the risk configuration's
  question, unchanged here.
- **Incidents doubled** (10 vs 5): every one is `finish_reason=length` at
  `max_tokens` 32 768 with the reasoning past 100 000 characters. In the
  same early window (to 00:30) the rate matched run X and B′ (2 truncation
  repairs in 14 calls vs 2 in 16 and 7 in 19), so the prompt's length is
  not the cause; the budget is. `evidence_without_verdict` 74 (36 in X) is
  the fallout: an incident bar leaves its evidence pending.

## 5. Verdict on the frozen window

The old problems did not come back: the direction reading held
(`direction_accuracy_60m` 0.496 vs 0.465, bias changes 12 vs 21, no short
into the 13:45 breakout), the code-made incoherence is gone, no entry was
taken at the market, and a thesis can wait for its pullback and move with
the leg. The entry model changed *where* the system enters, not *which
way*, and on this day the way was wrong three times at the open. The
benchmark ([2026-09-20_entry_model_benchmark_2022.md](2026-09-20_entry_model_benchmark_2022.md))
is the test of whether the entry statistics hold across regimes and
whether a right reading now gets filled.

Replay from the Eye alone: `replay OK: 4 episodes, 293 llm calls, 1371
revisions, 67 trade records reproduced`. This journal was recorded before
spec §1.7 (every pool placed in `price_relations`); under the amended code
its inputs hash differently, so it is evidence, not a regression baseline.

## 6. Commands

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider
.venv/bin/python -u -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high --label entry-model
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/b8870c30ec04d5df --run-dir outputs/brain_journal/ab572b09d59bc4d7 --run-dir outputs/brain_journal/3cc1bb402bb7b964 --write
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/b8870c30ec04d5df
.venv/bin/python -u -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high --label entry-model-pools-2
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/2710e0b78a707e99 --run-dir outputs/brain_journal/b8870c30ec04d5df --run-dir outputs/brain_journal/ab572b09d59bc4d7 --write
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/2710e0b78a707e99
```

## 7. The same window under the pool fix (`2710e0b78a707e99`, label `entry-model-pools-2`, 2026-09-21)

Spec §1.7 (every pool placed in `price_relations`) re-run on the frozen
window after the benchmark's first pass; the second attempt
(`19dd98812311ce3d`) was lost to HTTP 402 and is not read.

| | pools `2710e0b7` | entry model `b8870c30` | run X `ab572b09` |
| --- | --- | --- | --- |
| LLM calls / repairs / incidents / cost (peak USD) | 278 / 30 / 3 / 5.88 | 293 / 49 / 10 / 6.31 | 290 / 30 / 5 / 5.91 |
| input chars, median | 28 190 | 24 721 | 24 126 |
| episodes / sleeps | 5 / 4 | 4 / 3 | 11 / 11 |
| sharp-move coverage (all / RTH) | 77 of 81 / 25 of 25 | 77 of 81 / 25 of 25 | 77 of 81 / 25 of 25 |
| ACTIONABLE replies | 104 | 147 | 46 |
| `opportunity_incoherent` (of which the side rule / a wrong-side invalidation) | 6 (1 / 5) | 3 (1 / 2) | 8 (0 / 0; all code's) |
| bias changes / NEUTRAL revisions | 16 / 34 | 12 / 16 | 21 / 46 |
| `direction_accuracy_60m` / ACTIONABLE accuracy | 0.440 / 0.350 | 0.496 / 0.414 | 0.465 / 0.318 |
| orders submitted / filled / expired / replaced | 9 / 4 / 0 / 5 | 10 / 6 / 2 / 2 | 9 / 7 / 1 / 1 |
| refusals `thesis_engaged` / `entry_marketable` | 59 / 2 | 16 / 1 | 4 / — |
| vetoes `reward_risk` / `daily_stop` | 6 / 2 | 2 / 2 | 13 / 1 |
| fills: median wait / location 60 / 240 / chased / right 60m | 1 min / 0.56 / 0.67 / 1 / 1 of 4 | 4.5 / 0.46 / 0.51 / 0 / 2 of 6 | 2 / 0.11 / 0.49 / 0 / 4 of 7 |
| closed: stop / bias_reversed | 3 / 1 | 6 / 0 | 5 / 1 (+1 invalidation) |
| realized (points / USD) | **−141.5 / −2 830** | −167 / −3 340 | −146.5 / −2 930 |

| filled | thesis | side | entry object (offset at submit) | fill | wait | MFE / MAE 60m | closed | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-02 18:02 | T1 | SHORT | `BSL_15m_1` (+1.2 above) | 16389.50 | 1 min | 1.25 / 0.62 R | 19:31 stop | −16.25 |
| 01-02 22:15 | T1 | LONG | `FVG_15m_8` (contains, midpoint) | 16404.50 | 1 min | 0.82 / 0.48 R | 01:12 stop | −8.25 |
| 01-03 02:27 | T2 | SHORT | `FVG_5m_16` (+0.4 above) | 16390.50 | 2 min | 0.13 / 0.54 R | 05:01 bias_reversed | −21.5 |
| 01-03 06:16 | T3 | LONG | `FVG_5m_26` (contains, midpoint) | 16441.00 | 1 min | 0.23 / 0.12 R | 08:59 stop | −49.5 |

What is the same: no fill at the market (two chases refused), no expiry,
the replacements free (five `signature_changed` cancels, the thesis kept
its expressions), no position into the 13:45 breakout, and the daily stop
at 08:59 vetoing the afternoon LONG (11:45 and 16:05 in the first pass;
here two `daily_stop` vetoes as well). What is different: with 67 pools
placed the input grew 14 % and the model's reading of the day moved —
bias changes 16 (12), NEUTRAL revisions 34 (16), `direction_accuracy_60m`
0.440 (0.496): the day's reading sits in a 0.44–0.50 band from run to run,
and the entry model does not move it. Two behaviours of the model stand
out in this run and are not code's: it named an invalidation *above* a
LONG's entry five times between 21:45 and 22:15 (`stop=16523.25
entry=16404.0`) while each call showed it the previous rejection in
`last_update.rejections`; and it re-proposed an engaged thesis through
other objects 59 times (`thesis_engaged`, journaled and refused at no
cost). The loss is three stops and a bias-reversal exit on the overnight
and pre-market reads — the direction, again.

Replay from the Eye alone: see the receipt's commands; this run is one of
the eleven regression baselines
([regression_baselines.json](regression_baselines.json)).
