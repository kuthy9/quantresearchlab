# The stop floor and the event sleep over the ten 2022 windows

*2026-09-21. Branch `stop-floor-events`; spec
[2026-09-21-stop-floor-event-sleep-design.md](../specs/2026-09-21-stop-floor-event-sleep-design.md).
Label `stop-floor-events`, DeepSeek `deepseek-flash@high`, `--broker sim`,
`--max-llm-calls 400`, three windows at a time. Compared with the entry
model's third pass (label `entry-model-pools-2`,
[2026-09-20_entry_model_benchmark_2022.md](2026-09-20_entry_model_benchmark_2022.md) §3).*

## 0. What changed between the passes

Two rules and nothing else (the Eye, the bias reading and the entry model
are the third pass's): the hard stop is never nearer the entry than one
bar of the thesis's governing scale (1m ATR × √minutes; the gate sizes on
that distance), and the Brain sleeps through CPI, NFP and FOMC statements
— flat, every expression withdrawn — and wakes fresh on the first bar
after the window. The prompt tells the model the floor and that the target
must lie at least two governing bars away. Ten of ten windows finished;
9.62 USD.

| window | third pass | this pass |
| --- | --- | --- |
| 2022-01-24 12:00–16:00 extreme reversal | `4496569b4eca5709` | `d372d3c2f239cdda` |
| 2022-02-24 08:30–12:30 V reversal | `14afbbb0c1caaadf` | `eae1ed05146d3fa3` |
| 2022-05-10 09:30–13:00 two-sided chop | `69aa3b612fd33c99` | `d0304a4ab8f4b8f9` |
| 2022-06-27 09:30–13:00 low conviction | `156861c9f767ed41` | `901c12d4f088f9bf` |
| 2022-07-27 13:30–16:00 FOMC strong long | `613f5d9ee00a96d3` | `3e92f4ac87242abc` |
| 2022-08-26 09:45–13:00 strong short trend | `e7c4c845e13913c0` | `902762c7c41f6a67` |
| 2022-09-13 08:15–11:30 CPI one-way down | `53d8562d97822804` | `fa587b3d45ca81de` |
| 2022-10-13 08:15–12:30 CPI huge reversal | `d1cce1796256ee9e` (halted) | `3bc4b4181f7b48ed` |
| 2022-10-19 09:30–12:30 mixed trend | `f600fa660070fa2a` | `9c18fc7e40fb4520` |
| 2022-11-10 08:15–11:30 CPI strong long | `375ed016339ffe3c` | `b858c9529dd3e4c1` |

## 1. The paired table

`calls / USD`, the bias direction accuracy over the next 60 minutes
(readings), the ACTIONABLE accuracy (readings), rejections
`opportunity_incoherent` / gate vetoes `reward_risk` / `position_size`,
orders submitted / filled / expired, the fills' median stop distance in
points (in 1m ATRs at submission), contracts, wait, location in the
60-bar range, chased, right an hour later, MFE / MAE in R, exits, and
points (`scratchpad/compare_stop_floor.py`).

**Third pass (entry model, every pool placed)**

| window | calls / USD | dir acc (n) | ACT acc (n) | incoh / rr / size | sub / fill / exp | stop pts (ATR×) | qty | wait | loc60 | chased | right | MFE / MAE | exits | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-24 | 54 / 1.14 | 0.25 (36) | 0.32 (25) | 0 / 0 / 1 | 3 / 3 / 0 | 63.8 (1.70) | 1 | 13 | 0.42 | 1 | 1 | 2.57 / 0.76 | stop ×2 | −114.50 |
| 02-24 | 52 / 1.43 | 0.12 (43) | 0.17 (12) | 1 / 6 / 8 | 1 / 1 / 0 | 16.0 (1.03) | 3 | 1 | 0.37 | 0 | 0 | 0.25 / 23.38 | stop | −48.00 |
| 05-10 | 49 / 1.15 | 0.19 (32) | 0.23 (26) | 0 / 7 / 5 | 1 / 1 / 0 | 17.8 (0.69) | 3 | 65 | 0.00 | 0 | 0 | 6.92 / 0.04 | open | — |
| 06-27 | 42 / 1.11 | 0.36 (33) | 0.32 (22) | 0 / 4 / 4 | 2 / 1 / 0 | 45.8 (3.69) | 1 | 10 | 0.65 | 0 | 0 | 1.01 / 0.57 | stop | −45.75 |
| 07-27 | 36 / 0.92 | 0.96 (25) | 0.93 (15) | 1 / 3 / 0 | 4 / 1 / 0 | 14.5 (0.84) | 3 | 2 | 0.05 | 0 | 1 | 14.00 / 1.10 | stop | −43.50 |
| 08-26 | 50 / 1.13 | 1.00 (28) | 1.00 (20) | 1 / 3 / 0 | 4 / 1 / 0 | 20.5 (0.67) | 3 | 1 | 0.92 | 1 | 1 | 9.89 / 0.67 | target | +165.00 |
| 09-13 | 37 / 0.84 | 0.79 (29) | 0.60 (5) | 1 / 3 / 0 | 3 / 1 / 0 | 6.2 (0.33) | 3 | 4 | 0.74 | 1 | 1 | 20.04 / 1.44 | stop | −18.75 |
| 10-13 | 7 / 0.25 | 0.14 (7) | 0.00 (6) | 0 / 1 / 0 | 2 / 1 / 0 | 14.5 (0.81) | 3 | 1 | 0.22 | 0 | 0 | −21.74 / 28.28 | bias_reversed | −1055.25, halted |
| 10-19 | 43 / 0.99 | 0.45 (29) | 0.67 (12) | 1 / 3 / 0 | 2 / 2 / 0 | 42.1 (2.86) | 2 | 1 | 0.00 | 0 | 1 | 1.76 / 3.38 | stop | −61.50 |
| 11-10 | 38 / 0.97 | 0.73 (26) | 0.67 (9) | 1 / 4 / 0 | 3 / 2 / 0 | 17.5 (1.98) | 3 | 10 | 0.36 | 1 | 0 | −6.31 / 16.33 | stop ×2 | −105.00 |
| **total** | 408 / 9.93 | | | 6 / 34 / 18 | 25 / 14 / 0 | 19.5 (1.60) | 3 | | | 4 | 5 | | 9 stops, 1 target, 1 flatten, 3 open | **−1327.25** (−272.00 without 10-13) |

**This pass (stop floor + event sleep)**

| window | calls / USD | dir acc (n) | ACT acc (n) | incoh / rr / size | sub / fill / exp | stop pts (ATR×) | qty | wait | loc60 | chased | right | MFE / MAE | exits | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-24 | 54 / 1.31 | 0.25 (36) | 0.20 (10) | 0 / 0 / 11 | 0 / 0 / 0 | — | — | — | — | 0 | 0 | — | — | — |
| 02-24 | 54 / 1.51 | 0.12 (43) | 0.00 (17) | 0 / 2 / 3 | 1 / 1 / 0 | 32.8 (2.24) | 2 | 7 | 0.00 | 0 | 0 | 2.51 / 6.21 | stop | −65.50 |
| 05-10 | 50 / 1.34 | 0.19 (32) | 0.21 (14) | 0 / 2 / 10 | 1 / 1 / 0 | 50.2 (5.25) | 1 | 1 | 0.77 | 0 | 0 | 1.19 / 3.00 | stop | −50.25 |
| 06-27 | 44 / 1.18 | 0.32 (34) | 0.17 (6) | 1 / 3 / 1 | 3 / 1 / 0 | 46.0 (3.88) | 1 | 1 | 0.28 | 0 | 0 | −0.05 / 3.15 | stop | −46.00 |
| 07-27 | 6 / 0.19 | — (0) | — (0) | 0 / 2 / 0 | 0 / 0 / 0 | — | — | — | — | 0 | 0 | — | — | — |
| 08-26 | 50 / 1.07 | 1.00 (29) | 1.00 (18) | 0 / 2 / 2 | 3 / 0 / 0 | — | — | — | — | 0 | 0 | — | — | — |
| 09-13 | 24 / 0.48 | 1.00 (17) | — (0) | 0 / 0 / 0 | 0 / 0 / 0 | — | — | — | — | 0 | 0 | — | — | — |
| 10-13 | 39 / 0.93 | 0.07 (28) | 0.00 (3) | 1 / 0 / 1 | 1 / 1 / 0 | 67.5 (3.41) | 1 | 5 | 0.32 | 0 | 0 | 0.34 / 5.70 | stop | −67.50 |
| 10-19 | 42 / 0.94 | 0.44 (27) | 0.21 (19) | 0 / 0 / 0 | 3 / 2 / 0 | 55.4 (4.60) | 1 | 3 | 0.35 | 0 | 0 | 1.17 / 1.14 | stop | −57.25 |
| 11-10 | 28 / 0.67 | 0.82 (17) | 1.00 (1) | 0 / 2 / 0 | 1 / 0 / 0 | — | — | — | — | 0 | 0 | — | — | — |
| **total** | 391 / 9.62 | | | 2 / 13 / 28 | 13 / 6 / 0 | 51.9 (3.89) | 1 | | | 0 | 0 | | 6 stops | **−286.50** |

The bias accuracies are the same numbers in both passes on every window
that woke at the same bar (01-24, 02-24, 05-10, 06-27, 08-26): the bias
reading did not change, by construction. 07-27, 09-13, 10-13 and 11-10
wake later (the sleep windows) and read a different stretch of tape.

## 2. The event sleep, as journaled

| window | release | asleep | woke | first Eye wake before | fill on the release bar |
| --- | --- | --- | --- | --- | --- |
| 07-27 | FOMC statement 14:00 | 13:30 (window start) → 15:30 | 15:30 `event_ended:FOMC:2022-07-27T18:00:00Z` | 13:30 | — |
| 09-13 | CPI 08:30 | 08:15 → 09:00 | 09:00 `event_ended:CPI:2022-09-13T12:30:00Z` | 08:15 | — |
| 10-13 | CPI 08:30 | 08:15 → 09:00 | 09:00 `event_ended:CPI:2022-10-13T12:30:00Z` | 08:15 | none (third pass: a LONG limit 14.5 points under price filled on the 350-point bar, −1 055, halt) |
| 11-10 | CPI 08:30 | 08:15 → 09:00 | 09:00 `event_ended:CPI:2022-11-10T12:30:00Z` | 08:15 | none (third pass: a SHORT filled at 08:31, −29.8 R MFE, stopped) |

Every run started inside its window, so no `EVENT_SLEEP` transition and no
`event_sleep` flatten occurred in this pass (the transition itself is
tested on the runtime and the machine — `test_runtime.py`,
`test_order_fsm.py`); the calendar wake fired on the first bar after each
window with no Eye event needed. The 07-27 window keeps 30 minutes of
tape (6 calls): the FOMC sleep ends at 15:30 by the rule the user set, and
the strong-long afternoon happened inside it.

## 3. The stop floor, as journaled

- **No stop inside the noise.** The fills' stop distances are 32.75,
  50.25, 46.0, 67.5, 57.25 and 53.5 points — 2.2 to 5.3 one-minute ATRs,
  median 3.89 (third pass: median 1.60, minimum 0.33). Three of six carry
  `stop.floor.governing_bar`; the other three name objects already beyond
  the floor. Contracts: 1 on five fills, 2 on one (third pass: 3 on nine
  of fourteen).
- **Sizing moved to the account.** `position_size` vetoes rose from 18 to
  28, `reward_risk` vetoes fell from 34 to 13. The size veto is the gate's
  reading of a 100 000 USD account trading NQ (20 USD a point) at 1.5 %: a
  15m floor at a 1m ATR of 25 is 97 points, 1 935 USD a contract, more than
  the 1 500 USD budget. On 01-24 (ATR 33–42, floor 130–160 points) every one
  of 11 ACTIONABLE proposals was vetoed this way and nothing was submitted;
  on 05-10, ten.
- **The model lowered the governing scale to fit the stop.** Third pass:
  governing scales 15m ×84, 1H ×39, 5m ×0 across every proposal; this
  pass: 15m ×84, 5m ×21, 1H ×0. The clearest case is 08-26 (§4). Under a
  1H bias the reducer accepts a 5m thesis (rule 4b refuses only a scale
  *above* the bias), and a 5m floor is 2.24 ATRs.
- **Targets stayed near.** The trend days' proposals were vetoed
  `reward_risk` at 1.12, 1.32, 1.62, 1.68, 1.69: the target was the next 5m
  pool while the stop was a governing bar. The prompt says to name the
  destination on the governing scale; the veto feedback said the ratio; the
  model re-proposed nearer entries and 5m scales instead.
- **Every fill was a stop, all six on reversal or chop days.** Right an
  hour later: 0 of 6 (third pass 5 of 14, all on trend days). The trend
  days produced no fill at all: 08-26 and 11-10 by the size veto and a
  runaway leg, 07-27 and 09-13 by the sleep window and the ratio (§4).

## 4. Window notes

- **08-26 (strong short, bias SHORT@1H, accuracy 1.00).** 10:03: the
  right thesis, SHORT on the 15m, entry FVG_5m_4 at 13 035 (31 points over
  the 13 004 close), stop at the floor 13 131.75 (96.75 points), target
  SSL_1H_1 at 12 823.5, ratio 2.19 — vetoed `position_size` (1 935 USD a
  contract). 10:04: the same three objects with `governing_timeframe` 5m —
  stop at the 5m swing 13 102.5, ratio 3.13 — submitted, one contract;
  10:05: the entry object left the Eye's view, the limit cancelled
  (`plan_dropped`); 10:10–10:20 NEUTRAL. 10:25 the 15m thesis again, size
  veto (2 335 USD); 10:28 the 5m version again, submitted at 13 071.25 with
  the close at 12 968 — the leg had left; 10:31 moved to 13 044.75 (close
  12 933); dropped 11:08 unfilled; the day went to 12 759 by 13:00. The third pass
  had sold this day at 10:29 with a 20.5-point stop and reached the target
  (+165). The floor priced the right thesis correctly and the account could
  not pay for it.
- **07-27 (FOMC).** Asleep until 15:30; then LONG@1H, two proposals at
  15:45 and 16:00 with ratios 1.68 and 1.62 (targets BSL_1H_7 47 and 105
  points away against floors of 28.5 and 65 points); no order.
- **09-13 (CPI, one-way down, bias SHORT, accuracy 1.00).** Woke at 09:00
  into a market already 470 points below its pre-release price (12 923 at
  08:29, 12 456 at 08:59); two DEVELOPING
  proposals (09:32 ratio 0.17 with a 15m CLOSE_BEYOND stop 448 points away;
  10:30 ratio 0.40) and no ACTIONABLE in 24 calls; slept idle at 11:15. The
  third pass sold at 10:19 with a 6.25-point stop and was stopped for 1R
  before a 21R run.
- **10-13 (CPI, huge reversal).** No fill on the release bar (asleep). Woke
  09:00 with the bias SHORT@4H (accuracy 0.07 on a day that rose 440
  points from its 08:30 low to the window's end); one SHORT at 10:31, entry 10 629.75, stop 10 697.25 (67.5
  points, the pool beyond the floor), one contract, stopped at 14:57 for
  −67.5. Third pass: −1 055.25 and a drawdown halt. The direction was as
  wrong as before; the size of being wrong is one governing bar of one
  contract.
- **11-10 (CPI, strong long, bias LONG@1H/4H, accuracy 0.82).** Woke 09:00;
  ratios 1.32 and 1.69 vetoed at 10:30–10:31; 11:05 a LONG at 11 365.25
  (floor stop 51.25, ratio 2.02) submitted with the close at 11 427 — 62
  points below a market that did not come back; unfilled at the window's
  end. The third pass had sold the release bar (−29.8 R MFE) and bought
  at 10:31 with an 18.5-point stop, both stopped.
- **01-24, 02-24, 05-10, 06-27, 10-19 (reversals and chop).** Bias
  accuracies 0.12–0.44 as before. Six fills, six stops of 46–67.5 points
  at one or two contracts (−286.5 in all); the third pass lost −114.5,
  −48, 0, −45.75, −61.5 on the same days with three contracts on 16–20
  point stops. Being wrong now costs one governing bar once, not three
  contracts of noise several times.

## 5. Verdict

Both rules did what the spec says, on every window: no stop inside two
one-minute ATRs (median 3.89, one bar of the 15m being 3.9), contracts
sized by that distance, no order resting
through a release, the calendar wake on the first bar after each window,
and the 10-13 class of loss gone (−67.5 for −1 055.25). They did not make
the ten windows profitable: −286.5 against −272.0 for the third pass
without its halted day, with six fills instead of fourteen and none of
them right.

What the pass exposes, in the order it matters:

1. **The instrument is too large for the account at the thesis's scale.**
   At 1.5 % of 100 000 USD, NQ buys one contract only when one governing
   bar is under 75 points — a 15m thesis at a 1m ATR under 19, a 1H thesis
   under 10. The trend days that pay (08-26, 01-24's afternoon) run at ATRs
   of 25–40. This is a configuration decision — a smaller contract (MNQ,
   2 USD a point), a larger account, or a larger fraction — and not a
   strategy change; the receipt records it and changes nothing.
2. **The governing scale is chosen to fit the stop.** 21 proposals on the
   5m, none before. The scale the floor keys on should be the bias's, or
   at most one below it, not the model's to choose per proposal (a
   reducer rule, the mirror of 4b).
3. **Targets are still the next 5m pool.** With a governing-bar stop the
   ratio needs a governing-scale destination; the model names it in
   `destination_candidates` and not in the opportunity.
4. **Direction on reversals is untouched**, as intended for this build.

## 6. Baselines

Re-frozen on this pass's ten windows and the frozen window's run (§7 of
the frozen receipt) once the replays reproduce them; the third-pass
journals were recorded under controller schema 4 and the unfloored
geometry and no longer replay.

## 7. Commands of this receipt

```bash
.venv/bin/python -m brain.scripts.run_benchmark --client deepseek --broker sim --reasoning-effort high --label stop-floor-events --max-llm-calls 400 --parallel 3
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/d372d3c2f239cdda --run-dir outputs/brain_journal/eae1ed05146d3fa3 --run-dir outputs/brain_journal/d0304a4ab8f4b8f9 --run-dir outputs/brain_journal/901c12d4f088f9bf --run-dir outputs/brain_journal/3e92f4ac87242abc --run-dir outputs/brain_journal/902762c7c41f6a67 --run-dir outputs/brain_journal/fa587b3d45ca81de --run-dir outputs/brain_journal/3bc4b4181f7b48ed --run-dir outputs/brain_journal/9c18fc7e40fb4520 --run-dir outputs/brain_journal/b858c9529dd3e4c1
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/902762c7c41f6a67   # and the other nine
```
