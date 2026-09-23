# The thesis scale, the structural exit and the bias decay over the ten 2022 windows

*2026-09-22. Branch `scale-exit-bias`; spec
[2026-09-22-scale-exit-bias-design.md](../specs/2026-09-22-scale-exit-bias-design.md).
Label `scale-exit-bias-2` (the second pass, on the tree the review fixed),
DeepSeek `deepseek-flash@high`, `--broker sim`, `--max-llm-calls 400`, three
windows at a time. Compared with the stop floor's pass (label
`stop-floor-events`,
[2026-09-21_stop_floor_event_sleep_benchmark_2022.md](2026-09-21_stop_floor_event_sleep_benchmark_2022.md)).
Ten of ten windows finished; 389 calls, 8.68 USD.*

## 0. What changed between the passes

The five rules of the spec and nothing else (the Eye, the entry model, the
stop floor and the event sleep are the previous pass's):

1. **The risk budget** is 2.5 % of equity at BASE (was 1.5 %), 3 % at
   A_PLUS, 5 % a day, 10 % drawdown, two open positions; the gate vetoes
   `ACCOUNT_RISK` when the open risk plus the new trade's exceeds the
   session's remaining budget (schema 4).
2. **The thesis scale is code's**: the reducer sets `governing_timeframe`
   from the bias (`4H → 1H`, `1H → 15m`, `15m → 15m`); the model no longer
   names it.
3. **Invalidation on the thesis scale or one below; target on the thesis
   scale or above** (`opportunity_target_scale`), so a 15m thesis cannot
   aim at the next 5m pool.
4. **The exit is structural**: a bias flip no longer closes a position;
   the machine flattens and withdraws working entries only on an MSS / BOS
   against the thesis on its own scale (`structure_reversed`).
5. **The bias decays**: two structural events against it on its scale or
   the scale below, with no same-direction event between, set the bias
   NEUTRAL (`bias_decayed`); the decayed pair cannot be re-asserted until
   that scale prints an MSS / BOS its way (`bias_reassert_refused`); the
   memory survives sleep and the episode boundary.

| window | stop floor | pass 1 (`scale-exit-bias`) | pass 2 (`scale-exit-bias-2`) |
| --- | --- | --- | --- |
| 2022-01-24 12:00–16:00 extreme reversal | `d372d3c2f239cdda` | `5e5b88d5c65a1bb5` | `100c8acf3adb04a5` |
| 2022-02-24 08:30–12:30 V reversal | `eae1ed05146d3fa3` | `9576e0653d8a8d7d` | `1719aa369eb7c3ca` |
| 2022-05-10 09:30–13:00 two-sided chop | `d0304a4ab8f4b8f9` | `0c6f614437d019f4` | `d6acf4393c276edf` |
| 2022-06-27 09:30–13:00 low conviction | `901c12d4f088f9bf` | `3895e3e169f8759b` | `a10498251ccdfd14` |
| 2022-07-27 13:30–16:00 FOMC strong long | `3e92f4ac87242abc` | `ca7da6fbca85039e` | `507bfd3504e7b492` |
| 2022-08-26 09:45–13:00 strong short trend | `902762c7c41f6a67` | `ad86c57c40e3a50c` | `1e2b53787f575923` |
| 2022-09-13 08:15–11:30 CPI one-way down | `fa587b3d45ca81de` | `fa59079fbace156c` | `a36da2d72fbd82ba` |
| 2022-10-13 08:15–12:30 CPI huge reversal | `3bc4b4181f7b48ed` | `b430ed453af655d2` | `7d563931a3e95e97` |
| 2022-10-19 09:30–12:30 mixed trend | `9c18fc7e40fb4520` | `01205a4b0e5ff382` | `c0ce570bd26493f5` |
| 2022-11-10 08:15–11:30 CPI strong long | `b858c9529dd3e4c1` | `1e1173e4ed0003f1` | `7f7205546aebcc4a` |

Pass 1 ran on the tree before the review: −718.0 with eleven fills, three
right. Its worst two trades (01-24 −116.25 short, 05-10 a fresh long)
were entered by a wake that had forgotten the decay — the memory was
carried only inside an episode. The review confirmed that and seven more
defects (spec §1.9); pass 2 is the fixed tree. Pass 1 is kept as a record
of the defect, not as a baseline.

## 1. The paired table

`calls / USD`, the bias direction accuracy over the next 60 minutes
(readings), the ACTIONABLE accuracy (readings), rejections
`opportunity_incoherent` / gate vetoes `reward_risk` / `position_size` /
`account_risk`, bias decays / re-asserts refused, orders submitted /
filled, the fills' median stop distance in points (in 1m ATRs at
submission), contracts, right an hour later, MFE / MAE in R, exits, and
points on the closed trades (`scratchpad/compare_scale_exit_bias.py`).

**Stop floor (previous pass)**

| window | calls / USD | dir acc (n) | ACT acc (n) | incoh / rr / size / acct | decays / refused | sub / fill | stop pts (ATR×) | qty | right | MFE / MAE | exits | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-24 | 54 / 1.31 | 0.25 (36) | 0.20 (10) | 0 / 0 / 11 / 0 | 0 / 0 | 0 / 0 | — | — | 0 | — | — | — |
| 02-24 | 54 / 1.51 | 0.12 (43) | 0.00 (17) | 0 / 2 / 3 / 0 | 0 / 0 | 1 / 1 | 32.8 (2.24) | 2 | 0 | 2.51 / 6.21 | stop | −65.50 |
| 05-10 | 50 / 1.34 | 0.19 (32) | 0.21 (14) | 0 / 2 / 10 / 0 | 0 / 0 | 1 / 1 | 50.2 (5.25) | 1 | 0 | 1.19 / 3.00 | stop | −50.25 |
| 06-27 | 44 / 1.18 | 0.32 (34) | 0.17 (6) | 1 / 3 / 1 / 0 | 0 / 0 | 3 / 1 | 46.0 (3.88) | 1 | 0 | −0.05 / 3.15 | stop | −46.00 |
| 07-27 | 6 / 0.19 | — (0) | — (0) | 0 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | — | — | 0 | — | — | — |
| 08-26 | 50 / 1.07 | 1.00 (29) | 1.00 (18) | 0 / 2 / 2 / 0 | 0 / 0 | 3 / 0 | — | — | 0 | — | — | — |
| 09-13 | 24 / 0.48 | 1.00 (17) | — (0) | 0 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | — | — | 0 | — | — | — |
| 10-13 | 39 / 0.93 | 0.07 (28) | 0.00 (3) | 1 / 0 / 1 / 0 | 0 / 0 | 1 / 1 | 67.5 (3.41) | 1 | 0 | 0.34 / 5.70 | stop | −67.50 |
| 10-19 | 42 / 0.94 | 0.44 (27) | 0.21 (19) | 0 / 0 / 0 / 0 | 0 / 0 | 3 / 2 | 55.4 (4.60) | 1 | 0 | 1.17 / 1.14 | stop | −57.25 |
| 11-10 | 28 / 0.67 | 0.82 (17) | 1.00 (1) | 0 / 2 / 0 / 0 | 0 / 0 | 1 / 0 | — | — | 0 | — | — | — |
| **total** | 391 / 9.62 | | | 2 / 13 / 28 / 0 | 0 / 0 | 13 / 6 | 51.9 (3.89) | | 0 | | 6 stops | **−286.50** |

**This pass (thesis scale, structural exit, bias decay)**

| window | calls / USD | dir acc (n) | ACT acc (n) | incoh / rr / size / acct | decays / refused | sub / fill | stop pts (ATR×) | qty | right | MFE / MAE | exits | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01-24 | 55 / 1.24 | 0.06 (16) | 0.00 (3) | 1 / 0 / 2 / 0 | 1 / 2 | 0 / 0 | — | — | 0 | — | — | — |
| 02-24 | 52 / 1.32 | 0.45 (40) | 0.33 (18) | 0 / 1 / 2 / 0 | 0 / 0 | 2 / 1 | 117.5 (3.88) | 1 | 1 | 1.77 / 0.40 | open (+95.0) | — |
| 05-10 | 49 / 0.86 | 0.20 (10) | 0.25 (12) | 0 / 0 / 0 / 0 | 2 / 0 | 3 / 2 | 69.5 (4.56) | 2, 1 | 0 | 1.09 / 1.38 | stop, open (+77.0) | −100.50 |
| 06-27 | 44 / 1.07 | 0.04 (26) | 0.00 (5) | 0 / 1 / 2 / 0 | 0 / 0 | 2 / 1 | 42.5 (3.89) | 2 | 0 | 0.67 / 0.92 | stop | −85.00 |
| 07-27 | 6 / 0.14 | — (0) | — (0) | 0 / 1 / 0 / 0 | 0 / 0 | 0 / 0 | — | — | 0 | — | — | — |
| 08-26 | 48 / 1.05 | 1.00 (25) | 1.00 (17) | 0 / 0 / 0 / 0 | 0 / 0 | 3 / 0 | — | — | 0 | — | — | — |
| 09-13 | 24 / 0.44 | 1.00 (11) | — (0) | 0 / 0 / 0 / 0 | 1 / 0 | 0 / 0 | — | — | 0 | — | — | — |
| 10-13 | 41 / 0.95 | 0.14 (21) | 0.67 (3) | 0 / 4 / 4 / 0 | 1 / 0 | 1 / 0 | — | — | 0 | — | — | — |
| 10-19 | 41 / 0.98 | 0.22 (23) | 0.09 (11) | 1 / 0 / 0 / 0 | 0 / 0 | 4 / 3 | 81.2 (4.30) | 3, 1, 1 | 0 | 0.31 / 1.06 | stop, structure_reversed, open (−68.5) | −113.75 |
| 11-10 | 29 / 0.63 | 0.83 (18) | — (0) | 0 / 0 / 0 / 0 | 0 / 0 | 1 / 0 | — | — | 0 | — | — | — |
| **total** | 389 / 8.68 | | | 2 / 7 / 10 / 0 | 5 / 2 | 16 / 7 | 81.2 (4.30) | | 1 | | 3 stops, 1 structural, 3 open | **−299.25** (−137.0 per contract; +103.5 open at the window ends) |

The bias accuracies are no longer comparable across the passes: this pass
sets the bias NEUTRAL on decay (a NEUTRAL reading is not counted), so the
reading counts fall (01-24: 36 → 16) and the remaining readings are the
directional stretches only.

## 2. The five rules, as journaled

1. **Risk budget.** `position_size` vetoes fell from 28 to 10,
   `reward_risk` from 13 to 7, and every one of the 16 submissions was
   sized; no `account_risk` veto fired (never two positions at once in
   these windows). The 10 size vetoes are all stops of 127 to 293 points
   on the reversal days (01-24 133 and 208; 02-24 293 and 127.5; 06-27 148
   and 161; 10-13 171 to 211): 2.5 % of 100 000 USD buys one contract
   under 125 points, as the spec's §3 states. The other side of the same
   fraction: at 34.75 points the gate buys three contracts (10-19 09:30),
   at 42.5 and 50.25 two (06-27, 05-10). Those three trades are the three
   stops of the pass; contract-weighted they cost 299.25 points against
   137.0 per contract. The budget is a fraction, not a contract count —
   `max_quantity` is 5 — and the tight-stop trades, which the chop days
   produce, are the ones it multiplies.
2. **Thesis scale from the bias.** 33 distinct plans reached the machine:
   `governing_timeframe` 15m ×32, 1H ×1 (02-24 09:25, under SHORT@4H),
   5m ×0. The previous pass had 5m ×10 of 41. The one 1H thesis was
   sized (108.25 points, one contract) and submitted; no thesis was lowered
   to fit the gate — the size vetoes above were re-proposed on the same
   scale or dropped.
3. **Target and invalidation scales.** Targets: 15m ×24, 1H ×4, 4H ×5, 5m
   ×0 (previous pass: 5m ×4). Invalidations: the thesis scale ×11, one
   below ×22. No `opportunity_target_scale` or invalidation-scale rejection
   was recorded — the prompt's rule held without the reducer refusing.
   The fills' stops are 34.75 to 117.5 points, 3.9–4.6 one-minute ATRs by
   window medians (previous pass 3.9; the entry model 1.6).
4. **The structural exit.** `bias_reversed` ×0 (it no longer exists);
   `structure_reversed` ×1: 10-19 10:16, a SHORT filled at 10:15 closed
   −9.5 on the 15m's structure turning up, and the working entries
   withdrawn with it (`cancel structure_reversed` ×2). The three positions
   open at the window ends sat through bias changes: 02-24's LONG through
   nothing (LONG held, +95.0 at 12:30, right); 05-10's SHORT was entered at
   12:50 under SHORT@1H (+77.0 at 13:00); 10-19's LONG, entered 10:44
   under LONG@15m, stayed open through NEUTRAL at 11:00 and SHORT@15m at
   11:15 because the 15m printed no MSS / BOS down before 12:30 (−68.5 at
   the close, stop 103.25 untouched). Under the previous rule it would have
   been flattened at the 11:00 flip.
5. **The bias decay.** Five decays, two re-asserts refused, all on the
   reversal and chop days:
   - 01-24: SHORT@4H at 12:00 (the day's low printed 12:18), SHORT@1H
     12:45, `bias_decayed` 13:15 → NEUTRAL; the model's SHORT@1H at 14:15
     and 14:40 were `bias_reassert_refused` (NEUTRAL kept); SHORT@15m at
     14:45 was allowed (a different pair) and vetoed at 133 points; LONG@15m
     at 15:31 (its first plan incoherent, its second vetoed at 208 points),
     LONG@1H 16:00. The reading turned 57 minutes after the low and never
     went short again on the 1H; nothing was filled on the 800-point rise.
   - 02-24: SHORT@4H 08:40 (the low was 08:31), NEUTRAL 09:45, LONG@15m
     10:00, LONG@1H 10:19; the LONG filled 10:31 at 13 355 (117.5-point
     floored stop, one contract) and was +95 at 12:30 — the pass's one
     right fill. No decay fired: the model itself dropped the 4H short.
   - 10-13: SHORT@15m 09:00 (the low 09:34), `bias_decayed` 09:45 →
     NEUTRAL, then **SHORT@4H at 10:00** — a different pair, so allowed —
     decayed again at 11:00 (`bias_decayed:SHORT@4H:2`); LONG@15m 11:15,
     LONG@1H 12:00. The eight LONG plans 11:15–11:55 were vetoed (four
     `reward_risk` 1.48–1.75, four `position_size` at 171–211 points); the
     ninth, 96.5 points, was submitted at 12:15 and never filled. The
     reading was right from 11:15 (high 11:41); the account could not buy
     it.
   - 05-10: LONG@15m at 09:30, filled 09:31 at the day's high, stopped
     09:37 (−50.25 × 2); NEUTRAL 09:35 (`bias_decayed:LONG@15m:2`);
     SHORT@15m 11:00, SHORT@1H 11:10 (`since` kept from 11:00 — continuity
     is by direction); at 11:40 the reply moved the bias back to the 15m
     and that pair decayed at the day's low (11:30) on two 5m
     counter-events (`bias_decayed:SHORT@15m:2`); SHORT@1H at 12:45 was a
     different pair and allowed, the SHORT filled 12:50, +77 at 13:00.
   - 09-13: SHORT@15m 09:00, decayed 09:35 on the bounce into the 09:33
     high, SHORT@1H 10:00 and held; no plan was submitted on the one-way
     fall (12 508 → 12 262).

## 3. Window notes

- **Trend days (08-26, 09-13, 11-10, 07-27).** The bias was right and
  held (accuracy 1.00 / 1.00 / 0.83; 07-27 has six calls after the FOMC
  sleep). 08-26 submitted three 15m theses at pullback zones, all
  `plan_dropped` unfilled as the leg ran; 11-10 one, unfilled; 09-13 none.
  Zero fills on the trend days, as in the previous pass: the entry model
  waits for a pullback to a 15m object that a one-way day does not give.
  Nothing in this build touched entries (paused by the user).
- **10-19 (mixed).** SHORT@1H at 09:30; a SHORT at 11 153.75 filled
  09:31 (`entry.pool.midpoint`, the pool's midpoint 24.5 points under the
  close), three contracts on a 34.75-point stop, stopped 09:35 (−34.75 ×
  3 = −104.25) eight minutes before the day's low. SHORT again 10:15 at
  11 198, closed 10:16 by the 15m reversal (−9.5). LONG@15m 10:20, filled
  10:44 at 11 219.5, thirteen minutes after the high; open −68.5 at 12:30.
- **06-27 (low conviction).** Three SHORT plans 09:45–10:00 vetoed (148–161
  points); LONG@15m at 11:05, filled 11:06, stopped 12:07 (−42.5 × 2);
  SHORT submitted 12:30, unfilled.
- **Contracts.** Four of the seven fills were at one contract by the
  budget (117.5, 88.75, 81.25 and 103.25 points), three at two or three
  (34.75–50.25 points). The three multi-contract fills are the three
  stops.

## 4. Replays and commands

Every journal replays: `python -m brain.scripts.replay_journal --run-dir
outputs/brain_journal/<id>` reproduces the calls, revisions and trade
records of each of the ten runs (see §5 of the frozen-window receipt for
the frozen run). Comparison: `scratchpad/compare_scale_exit_bias.py
benchmark_ids_stop_floor.txt benchmark_ids_scale_exit_bias.txt`; the bias
timelines and structural events: `scratchpad/design_evidence.py`; the
plan scales: `scratchpad/plan_scales.py`; the open marks:
`scratchpad/open_marks.py`.

## 5. What this pass settles and what it leaves

- Settled: the thesis scale is the bias's (no 5m thesis), targets are on
  the thesis scale or above (no 5m target), a bias flip no longer closes a
  trade, a 4H or 1H bias ends within an hour of the turn on the reversal
  days, the decay memory holds across sleeps and episodes (pass 1's
  defect), and the rejection feedback names the decayed pair.
- Left, in order of cost: **(a)** the fraction sizes tight stops up —
  the three losers were 2–3 contracts and the winner one; at one contract
  the pass is −137.0 closed, −33.5 with the open marks. `max_quantity` 1
  (or a grade-keyed cap) is a configuration decision, not made here.
  **(b)** the reversal days' correct theses are 130–290-point stops on a
  20–40 ATR tape, over the 125 points one contract of NQ affords at 2.5 %
  of 100 000 USD (10 vetoes; 01-24 and 10-13 unfilled while right): a
  larger account or MNQ, as the spec's §3 says. **(c)** the trend days
  still fill nothing — the entry model, paused. **(d)** a decayed pair
  can be re-asserted on a *higher* scale, whose decay scales do not count
  the events that ended the lower one (10-13 SHORT@4H at 10:00 after
  SHORT@15m decayed at 09:45; 05-10 SHORT@1H at 12:45 after SHORT@15m
  decayed at 11:40) — by design (§1.5); the 10-13 reading decayed within
  the hour, the 05-10 one was right. Whether the refusal should cover the
  higher scales is open.
