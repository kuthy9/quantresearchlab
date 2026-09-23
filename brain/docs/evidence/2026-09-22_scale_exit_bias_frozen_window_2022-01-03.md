# The frozen window under the thesis scale, the structural exit and the bias decay

*2026-09-22. Run `2cd6fd64acccccfa` (label `scale-exit-bias-2`, the second
pass on the tree the review fixed), the frozen window `2021-12-27` warm-up
→ emit `2022-01-02 18:00` → `2022-01-03 17:00` New York, DeepSeek
`deepseek-flash@high`, `--broker sim`, 1 380 bars, 285 calls (278
answered), 6.7 hours, 6.00 USD. Compared with the same window under the
stop floor and the event sleep (`f4f6998a6ed59eb8`,
[2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md](2026-09-21_stop_floor_event_sleep_frozen_window_2022-01-03.md)),
whose Brain differs by the five rules of the spec
([2026-09-22-scale-exit-bias-design.md](../specs/2026-09-22-scale-exit-bias-design.md))
and nothing else. The tape: a quiet Globex night from 16 356 to a 06:40
high of 16 452.5, a 09:30 drive to the day's low 16 292 at 09:56, then a
rise to 16 508 at 16:22 and a 16 495.75 close.*

## 1. The paired reading

| | `f4f6998a` (stop floor + event sleep) | `2cd6fd64` (thesis scale, structural exit, bias decay) |
| --- | --- | --- |
| calls / USD / incidents | 279 / 6.24 / 14 | 285 / 6.00 / 9 (7 `max_tokens` truncations, 2 contract violations) |
| bias direction accuracy 60m (readings) | 0.50 (244) | 0.53 (150 — NEUTRAL readings are not counted) |
| ACTIONABLE accuracy 60m (readings) | 0.47 (197) | 0.45 (80) |
| proposals by state ACTIONABLE / DEVELOPING / NONE | 207 / 21 / 42 | 85 / 62 / 131 |
| distinct opportunities | 47 | 16 |
| bias changes / NEUTRAL revisions | 17 / 24 | 20 / 128 |
| bias decays / re-asserts refused / `opportunity_against_bias` | — / — / 0 | 5 / 0 / 5 |
| rejections `opportunity_incoherent` (model-side) | 5 | 2 |
| gate vetoes | `reward_risk` 3, `position_size` 3, `leverage` 1, `exposure` 2 | `reward_risk` 3, `leverage` 8, `exposure` 2, `position_size` 0, `account_risk` 0 |
| `thesis_refused` | `thesis_engaged` 96, `entry_marketable` 3 | `entry_marketable` 2 |
| submitted / filled | 16 / 9 | 5 / 3 |
| fills at the limit price | 9 of 9 | 3 of 3 |
| waits (minutes) | 28 / 1 / 43 / 1 / 1 / 2 / 23 / 101 / 15 | 74 / 6 / 12 |
| thesis scale of the plans | 15m, 5m (the model's) | 1H ×2 (under SHORT@4H), 15m ×12 (code's) |
| target scale of the plans | the model's, 5m pools allowed | 1H ×7, 15m ×7, 5m ×0 |
| stop distance, points (rule) | median 26.8, 15.25 to 71.0 | 43.0 (5m swing), 71.75 (15m floor), 52.25 (15m floor) |
| contracts | 2 overnight, 1 in RTH | 2 / 1 / 2 (the leverage cap holds two NQ on this account) |
| exits | stop ×1 (−49.0), `bias_reversed` ×7 (average −4), 1 open at the close | `structure_reversed` ×1 (−27.0 × 2), stop ×1 (−71.75), 1 open at the close (+87.5 × 2) |
| right an hour after the fill | 7 of 9 | 1 of 3 |
| daily stop | not reached | not reached |
| points, closed | −77.25 | −125.75 (−98.75 per contract) |
| points with the open position marked at 17:00 | −30.75 | +49.25 (−11.25 per contract) |
| replay | OK | OK: 2 episodes, 285 calls, 1 376 revisions, 41 trade records reproduced |

## 2. The five rules, as journaled

1. **Risk budget.** No `position_size` veto (three in the previous run):
   every stop the day produced — 43, 71.75 and 52.25 points on the fills,
   10.5 to 74.75 on the proposals — fits 2.5 % of the account. The eight
   `leverage` vetoes are the 8× cap holding two NQ contracts on 97–100
   thousand USD (328 000 USD notional each): every LONG the model proposed
   after 11:12, with two contracts already open, was refused by leverage,
   not by the budget. `account_risk` never fired.
2. **Thesis scale from the bias.** 14 distinct plans: `governing_timeframe`
   1H ×2 (the 19:00–19:01 SHORT under SHORT@4H), 15m ×12; the model did
   not name a scale and none was lowered.
3. **Target and invalidation scales.** Targets 1H ×7, 15m ×7, none on the
   5m; invalidations on the thesis scale ×10, one below ×4. The overnight
   plans (`stop.floor.governing_bar` at 10–17 points, ratios 5.2–7.4) were
   refused by leverage or exposure, not by the ratio: with a 1H or 15m
   pool as the target the ratio is no longer the binding veto (three
   `reward_risk` vetoes, as before).
4. **The structural exit.** `bias_reversed` ×0 (seven exits and fourteen
   cancels in the previous run); `structure_reversed` ×1: the overnight
   SHORT (16 421.25 × 2, filled 20:15 after a 74-minute wait under
   SHORT@4H) was flattened at 07:01 when the 15m structure turned up, −27
   × 2 — the 4H bias itself had decayed at 22:30, eight and a half hours
   earlier, and the position sat through it. The 11:12 LONG (16 408.25 ×
   2, `entry.pool.midpoint`) sat through the 11:15 decay of LONG@15m and
   the 14:35 one and was +87.5 a contract at the close (the day's high
   16 508 at 16:22); under the previous rule it would have been flattened
   at 11:15.
5. **The bias decay.** Five decays, none refused:
   - 22:30 (01-02): `bias_decayed:SHORT@4H:2` — the 4H short set at 19:00
     ended after two counter-events on the 4H / 1H / 15m; the earlier runs
     read this 4H short through the whole night and the day
     ([direction root cause](2026-09-18_direction_root_cause_2022-01-03.md)).
   - 08:30: `LONG@1H:2` (set 07:00 at the overnight high); 09:30:
     `SHORT@15m:2` (set 08:57 into the drive down); 11:15 and 14:35:
     `LONG@15m:2` — each LONG@15m was re-set later (13:45, 16:35) after
     the 15m printed its own MSS / BOS, so no `bias_reassert_refused`.
   - The bias was NEUTRAL in 128 of 285 revisions (24 before): the model
     proposed 131 NONE opportunities against 42, and 85 ACTIONABLE against
     207. Fewer, larger theses: 5 submissions for 16, 3 fills for 9.

## 3. The day's trades

- **T1, SHORT 16 421.25 × 2, 20:15 → 07:01, −27 × 2.** Proposed at 19:00
  under SHORT@4H on the 1H (the first version vetoed at ratio 1.01, the
  second submitted with a 43-point 5m-swing stop and a 1H target);
  filled 74 minutes later at the 15m BSL. The 4H bias decayed at 22:30;
  the night ranged 16 356–16 452; the 15m turned up at 07:01 and the
  machine flattened at 16 448.25.
- **T3, LONG 16 426 × 1, 09:46 → 09:49, −71.75.** LONG@15m at 09:40 into
  the 09:30 drive; entry at the 5m FVG near edge, stop at the 15m floor
  71.75 points under (16 354.25), one contract by the budget; the drive
  ran to 16 292 at 09:56, 62 points through the stop. The 15m floor did
  not save a long bought seven minutes before the low.
- **T5, LONG 16 408.25 × 2, 11:12 → open.** LONG@15m at 11:00, entry at
  the 5m SSL midpoint, stop at the 15m floor 52.25 under; filled 12
  minutes later; the LONG@15m decayed at 11:15 and the position stayed;
  +87.5 a contract at 17:00. The four LONG plans after it (13:45,
  16:35–16:50) were leverage-vetoed with two contracts open.
- Two other submissions never filled: the 07:10 LONG at 16 404.75 × 2
  (33.75-point stop) was cancelled `signature_changed` at 08:25 as its
  objects changed and the replacement (63.5-point stop, ratio 1.44) was
  vetoed at 08:26; the 08:57 SHORT at 16 421.5 × 2 was cancelled
  `plan_dropped` at 09:30 when SHORT@15m decayed and the model dropped the
  plan. The two `structure_reversed` cancels in the summary are T1's exit
  legs, withdrawn with the flatten; no working entry was on the reversed
  scale at 07:01.

## 4. Old defects, checked

- No fill at the market: 3 of 3 at the limit; two chases refused
  (`entry_marketable`).
- No code-made incoherence: `opportunity_incoherent` 2, both the model's
  (an entry 0.75 under the price at 20:34; an invalidation on the wrong
  side at 03:10).
- `thesis_engaged` refusals: 0 (96 before) — the model stopped re-proposing
  the thesis it already holds once the prompt says the position is the
  thesis's expression and the bias no longer needs re-asserting every
  call.
- Incidents 9 against 14: seven reasoning-budget truncations
  (`finish_reason=length`), one `opportunity.grade` null on a NONE
  opportunity, one `evidence_verdicts` without `resolution`. Recorded, not
  fixed here.

## 5. Commands

```bash
.venv/bin/python -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high --label scale-exit-bias-2
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/f4f6998a6ed59eb8 --run-dir outputs/brain_journal/2cd6fd64acccccfa
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/2cd6fd64acccccfa
```
