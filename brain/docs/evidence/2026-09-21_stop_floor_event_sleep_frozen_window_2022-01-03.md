# The frozen window under the stop floor and the event sleep

*2026-09-21. Run `f4f6998a6ed59eb8` (label `stop-floor-events`), the
frozen window `2021-12-27` warm-up → emit `2022-01-02 18:00` → `2022-01-03
17:00` New York, DeepSeek `deepseek-flash@high`, `--broker sim`, 1 380
bars, 326 calls (279 answered), 7.3 hours, 6.24 USD. Compared with the same
window under the entry model with every pool placed (`2710e0b78a707e99`,
[2026-09-20_entry_model_frozen_window_2022-01-03.md](2026-09-20_entry_model_frozen_window_2022-01-03.md) §7),
whose Brain inputs differ only by the prompt's floor sentences and the
controller's calendar (no release falls in this window — NFP is 01-07 —
so the calendar changes the run identity and nothing else here).*

## 1. The paired reading

| | `2710e0b7` (entry model, pools) | `f4f6998a` (stop floor + event sleep) |
| --- | --- | --- |
| calls / USD / incidents | 278 / 5.88 / 3 | 279 / 6.24 / 14 (9 `max_tokens` truncations, 5 contract violations) |
| bias direction accuracy 60m (readings) | 0.44 (232) | 0.50 (244) |
| ACTIONABLE accuracy 60m (readings) | 0.35 (103) | 0.47 (197) |
| bias flips in the day | 17 | 17 |
| rejections `opportunity_incoherent` (model-side) | 6 | 5 |
| gate vetoes | `reward_risk` 6, `daily_stop` 2 | `reward_risk` 3, `position_size` 3, `leverage` 1, `exposure` 2 |
| `thesis_refused` | `thesis_engaged` 59, `entry_marketable` 2 | `thesis_engaged` 96, `entry_marketable` 3 |
| submitted / filled / expired | 9 / 4 / 0 | 16 / 9 / 1 |
| fills at the limit price | 4 of 4 | 9 of 9 |
| waits (minutes) | 1 / 1 / 1 / 1 | 28 / 1 / 43 / 1 / 1 / 2 / 23 / 101 / 15 |
| location in the 240-bar range ≥ 0.8 | 1 | 2 (resting limits, filled at the limit) |
| stop distance, median points (1m ATRs) | 25.9 (7.04) — 8.25 to 49.5 | 26.8 (4.78) — 15.25 to 71.0, minimum 3.9 ATRs |
| contracts | 2 / 2 / 2 / 1 | 2 overnight (leverage-capped), 1 in RTH |
| exits | stop ×3 (−32.5, −16.5, −49.5), `bias_reversed` ×1 (−43) | stop ×1 (−49.0), `bias_reversed` ×7 (−14, −1, −11, −4.5, −12.75, +18.25, −3.25), 1 open at the close |
| right an hour after the fill | 1 of 4 | 7 of 9 |
| daily stop | 08:59 New York, the afternoon vetoed | not reached |
| points | −141.5 | −77.25 |
| replay | OK | OK: 2 episodes, 279 calls, 1 376 revisions, 208 trade records |

## 2. What the floor did here

The window is a quiet Globex night (1m ATR 2.8–4.3) and a 09:30 drive
down then up (ATR 10–13). Overnight the floor moved three stops
(`stop.floor.governing_bar`: 16.5, 15.25 and 16.75 points, 3.9 ATRs each)
where the object's edge sat 8–12 points away, and left the rest at
objects already beyond it; in RTH every stop was 41–71 points at one
contract. Nothing was stopped inside the noise: the one stop of the day
(T6, LONG at 16 409.25 filled 05:07, stopped at 09:00 for −49) sat 24.5
points under the entry — 8.6 overnight ATRs — and was taken out by the
open's 60-point drive. The third pass lost three stops of 8–35 points and
the daily limit at 08:59; this pass never reached the daily limit and
traded the afternoon.

## 3. What decided the P&L instead

Seven of eight closed trades ended by `bias_reversed`, at an average of
−4 points, within 5 to 90 minutes of the fill — the day's 17 bias flips
(the same 17 as the third pass) took the positions out before any stop
or target was near. The afternoon LONG T7 (16 386.75 at 11:38, the low of
the day's range, right an hour later) was flattened at 16 405 (+18.25) on a
flip; its target was 16 512. The stop floor removed the noise stop-out and
the daily-stop veto; the bias's own instability now sets the exits. That
is the next receipt's subject, not this one's.

## 4. Old defects, checked

- No fill at the market: 9 of 9 at the limit; three chases refused
  (`entry_marketable`); waits up to 101 minutes.
- No code-made incoherence (`opportunity_incoherent` 5, all the model's:
  an invalidation named on the wrong side — the same fault as §7 of the
  entry-model receipt, 6 there).
- No expiry block: one TTL expiry (a 15m object's 225 bars), the thesis
  refunded.
- Replacement free: `signature_changed` ×4 cancels; `thesis_engaged` 96
  refusals — the model keeps re-proposing the thesis it already holds, more
  than before (59), because it holds positions longer.
- Incidents: 14 against 3, nine of them the reasoning budget
  (`max_tokens=32768`, `finish_reason=length`) — the same failure the
  benchmark receipts have recorded on fast tape; the prompt is 12 lines
  longer and the inputs are not. Recorded, not fixed here.

## 5. Commands

```bash
.venv/bin/python -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high --label stop-floor-events
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/f4f6998a6ed59eb8 --run-dir outputs/brain_journal/2710e0b78a707e99
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/f4f6998a6ed59eb8
```
