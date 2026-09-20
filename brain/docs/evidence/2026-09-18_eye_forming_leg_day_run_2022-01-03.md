# Eye layer of the direction fix — the forming leg on the 2022-01-03 session

Layer 1 of
[specs/2026-09-18-direction-eye-brain-execution-design.md](../specs/2026-09-18-direction-eye-brain-execution-design.md)
(plan
[plans/2026-09-18-direction-eye-layer.md](../plans/2026-09-18-direction-eye-layer.md)):
the Eye's per-scale facts describe the leg that is forming, date the last
displacement and name a broken protection; nothing else changed (the
prompt is byte-identical to `72ea13c7`'s). Two checks: the deterministic
audit of the facts, and run **E** (`5c491789ccef7367`) on the frozen window
against `72ea13c7fcbc1cff`. Times are New York.

## 1. The facts, without an LLM (`audit_scales.py`)

`.venv/bin/python -m brain.scripts.audit_scales --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00`

| scale | before (the diagnosis) | now |
| --- | --- | --- |
| 4H | `active=short, expansion` 02:00 → 17:00 while price rose 100 points | `active=long, last=short, retracement, forming +1.22 ATR` from 02:00; `active=short` only in the 09:53 flush; MSS long 17:00 |
| 1H | `active=short, expansion, displacement 0.64` 05:00 → 07:00 at the start of the leg up | 05:00 `active=long, retracement, forming +0.92, displacement short / 10 bars old`; 07:00 MSS long → `reversal_attempt`; 09:39 `reset=long`, `balance`, displacement short / 15 bars; 10:00 displacement short / 0; 13:00 displacement long / 0 |
| 15m | `internal short` 10:00 → 13:45 with nothing saying which way price was going | same structure, plus `active=long, forming +2.9 / +3.0 ATR` at 10:30 / 11:00 and `disp=long/0` at 11:00; MSS long 13:45 → `expansion` |

2022-01-04 (`--warmup-start 2021-12-28 --emit-start 2022-01-03T18:00 --end 2022-01-04T17:00`)
runs clean: the 1H reads `reset=short` and `transition` at 11:00 when
price left its range on the sell-off and `balance` at 16:00 when it came
back; the 4H's active leg flips by ±0.05 ATR around a swing at 10:39–10:43
and 15:51–15:52 — the noise the spec named, published with its size.
`expansion` against an opposite forming leg cannot occur any more: the
phase is derived from the active leg.

## 2. Run E against `72ea13c7` (same prompt, same window, same effort)

| | run E `5c491789` | `72ea13c7` |
| --- | --- | --- |
| LLM calls / cost (peak USD) | 334 / 6.40 | 343 / 6.59 |
| episodes / sleeps | 1 / 0 | 6 / 5 |
| sharp-move coverage (all / RTH) | 77 of 81 / 25 of 25 | 78 of 81 / 25 of 25 |
| ACTIONABLE replies | 92 | 111 |
| ACTIONABLE overnight SHORT / LONG | 40 / 23 | 55 / 11 |
| ACTIONABLE RTH SHORT / LONG | 11 / 18 | 11 / 34 |
| direction changes of the opportunity (session / RTH) | 13 / 4 | 14 / 8 |
| `direction_accuracy_60m` (state revisions with a direction) | 0.381 (218) | 0.359 (231) |
| `direction_accuracy_60m` (ACTIONABLE replies) | 0.371 (89) | 0.364 (110) |
| orders / fills / expired | 22 / 14 / 6 | 26 / 9 / 12 |
| closed: stop / target / invalidation | 8 / 3 / 2 | 6 / 1 / 2 |
| realized (points / USD by the fills) | −109.5 / −2 190 | −103 / −2 060 |
| open at the close | 2 LONG at 16498 (−90 USD) | none |
| RR vetoes / thesis refusals | 17 / 5 | 16 / 3 |
| understanding replaced | 34 | 41 |

`direction_accuracy_60m` is the share of state revisions whose stated
direction (the opportunity's; the bias from the Brain layer on) matched the
sign of the close sixty minutes later (session scratch `direction_accuracy.py`;
the summarizer carries it from the Brain layer). The run that started this
work, `7ec17f06`, scores 0.394 on 368 revisions — every run so far is
*below one half* on this day: the stated direction is wrong more often than
right an hour out.

Closed trades of run E:

| closed | thesis | side | entry → exit | exit | points |
| --- | --- | --- | --- | --- | --- |
| 18:03 | T1 | SHORT | 16389.50 → 16397.50 | stop | −16.0 |
| 19:25 | T2 | SHORT | 16394.00 → 16399.00 | stop | −10.0 |
| 22:00 | T4 | SHORT | 16406.50 → 16416.50 | stop | −20.0 |
| 00:04 | T5 | SHORT | 16406.50 → 16411.50 | stop | −10.0 |
| 01:11 | T3 | LONG | 16408.25 → 16404.00 | invalidation | −8.5 |
| 03:26 | T7 | SHORT | 16403.00 → 16407.75 | invalidation | −9.5 |
| 04:15 | T8 | SHORT | 16395.00 → 16403.75 | stop | −17.5 |
| 05:13 | T10 | LONG | 16409.50 → 16418.25 | target | +17.5 |
| 06:02 | T12 | LONG | 16427.75 → 16438.25 | target | +21.0 |
| 08:58 | T15 | SHORT | 16424.50 → 16408.25 | target | +32.5 |
| 09:10 | T16 | SHORT | 16387.50 → 16391.75 | stop | −8.5 |
| 11:12 | T18 | LONG | 16421.50 → 16404.00 | stop | −35.0 |
| 13:42 | T20 | SHORT | 16421.50 → 16444.25 | stop | −45.5 |

## 3. What the facts alone moved

- **The overnight rise was read earlier.** The Brain went LONG for good at
  05:45 ("The delivering scales now drive … 15m long, phase expansion, a
  bullish displacement only 2 bars old") and wrote at 06:28 "the 4H/1H
  backdrop is short but neither scale is delivering supply — both sit in
  retracement with the active leg long". In `72ea13c7` the first durable
  long came with the 1H MSS at 07:00, five points under the top. Two
  longs reached their targets (T10, T12). The 15m acceptances of
  05:15–06:30 were SUPPORT (seven CONTRADICT before); the 15m MSS at 05:00
  was still CONTRADICT — the reply that flipped came three calls later.
- **The overnight side balanced.** ACTIONABLE overnight went from 55 SHORT
  / 11 LONG to 40 / 23; the ages appear in the reasoning ("1H still only
  retracing an 11-bar-old bearish displacement").
- **RTH churn halved** (4 direction changes against 8), coverage held.

## 4. What they did not move

- **The frame.** Nearly every reply still opens with "HTF stays bearish …
  the up-move remains a counter-leg"; at 13:42 the Brain shorted into the
  breakout again ("The 4H still governs short", T20, −45.5, the worst trade
  of the day), three minutes before the 15m MSS long it then followed. The
  1H `reset: long` from 09:39 was read as "a fresh internal long reset"
  once and never as a live side.
- **Direction accuracy** rose two points and stays below one half. The
  facts are timelier; the reading is the model's, and it is not fixed by
  facts alone — as the spec expected of this layer.
- **The right-side thesis after 13:45 died the same way.** T21 LONG: a
  limit at 16449.25 (15m FVG) expired, two RR vetoes, a limit at 16467.75
  expired; T22 filled at 16498 at 16:43, the session high, and was open at
  the close. The execution layer's fixes stand.
- **No sleep all session.** The Brain never set `continue_active: false`
  (three times in `72ea13c7`) and the idle rule never fired: one episode,
  1380 awake bars. Calls did not rise (the phase-transition bookkeeping
  removed about as many as the awake stretch added), but the Brain-layer
  prompt must restore the sleep rule beside the bias section.

## 5. Verdict on the layer

The Eye now says what is forming, with its size and age, and both
producers agree (parity test, 1483 tests green). The layer is accepted:
it removed the false "expansion short" labels that the diagnosis found
under every overnight refusal, and it moved the overnight reading two
hours earlier. It did not move the day's direction, which was never the
Eye's to move. The Brain layer follows.

## 6. Commands

```bash
.venv/bin/python -m brain.scripts.audit_scales --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00
.venv/bin/python -m brain.scripts.audit_scales --warmup-start 2021-12-28 --emit-start 2022-01-03T18:00 --end 2022-01-04T17:00
.venv/bin/python -u -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/5c491789ccef7367 --run-dir outputs/brain_journal/72ea13c7fcbc1cff --write
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider
```
