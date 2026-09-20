# Execution layer of the direction fix — run X on the 2022-01-03 session and the guard run on 2022-01-04

Layer 3 of
[specs/2026-09-18-direction-eye-brain-execution-design.md](../specs/2026-09-18-direction-eye-brain-execution-design.md)
(plan
[plans/2026-09-18-direction-execution-layer.md](../plans/2026-09-18-direction-execution-layer.md)):
an expiry gives the thesis its expression back, a limit through the market
fills at the open in the simulator, a position on the other side of the
Brain's bias is flattened at market (`bias_reversed`, added after run B′),
and the summarizer counts `missed_trends`. Two fixes outside the three
layers came with it: `--label` enters the run identity (executor code is
not hashed, so run X was first refused as a duplicate of B′), and the
pending-evidence runaway found by run X's first attempt (§3.1). This
receipt closes the build with the layered comparison. Times are New York;
points are contract-points (two contracts).

## 1. Deterministic checks

- Suite: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider` — 1511 passed.
- Replay from the Eye alone (`replay_journal.py`): run X `ab572b09d59bc4d7`
  — 11 episodes, 290 LLM calls, 1337 revisions, 69 trade records
  reproduced; the 2022-01-04 run `5cfe65bcc480a032` — 5 episodes, 246
  calls, 1365 revisions, 71 trade records reproduced. Both are now the
  regression baselines in `regression_baselines.json`
  (`test_regression_baseline.py -m research_orchestration -o addopts=''`).

## 2. Run X against B′, B, E and `72ea13c7` (the frozen window)

| | run X `ab572b09` | run B′ `3cc1bb40` | run B `07f897f4` | run E `5c491789` | `72ea13c7` |
| --- | --- | --- | --- | --- | --- |
| LLM calls / cost (peak USD) | 290 / 5.91 | 302 / 6.83 | 297 / 7.09 | 334 / 6.40 | 343 / 6.59 |
| episodes / sleeps (idle, `continue_active=false`) | 11 / 11 (6, 5) | 14 / 14 (5, 9) | 5 / 4 (3, 1) | 1 / 0 | 6 / 5 (2, 3) |
| sharp-move coverage (all / RTH) | 77 of 81 / 25 of 25 | 80 of 81 / 25 of 25 | 77 of 81 / 25 of 25 | 77 of 81 / 25 of 25 | 78 of 81 / 25 of 25 |
| repairs / malformed incidents | 30 / 5 | 37 / 2 | 31 / 7 | 20 / 1 | 20 / 2 |
| ACTIONABLE replies | 46 | 76 | 103 | 92 | 111 |
| ACTIONABLE overnight SHORT / LONG | 23 / 15 | 38 / 22 | 11 / 58 | 40 / 23 | 55 / 11 |
| ACTIONABLE RTH SHORT / LONG | 3 / 5 | 13 / 3 | 20 / 14 | 11 / 18 | 11 / 34 |
| `opportunity_incoherent` rejections | 8 | 16 | 22 | 0 | 1 |
| bias changes / NEUTRAL revisions | 21 / 46 | 14 / 21 | 28 / 37 | — | — |
| directional segments agreeing with their own drift | 8 of 22 | 6 of 20 | 8 of 26 | — | — |
| `direction_accuracy_60m` (state revisions) | 0.465 (230) | 0.502 (267) | 0.445 (247) | 0.381 (218) | 0.359 (231) |
| `direction_accuracy_60m` (ACTIONABLE replies) | 0.318 | 0.384 | 0.388 | 0.371 | 0.364 |
| orders / fills / expired | 9 / 7 / 1 | 18 / 10 / 7 | 19 / 10 / 2 | 22 / 14 / 6 | 26 / 9 / 12 |
| closed: stop / target / invalidation / bias_reversed | 5 / 0 / 1 / 1 | 7 / 1 / 2 / — | 7 / 1 / 2 / — | 8 / 3 / 2 / — | 6 / 1 / 2 / — |
| `missed_trends` (missed of expired) | 1 of 1 | 3 of 7 | 0 of 2 | 2 of 6 | 4 of 12 |
| realized (points / USD) | **−146.5 / −2 930** | −139.5 / −2 790 | −121.0 / −2 420 | −109.5 / −2 190 | −103 / −2 060 |
| RR vetoes / daily-stop vetoes / thesis refusals | 13 / 1 / 4 | 15 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 5 | 16 / 0 / 3 |

Closed trades of run X:

| closed | thesis | side | entry → exit | exit | points |
| --- | --- | --- | --- | --- | --- |
| 18:03 | T1 | SHORT | 16389.50 → 16397.50 | stop | −16.0 |
| 19:31 | T2 | SHORT | 16389.50 → 16405.75 | stop | −32.5 |
| 01:12 | T3 | LONG | 16402.50 → 16393.00 | stop | −19.0 |
| 03:26 | T4 | SHORT | 16404.00 → 16407.75 | invalidation | −7.5 |
| 05:01 | T5 | SHORT | 16409.50 → 16412.00 | **bias_reversed** | −5.0 |
| 08:34 | T1 | LONG | 16438.75 → 16422.00 | stop | −33.5 |
| 09:48 | T2 | LONG | 16413.50 → 16397.00 | stop | −33.0 |

### 2.1 What the layer's rules did in the run

- **The bias-reversal exit fired once**: T5 SHORT (opened 04:57) was
  flattened at the 05:00 LONG flip for −2.5 a contract, four minutes into
  the trade, and the Brain's next long (T3, opened 05:16 in run B′'s
  counterpart) was not blocked by it. **No short was open into the 13:45
  breakout** — the first of the five runs without the 13:4x stop (−20.5,
  −45.5, −32.5, −58.5 before).
- **The expiry refund** had one expiry to act on (`missed_trends` 1 of 1:
  the tape ran a full R away from the limit). **The marketable fill**
  never occurred: all seven entries filled at their limit.
- **The daily stop decided the day.** The seventh loser (09:48) took the
  session past 2.5 % of its opening equity; from then on the gate vetoed
  every new entry ("the session of 2022-01-03 lost 2.5% of its opening
  equity; no new entries today"). The correct afternoon thesis — bias
  LONG/15m from 13:00, LONG/1H from 13:45, the +73-point drift — produced
  four LONG proposals: 13:45, 14:20 and 16:45 were dropped by the reducer
  as `opportunity_incoherent` (a LONG whose target lay below its entry),
  15:00 was vetoed `daily_stop`. Nothing was expressed.

## 3. The two fixes outside the layers

### 3.1 The pending-evidence runaway (run X's first attempt, `09c02ebd39b02f1a`, stopped)

From 10:15 every DeepSeek reply came back "content is empty" — 25 calls in
a row with no reply. Each empty reply left its evidence unjudged; the
runtime re-offered all of it on the next call; the input grew from 29k to
52k characters (19 → 124 pending items) and no reply could come any more.
The cause of the first empties is the reasoning budget: completion tokens
of 25–31k against `max_tokens` 32 768 were common on the RTH inputs of
every run, and the client reported an empty content before it looked at
`finish_reason`. Fixed (spec §5): `max_pending_evidence` (32, in
`main_brain.json`) bounds the ledger's pending items — the oldest expire
as `evidence_expired`, in both the incident carry-forward and the update
path — and the client's incident now names the finish reason, the
completion tokens and the reasoning size (a truncation is reported as one
before the empty check). In attempt 2 the input peaked at 36k characters
at 09:45 and fell back to 24–28k; the 10:15 call that had started the
runaway succeeded on its third attempt; the bound itself never had to
expire an item.

### 3.2 `--label`

The run identity hashes the tape, the window, the model, the prompt, the
two configs and the Eye's identities — not the executor code — so run X
was refused as a duplicate of B′. `--label <text>` enters the identity for
a deliberate re-run on the same Brain inputs; without it every id is
unchanged.

## 4. The guard run: 2022-01-04 (`5cfe65bcc480a032`)

The day none of this was tuned on (`--warmup-start 2021-12-28
--emit-start 2022-01-03T18:00 --end 2022-01-04T17:00`), same code and
prompt as run X.

| | 2022-01-04 |
| --- | --- |
| LLM calls / cost (peak USD) | 246 / 5.82 |
| episodes / sleeps (idle, `continue_active=false`) | 5 / 4 (3, 1) |
| sharp-move coverage (all / RTH) | 75 of 79 / 26 of 26 |
| repairs / malformed incidents | 23 / 2 |
| ACTIONABLE overnight LONG / SHORT | 20 / 5 |
| ACTIONABLE RTH SHORT / LONG | 18 / 7 |
| bias changes / NEUTRAL revisions | 6 / 14 |
| `direction_accuracy_60m` (state revisions / ACTIONABLE) | 0.487 (224) / 0.292 |
| orders / fills / expired / `missed_trends` | 11 / 4 / 2 / 1 of 2 |
| closed: stop / target / invalidation / bias_reversed | 2 / 1 / 1 / 0 |
| realized (points / USD) | −3.5 / −70 |

| closed | thesis | side | entry → exit | exit | points |
| --- | --- | --- | --- | --- | --- |
| 02:11 | T2 | LONG | 16533.00 → 16522.25 | invalidation | −21.5 |
| 09:32 | T4 | LONG | 16503.50 → 16474.75 | stop | −57.5 |
| 13:31 | T2 | SHORT | 16229.00 → 16162.50 | target | +133.0 |
| 15:11 | T3 | SHORT | 16222.25 → 16251.00 | stop | −57.5 |

The bias read the sell-off: SHORT/15m at 09:34 ("external/internal/BOS/
MSS all short with a 1-bar-old short displacement in expansion"), SHORT/1H
at 10:00 (MSS short, active leg −3.24 ATR) and short for the rest of the
session; the RTH proposals were SHORT 18 / LONG 7. One short reached its
target (+133); the long at the open (09:32, −57.5) and a short into the
afternoon bounce (15:11, −57.5) took the rest back. No position was open
at an opposing flip, so the bias-reversal exit had nothing to do.

## 5. The layered comparison

| | `72ea13c7` (start) | E (Eye) | B (Brain) | B′ (Brain, amended) | X (Execution) |
| --- | --- | --- | --- | --- | --- |
| `direction_accuracy_60m` | 0.359 | 0.381 | 0.445 | 0.502 | 0.465 |
| bias changes | — | — | 28 | 14 | 21 |
| LLM calls / coverage | 343 / 78 | 334 / 77 | 297 / 77 | 302 / 80 | 290 / 77 |
| sleeps | 5 | 0 | 4 | 14 | 11 |
| fills / closed | 9 / 9 | 14 / 13 | 10 / 10 | 10 / 10 | 7 / 7 |
| realized points | −103 | −109.5 | −121 | −139.5 | −146.5 |
| 13:4x short into the breakout | −20.5 | −45.5 | −32.5 | −58.5 | none |

## 6. Verdict

**What the build fixed, and can show.** The Eye describes the forming leg
with its size and age and both producers agree; the Brain carries an
explicit bias with a journaled basis, rule 4b holds every opportunity
inside it, the hysteresis halved the whipsaw and the 5m never sets it; the
controller sleeps again and calls fell from 343 to 290 with RTH coverage
25 of 25 throughout; a position the bias has turned against is closed at
market, and the 13:4x short that lost in every earlier run did not happen;
an expiry no longer spends the thesis; the runaway that could black out the
Brain for a session is bounded and its incidents say why. Every run
replays from the Eye alone; two are frozen as baselines.

**What it did not fix.** Realized P&L on the 3rd fell with every layer,
−103 → −109.5 → −121 → −139.5 → −146.5, while the direction reading rose.
The losses have one shape in every run with a bias: the entry is taken at
the extreme of the leg that set the bias (07:09 LONG at 16438.75 four
points under the overnight high; 09:47 LONG at 16413.5 six minutes before
the flush) and the loser is stopped by the next leg. In run X seven such
entries before 10:00 tripped the daily stop, and the day's correct thesis
could not be expressed at all. The reading itself is the model's and is
wrong more often than right at the segment level on this day (8 of 22);
on the 4th it was right all afternoon and the day closed flat.

**Left on the table, named and not tuned.** (1) The expression rule —
"after a BOS on the bias scale, the object that contains price" — is the
Brain's entry rule and the source of the entries at the extreme; changing
it needs a day other than the 3rd. (2) `opportunity_incoherent` went from
0–1 to 8–22 per run with the bias section: the model writes LONG
proposals whose target lies below the entry, and each one is a lost
expression of a possibly correct thesis. (3) `max_tokens` 32 768 is close
to what the model's reasoning uses on RTH inputs. (4) The daily stop's
2.5 % is reached by seven small losers; whether it should is the risk
policy's question, not this build's.

## 7. Commands

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider
.venv/bin/python -u -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high --label execution-layer
.venv/bin/python -u -m brain.scripts.run_llm_brain --client deepseek --broker sim --warmup-start 2021-12-28 --emit-start 2022-01-03T18:00 --end 2022-01-04T17:00 --max-llm-calls 1000 --reasoning-effort high --label execution-layer
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/ab572b09d59bc4d7 --run-dir outputs/brain_journal/3cc1bb402bb7b964 --run-dir outputs/brain_journal/07f897f45fab5025 --run-dir outputs/brain_journal/5c491789ccef7367 --run-dir outputs/brain_journal/72ea13c7fcbc1cff --write
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/ab572b09d59bc4d7
.venv/bin/python -m brain.scripts.replay_journal --run-dir outputs/brain_journal/5cfe65bcc480a032
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest brain/tests/test_regression_baseline.py -m research_orchestration -o addopts='' -p no:cacheprovider
```
