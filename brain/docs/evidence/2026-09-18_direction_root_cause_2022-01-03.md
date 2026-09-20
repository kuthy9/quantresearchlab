# Direction — what moves the Brain's reading, and why it did not move with the tape

Run `72ea13c7fcbc1cff` (2022-01-03, thesis lifecycle + Risk v2, receipt
[2026-09-18_risk_v2_day_run_2022-01-03.md](2026-09-18_risk_v2_day_run_2022-01-03.md)).
Question: what turns the Brain from bearish to neutral to bullish, and why
did that not happen once the day's rise began. Diagnosis only; nothing was
changed. Times below are New York. Evidence: the run's journals (every LLM
input and reply, every state revision, every order), the Eye source, and
session scratch scripts (`direction_timeline.py`, `reply_excerpts.py`).

## 1. What can move the direction

Nothing in code. `BrainState` has no bias, regime or direction field; the
reading exists only as text (`market_understanding`,
`active_expectation.thesis`) and as `opportunity.direction`. The reducer's
rule 3 checks one thing about a reply that says `understanding_holds:
false` — that the text differs from the prior state
([reducer.py](../../core/reducer.py)). Verdicts are bookkeeping: no count of
contradictions, no threshold, no transition. The Sleep Controller decides
*when* the Brain is asked (any 15m/1H/4H reaction, the four 5m kinds,
relation changes on 15m and above) and carries no direction
([sleep_controller.json](../../configs/sleep_controller.json)).

So the transition is the model's own judgment on every call, framed by the
prompt: steps 5–9 (alignment → assessment → counter-delivery → restructure →
expectation), *"A counter candle is not delivery; delivery is displacement
that the Eye reports as such (a displacement, an MSS, a qualified BOS on the
relevant scale)"*, and *"The thesis is judged on its governing scale"*.

The journal shows the rule the Brain actually applied. It rewrote its
understanding 41 times in 343 calls. Every rewrite that changed direction
cites an Eye structural event on the scale it called governing or above
(1H MSS long 07:00, 15m MSS short 08:45, 1H MSS short 10:00, 5m MSS short
11:40, 1H displacement long 13:00, 15m MSS long 13:45); every refusal to
change cites the absence of one there:

| call | Eye 15m | Brain's reply (steps 6–8) |
| --- | --- | --- |
| 05:00 | MSS long | "deep pause, not failure — no 1H bullish BOS"; "sub-scale: no bullish displacement or MSS on the 1H" |
| 05:15 | acceptance long | "a deferral, not a failure: the 1H/4H short structure … 1H expansion-short are all unchanged" |
| 06:30 | 3 × acceptance long | "strain/pause, not failure: no 1H bullish BOS … the acceptances are sub-scale, not displacement" |

The 15m MSS long and the seven 15m acceptances of 05:00–06:30 were all
verdicted CONTRADICT and changed nothing.

## 2. The session, scale by scale

| window | price | 4H label | 1H label | 15m label | Brain |
| --- | --- | --- | --- | --- | --- |
| 18:00–06:36 | 16356 → 16452 | external short; **expansion short** from 02:00 | external short, protected high 16464 intact, **expansion short, displacement 0.64** (frozen 18:01 → 10:00) | external long from 18:45; MSS long 05:00; 7 acceptances long | SHORT T1–T7, 11 rewrites all "4H and 1H remain structurally short"; stops 19:31, 21:34, 05:01 ×2 |
| 07:00 | 16447 (5 under the overnight high) | same | **MSS long** (internal long) | long | flips LONG T1 in the same call; stopped 07:46 as price drops into the open |
| 08:30–09:02 | 16431 → 16373 | same | reversal attempt | **MSS short** 08:45 | SHORT T2 ACTIONABLE 08:45; target 08:58 (+31, the only winner); sleeps |
| 09:10–09:45 | 16391 → 16484 → 16444 | same | 09:39 acceptance above 16464 → `external=None`, **balance** | BOS long 09:45 | LONG T1–T4 (two TTL expiries, one RR veto) |
| 09:48–10:10 | 16444 → 16292 → 16364 | same | **MSS short** 10:00 (last 1H structural event of the day) | structure break short, internal short | SHORT T5, LONG T6 (5m), SHORT T6 |
| 10:15–13:45 | 16292 → 16444 → 16374 → 16460 | **expansion short** all day | balance, internal short, last MSS short | `external long / internal short / MSS short` until 13:45 | NONE → LONG T7/T8 (10:45; T8 filled 16391.5, close-beyond exit 11:41) → SHORT T9 (11:40 on a 5m MSS short; 12:00 "down displacement on 5m, 15m and 1H"; filled 16411.25, stopped 12:47) → LONG T2 (13:00) → SHORT T3 (13:14: "the 4H bearish delivery dominates"; filled 16450.25 at 13:43, invalidated 13:46 at 16460.5 as the 15m broke out) |
| 13:45–16:36 | 16460 → 16508 | same; MSS long at **17:00** | balance; acceptance long 15:00, 16:00 — no BOS, no MSS | **MSS long 13:45**, acceptances 14:00 ×3, qualified BOS 16:30 | LONG T4 from 14:05 (DEVELOPING) / 14:20 (ACTIONABLE): limit 16449.25 at FVG_5m_8, TTL 15 — 1m low 16453.00 (missed by 3.75); two RR vetoes 15:30/15:35; limit 16449.25 again 15:44 — low 16450.25 (missed by 1.00); 16:35 `expressions_exhausted`, 16:36 `thesis_closed` |

Eight direction changes between 09:48 and 14:05. The one thesis that had the
day's direction after the breakout never filled.

## 3. Why the conditions never fired on the rise

On every scale the Brain treats as governing, the Eye's labels said "short"
or "balance" for the whole rise. Four mechanisms, each in code:

1. **The displacement score is frozen between displacement events of that
   scale.** `displacement_score` changes only on a `DISPLACEMENT_OBSERVED`
   event of the same timeframe
   ([market_state.py:2646-2663](../../../eyes/core/market_state.py)); the
   live computation exists for 5m only (`_displacement`, line 5776). The 1H
   "0.64, short" printed at 18:01 stood until 10:00 — sixteen hours and 100
   points — and the Brain quoted it in nearly every overnight reply as proof
   the 1H down-leg was live.
2. **The delivery phase describes the leg that just ended.**
   `active_leg_direction` is the direction of the last
   `structural_leg_created` (lines 5960-5966), and a leg is created when its
   *end* swing is confirmed, `swing_span` = 2 bars after the pivot
   ([structure.py:51-56](../../../eyes/core/structure.py),
   [semantic_event_emitter.py:2467](../../../eyes/core/semantic_event_emitter.py)).
   `_delivery_phase` (lines 1607-1632) returns EXPANSION whenever that leg
   points the external way. The 4H leg into the Dec 31 low was confirmed at
   02:00 → "4H expansion short" for fifteen hours while price rose 16398 →
   16500. The 1H leg into the 03:00 low was confirmed at 05:00 → "1H
   expansion short" at the exact bar the 05:00–06:30 leg up began.
3. **A broken protected swing leaves the scale directionless until two
   aligned swings confirm.** The 09:39 acceptance above SWING_H_1H_1 (16464)
   set `external_direction=None, protected_swing_intact=False`
   (market_state.py:2607-2620); a new direction needs a CONFIRMED structure —
   an aligned confirmed high *and* low (structure.py:777-783), each confirmed
   two 1H bars after its pivot — and an MSS core needs a BOS with OPPOSED
   scope against a confirmed opposite structure (structure.py:871-875,
   emitter 3253-3261). After the 10:00 MSS short no 1H structure was
   confirmed in either direction before the close, so the 1H emitted no BOS
   and no MSS from 10:00 to 17:00 — only `acceptance_confirmed long` at
   15:00 and 16:00, which the prompt does not count as delivery — and its
   phase read BALANCE (external None inside an active dealing range) for the
   entire rally.
4. **The 4H is one bar per four hours.** Its MSS long came at 17:00, with
   the bar that opened at 13:00.

The Brain was told to judge the thesis on those scales and to accept only
Eye-reported delivery there as counter-delivery. It complied.

## 4. What is the Brain's own doing

- It anchors on the 4H. `governing_timeframe` was 1H in 46 ACTIONABLE
  replies and 15m in 65, never 4H — yet "4H still governs" opens most RTH
  replies, and at 13:14 it shorted a 15m premium because "the 4H bearish
  delivery dominates", three minutes of trading before the 15m broke out.
- It reads `external long / internal short` as "counter-trend bounce" every
  time (10:15, 11:40, 12:00, 13:14) rather than as a base.
- It never uses the tape's own drift. `session_open` 16356 and the close are
  in every input; no reply cites the 90-point overnight rise or the 150-point
  recovery from 16292 as evidence. The prompt's "objects only, never write a
  price" pushes that way.
- With the direction finally right (14:05), it expressed it as a retracement
  limit at a 5m FVG, twice, in a tape that did not retrace; `max_expressions
  2` then closed the thesis. The lifecycle rule was built for stop-outs; it
  also ends a thesis that merely expired.

## 5. Not decided here

Candidates, for the owner to weigh — each changes a different component:

- Eye: carry the age of `displacement_score` and of the active leg (bars
  since the event), or decay them; report the forming leg beside the
  confirmed one; expose "higher highs and higher lows since the reset" so a
  directionless scale is not read as balance.
- Brain / prompt: let the 15m govern *direction* with 1H/4H as location;
  count N aligned acceptances on the governing scale as delivery; do not
  read a stale expansion label as a live leg; name the session drift as
  evidence.
- Execution: a continuation entry (stop order beyond the governing-scale
  BOS) when the retracement limit expires, or an expiry that does not spend
  an expression.

## 6. Commands

```bash
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/72ea13c7fcbc1cff
```

The per-call timeline (Eye labels per scale beside the Brain's opportunity),
the 1H/4H event stream with verdicts, and the reply excerpts at the flip
points were produced by session scratch scripts over
`brain.core.journal.JournalReader`; they are not part of the repository.
