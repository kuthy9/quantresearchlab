# Trade-quality root cause — run `7ec17f066d232ba4` (2022-01-03, deepseek-flash@high)

Diagnosis only; no code changed. Every number below was computed from the
run's journal (`outputs/brain_journal/7ec17f066d232ba4`) and the 1m tape the
run read, with three throw-away scripts kept in the session scratchpad
(`trade_quality.py`, `trade_context.py`, `trade_counterfactual.py`).

Headline: 29 fills, 23 stops / 5 targets, −3 195 USD. The pipeline did what
the Brain asked; the Brain asked for the wrong trades. Ordered by weight:

1. **One directional thesis re-expressed all day through swapped objects.**
   The 4H structure read "short" on every call; the Brain kept a
   "HTF short governs / sell the rally" thesis for ~18 h while the 15m and 5m
   external structure read long at every submission. 27 of 47 orders were
   SHORT. The machine de-duplicates by object signature (direction + three
   entity ids), so changing one object makes a "new" trade: 47 orders, 47
   distinct signatures, but after 23 stops the next order came within 15 min
   in 14 cases, in the same direction in 11, sharing an object in 8, and
   within 1–2 min in 5. 42 of 47 submitted triplets were ACTIONABLE on the
   first call that named them (no DEVELOPING phase). 255 of 449 parsed
   replies (57 %) were ACTIONABLE; direction flipped between consecutive
   ACTIONABLE replies 49 times.
2. **Timeframe mismatch.** Thesis on 4H/1H; entry, invalidation and target on
   5m objects (invalidation objects: 5m ×22, 15m ×7, 1H/4H ×0; entries 5m ×22).
   Direction agreed with the 4H external structure in 19/29 fills, with the
   15m in 10/29, with the 5m active leg in 7/29. During RTH the 4H label
   stayed "short expansion" while price traded 16292 → 16498 and above the
   top of DR_4H_1 (the Brain's own 16:05 reading: "~1.15 of the range").
3. **Stops are at 1m-noise scale, and the budget makes them so.** 25/29 stops
   = a 5m pool's far edge + 1 tick (`stop.pool.far_edge`); median stop 8.5
   points = 0.58 × ATR(5m), 25th percentile 0.40. 22/29 entry objects were
   within 1 × ATR(1m) of price at submission (an at-market order with a
   2–10 point stop; 15/29 filled on the next bar). The gate's risk budget is
   equity × 0.005 = 500 USD → at 20 USD/point one NQ contract can hold at
   most a 25-point stop; RTH ATR(5m) was 26–50 points from 10:00 to 12:00.
   All five `position_size` vetoes fell in RTH (27–53-point stops); each was
   followed within minutes by the same trade with a nearer invalidation, and
   the Brain says so in step 13 ("Risk now fits the account: invalidation
   SWING_H_1H_1 (+2.66) is nearer than the vetoed BSL_15m_10 (+4.40)";
   "~$435 inside the 495.70 budget that vetoed the 53-point version"). 105 of
   449 step-13 traces reason about budget / veto / 1.5 / "nearer" / "tight".
   The prompt's execution-feedback section instructs exactly this ("name a
   nearer invalidation"). `min_reward_risk` 1.5 itself was not binding (4
   vetoes; median realised RR 2.9) — but step 13 makes RR the criterion, and a
   tight stop is what produces a high RR.
4. **The stop is not the Brain's invalidation.** Step 12 names event
   invalidations ("acceptance and hold above X", "a 15m bullish MSS"); the
   geometry rule converts the object into a touch stop one tick past its
   edge. 12 of 23 stop-outs were wicks: the 5m bar that stopped the trade
   closed back on the trade's side of the stop.
5. **Widening stops does not rescue the day** (so 3–4 are symptoms, not the
   root). Same fills and targets, stop = k × ATR(5m), first touch within
   240 min: actual → 23 stops / 5 targets, net −7.8 R; k = 1.0 → 23 / 5,
   −14.8 R; k = 1.5 → 18 / 8, −9.1 R; k = 2.0 → 16 / 9, −8.6 R. After a
   stop-out price went ≥ 1 R further against the trade within 60 min in
   17/23 cases; the target was reached later in only 4/23 (60 min) / 5/23
   (120 min). Mirroring every trade's direction gives 17 stops / 11 targets —
   both sides lose at this entry scale. Direction and timing are the root.
6. **RTH.** Not RTH as such: the same behaviour produced 13 stops / 5 targets
   (+455 USD) in the overnight range (ATR(5m) 4–9) and 10 stops / 0 targets
   (−3 650 USD) in a trend session (−90 points to 09:50, +200 to the close;
   ATR(5m) up to 50). The Brain faded the rally in 8 of 10 RTH trades and
   bought during the drop in the other 2 (09:40, 09:46; both stopped within
   4 min). The budget cap bound only in RTH.
7. **Cadence.** 520 calls in 1 380 bars: 233 of 516 inter-call gaps were one
   bar. 252 calls (49 %) were triggered only by a watched object's
   `position` flip (price crossing a 5m pool / FVG the Brain named), with no
   new Eye evidence; the rest mostly by 5m `displacement_observed` (129),
   `level_reached` (82), `acceptance_confirmed` (56). `continue_active` was
   true in 449/449 replies; `watch_next` was never emptied; the idle archive
   fired 3 times all day (the Brain proposes an opportunity in 57 % of
   calls, which resets the idle rule). Latency (p50 41 s, p90 76 s) had no
   effect on backtest fills (simulated clock); live it would delay 54/516
   calls by ≥ 1 bar, and 27/29 limits were still reachable two bars later
   within the TTL. TTL expiries 10, `signature_changed` replacements 6,
   `plan_dropped` 2. Cadence costs money (8 USD/day) and drives churn (a
   new reply every 1–3 min, each allowed to name a "new" trade); it is not
   the P&L root.
8. **Responsibilities.** The gate never moves a price (correct). But the
   Brain is told, via `last_veto` guidance in the prompt, to satisfy the
   gate's RR and budget by re-choosing the invalidation — the Risk Manager's
   constraint reshapes the thesis. The geometry module has no channel for
   the Brain's invalidation *type* (touch vs close-beyond) or *timeframe*.
   Sizing by dollar risk with no leverage bound let a 2.25-point stop carry
   5 contracts (09:05; 1.6 M USD notional on a 100 k account).

What this implies (not implemented; decisions for the owner):

- A thesis identity with a lifecycle (direction + governing scale +
  invalidation object + destination), one expression at a time, a cool-down
  or hard close after a stop, and a cap on expressions per thesis — the
  signature de-dup is at the wrong level.
- The invalidation object must live on the thesis's scale (or one below),
  and the Brain should state whether invalidation is a touch or a close
  beyond the object; a stop at "5m pool + 1 tick" for a 4H thesis is not an
  invalidation.
- `position_size` must not feed back as "name a nearer invalidation": it is
  "this instrument is too large for this stop" — skip, or trade MNQ
  (2 USD/point) where the same budget holds a 250-point stop.
- The gate should bound notional / contracts per stop width as well as
  dollar risk.
- The relation-flip UPDATE trigger and the never-empty `watch_next` keep
  the Brain re-reasoning every 1–3 bars; a trend reading should be
  re-examined on the thesis's scale, not on every 5m pool crossing.
