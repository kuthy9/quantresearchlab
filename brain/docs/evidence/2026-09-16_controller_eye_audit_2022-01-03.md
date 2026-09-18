# Controller and Eye audit, the cost cut, and the third real run — 2022-01-03

Follows [2026-09-16_llm_brain_audit_2022-01-03.md](2026-09-16_llm_brain_audit_2022-01-03.md).
Journals: `outputs/brain_journal/6960dbba4cae8b97/` (run 2, before) and
`outputs/brain_journal/46a8bcf33f5f1f04/` (run 3, after); gitignored, every
number below is read from them or from the audit scripts' output.

## 1. The Eye on the 2022-01-03 Globex session (1380 bars, 18:00 → 17:00 NY)

Checks against the raw 1m tape, every 30 minutes and at every bar:

| check | result |
| --- | --- |
| bar continuity in the emit window | 1380 bars, 0 non-one-minute steps, 0 duplicates |
| `MarketSnapshot.price` | = the last real 1m close on 46 of 46 checks |
| active 5m and 15m FVGs vs the three-candle geometry recomputed from resampled bars | 405 of 405 (5m), 309 of 309 (15m) match on direction, lower and upper |
| liquidity candidates (5m / 15m / 1H) sit on a real 1m high or low of the previous 10 days | 2987 of 2987 (`confirmed_swing` and `formed_liquidity_pool` sources) |
| RTH session high / low vs the bars since 09:30 | 26 of 27 (the one miss is the 09:30 boundary of the check itself) |
| causality | every timestamp the Brain serializes is `<= known_at` (`assert_causal`, on every call) |

No delay, omission or error was found in what the Eye publishes to the Brain.

## 2. The Sleep Controller

Sharp moves: 81 bars where the next 15 bars' range exceeds 2.5 × ATR(14), 25
of them in RTH. A move counts as covered when a wake-set event fires within
[-3, +10] minutes of its start.

| wake set | wake bars | covered | RTH |
| --- | --- | --- | --- |
| 15m / 1H / 4H reactions only | 90 (7 %) | 58 / 81 | 17 / 25 |
| + `mss_core_confirmed`, `qualified_bos`, `sweep_confirmed` on 5m (the set until this audit) | 111 (8 %) | 62 / 81 | 19 / 25 |
| **+ `displacement_observed` on 5m (now)** | **192 (14 %)** | **77 / 81** | **24 / 25** |
| + `acceptance_confirmed` on 5m | 209 (15 %) | 78 / 81 | 24 / 25 |
| + `fvg_first_retest` or `structure_break` on 5m | 220 / 214 (16 %) | 78 / 81 | 24 / 25 |

The uncovered moves under the old set — including the day's two largest,
10:04 NY (67 points) and 10:36 NY (55.5 points) — carried
`displacement_observed@5m` and formation events only. One kind closes the
gap; the next candidates buy one more move for 10 % more wake bars.

UPDATE was the looser side: 22 % of bars carried a 5m+ transition, most of
them bookkeeping (`level_touched`, `liquidity_level_created`,
`swing_confirmed`, `structural_leg_created`, `delivery_phase_updated`), and
29 of run 2's 79 calls carried no new evidence at all. The controller
(schema 2) now names the bookkeeping kinds once and uses the list twice:
they never wake, and awake they never trigger a call — they ride as evidence
with the next reaction. Watched-object relation changes still trigger a call.

## 3. Where the tokens went, and the cut

Run 2's input was 48 k characters a call: `prior_state` 34 k (evidence ledger
20 k, `object_registry` 11.6 k), `price_relations` 6.9 k for 91 objects of
which 28 sat within 3 ATR. Now: no registry in the input, twelve items per
evidence list with notes cut at 160 characters, relations within 4 ATR plus
every object the state names, `reasoning_effort` configurable.

## 4. Run 3 — DeepSeek, `reasoning_effort=low`, the new controller and input

| | run 2 (before) | run 3 (after) |
| --- | --- | --- |
| window, budget | 2022-01-03 09:00–12:00 NY, 80 calls | same |
| episodes | 1, never slept | 6; 5 archived — 3 on the model's own `continue_active=false`, 2 by the idle rule |
| controller | WAKE 1 · UPDATE 78 · TICK 102 · asleep 0 | WAKE 6 · UPDATE 72 · TICK 90 · asleep 13 |
| calls / budget exhausted at | 79 / 11:42 NY | 78 / 11:51 NY |
| repairs | 6 (3 empty replies, 2 stale evidence ids, 1 missing key) | 5 (all a verdict on an id from `prior_state.evidence`; the prompt now forbids it) |
| input characters, median | 48 007 | 17 824 |
| prompt tokens (cache hits) | 1 318 713 (275 840) | 607 633 (185 856) |
| completion tokens | 1 012 334 | 425 305 |
| latency per reply, median / max | 56.1 s / 137.1 s | 26.1 s / 65.4 s |
| wall clock | 86.7 min | 40.5 min |
| confidence | MEDIUM 49 · LOW 23 | MEDIUM 22 · LOW 53 |
| opportunities proposed | 20 (6 ACTIONABLE), 5 survived geometry | 11 (all DEVELOPING), 0 survived (`opportunity_incoherent` 11) |
| deferred bookkeeping evidence delivered | — | 5 calls carried one deferred item |

Tokens fell by 57 % and latency by half; the cycle SLEEP → WAKE → … → SLEEP
ran five times on the real model, twice through the idle rule. But run 3
proposed no coherent opportunity and read itself as LOW confidence on 53 of
75 replies. Three things changed at once (the input, the controller, the
reasoning budget), so which one cost the opportunities is not established by
this run; the LONG / SHORT geometry failures are the same kind run 1 showed.
`main_brain.json` therefore keeps `reasoning_effort` unset (DeepSeek's
default, high) and the input and controller changes; the next measurement
that separates the effects is the same window at high effort.

Replay of run 3 (`replay_journal.py --run-dir outputs/brain_journal/46a8bcf33f5f1f04`,
after the code of this receipt): **replay OK — 6 episodes, 78 llm calls,
168 revisions, 0 trade records reproduced**. Run 2 (`6960dbba4cae8b97`)
predates the input shape of this receipt and no longer replays sha-equal.

## 5. The stack behind the Brain

The Risk gate and the order machine ([../../execution/docs/README.md](../../execution/docs/README.md))
were not exercised by run 3 (no opportunity survived geometry). Their
receipts are the tests: 27 execution tests (contracts, the simulated
broker, the plan builder, the machine, the IBKR adapter against a fake, the
end-to-end stack on the synthetic Eye with replay) and 13 risk tests, plus
an echo run with `--broker sim` on this window whose journal replays with
the machine (24 episodes, 0 trade records).
