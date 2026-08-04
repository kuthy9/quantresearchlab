# v2 current-revision validation record

Revision date: 2026-07-25

## Gate and identities

The mandatory trading-logic/code self-review was completed before each new test
stage. The final automated suite passes 58/58 tests. The causal OHLCV source is
the strict previous-session front with SHA-256
`5057fe574b82b26e3fe8a7798607a177b847876fee7f5ef41a938c7c87499bfc`.
The latest OHLCV holdout and the August-December 2024 MBO holdout remain
unrevealed.

## Clock and source continuity

The production reader completed the full 2022-2023 calibration stream:
750,915 bars including warmup and 707,580 calibration decision minutes. Explicit
equity-index holiday rules include abbreviated post-Thanksgiving trading through
13:15 New York time, July 3 early closes, historical maintenance pauses, Good
Friday exceptions, and Juneteenth.

The full 2017-2026 source audit is intentionally not marked passed. Under the
current corrected clock it identifies 12,194 absent registered trading minutes
and 99 open-market gaps longer than
the five-minute no-trade synthesis cap, including source-wide outages. Unknown
gaps remain hard failures rather than being relabeled as exchange closures or
filled with invented prices. This is a source-completeness limitation, not an
unresolved timestamp/DST bug.

## MBO execution reality

The DBN dependency is installed. Deterministic random-month selection chose July
2024 from the unsealed June-July development interval. Exact L3 reconstruction
processed 327,177,379 vendor events across 27 partitions and produced 31,275
valid decision-minute book rows. A second, still-unsealed June development
month was then materialized without changing the model or thresholds:
242,224,864 selected complete vendor events across 25 hash-verified partitions
produce 27,479/27,479 valid minute rows.

June contains the NQM4/NQU4 roll in one UTC partition and four Sunday feed
starts. Static review found and corrected two source-protocol defects before the
replay: vendor packets can contain multiple books and calendar spreads can have
zero or negative prices; Sunday snapshots arrive as a clear-only packet followed
by a separate packet of snapshot-flagged adds. Full-packet boundaries are now
recovered before selected-book replay, while nonpositive prices on the selected
outright still fail closed. The roll partition is 1,380/1,380 valid, all four
Sunday partitions pass, no value is clipped or imputed, and the suite passes
58 tests.

June median spread is 0.50 points, p95 is 0.75, p99 is 1.00, and the maximum
16.00 occurs at an 08:30 release minute. Median quote age is 0.101 seconds and
p99 is 3.157 seconds. July median spread is 0.75 points, p95 is 1.00, p99 is
1.25, and median/p99 book age is 0.083/2.891 seconds. No constant execution
fallback was used in either month.

The streaming reconstruction and full-month replay run on this machine. The
constraint is limited concurrent/system memory headroom, not inability to read
MBO. Displayed BBO/depth still cannot identify hidden liquidity, queue position,
latency, or impact beyond visible size; unknown impact remains vetoed rather
than estimated from a constant.

## Belief calibration

The frozen 2022-2023 path stream contains 10,701 paths, of which 10,695 have
eligible target/invalidation/deadline outcomes:

| Playbook | Eligible episodes | Target-first rate | Raw Brier | In-sample calibrated Brier* |
|---|---:|---:|---:|---:|
| displacement first pullback | 5,102 | 0.625 | 0.264 | 0.234 |
| failed auction value return | 3,198 | 0.548 | 0.264 | 0.247 |
| liquidity sweep reversal | 2,395 | 0.605 | 0.278 | 0.239 |

\*Descriptive only; it is not used as independent validation.

Calibration uses fixed quantile bins, Beta(1,1) smoothing, weighted monotone
PAVA, no PnL optimization, and no threshold search. The provisional
`executable_probability=0.66` was deterministically unreachable after
calibration because every map ended below it. It is replaced by the semantic
0.50 target-first gate; net action value remains the decision layer's job.

On both June and July paths, calibrated Brier is lower than raw Brier for all
three playbooks. Across 959 paths the aggregate Brier improves from 0.266 to
0.239. This supports probability-scale transfer for the registered
target-before-original-invalidation label, but does not prove a trading edge or
alignment to the managed-position return.

## Two-month development replay

The calibrated July replay contains 31,275 decisions. Model actions are 513
`enter`, 847 `wait`, 2,086 `hold`, 27 `protect`, and 27,802 `abstain`. The
independent risk layer approves 80 entry attempts and vetoes 433; the leading
vetoes are account/contract risk (393), cost (35), spread (18), invalid
structural stop (13), fillability (5), and deadline (1).

Order accounting is exact: 80 approved next-bar limit attempts equal 53 fills
plus 27 unfilled/expired attempts, with no end-of-window pending order. All 53
fills close in the interval. There are 40 stop exits and 13 target exits:

| Playbook | Closed trades | Net wins | Net R |
|---|---:|---:|---:|
| displacement first pullback | 22 | 8 | +2.875 |
| failed auction value return | 26 | 8 | -4.371 |
| liquidity sweep reversal | 5 | 1 | -0.433 |
| **Total** | **53** | **17** | **-1.929** |

Maximum cumulative drawdown is approximately -9.787R. Therefore profitability
validation fails for this month. The action/risk/execution/feedback plumbing is
empirically active, but this revision is not ready for capital.

The same frozen configuration and 45-day warmup on June contains 27,479
decisions. The model proposes 157 enters; risk approves 46, of which 37 fill
and close and nine expire unfilled. There are no pending orders. Thirty-two
trades stop and five reach target, producing -13.602R and a -14.523R maximum
cumulative drawdown:

| Playbook | June trades | June net R | July trades | July net R | Combined net R |
|---|---:|---:|---:|---:|---:|
| displacement first pullback | 24 | -4.339 | 22 | +2.875 | -1.464 |
| failed auction value return | 12 | -8.193 | 26 | -4.371 | -12.565 |
| liquidity sweep reversal | 1 | -1.070 | 5 | -0.433 | -1.503 |
| **Total** | **37** | **-13.602** | **53** | **-1.929** | **-15.532** |

Across both months, 126 approved attempts partition exactly into 90 filled and
closed trades plus 36 unfilled/expired attempts. There are no pending orders,
no invalid original stop geometries, no final stop that expands beyond original
risk, and no ambiguous same-bar trade. Every decision uses reconstructed MBO;
spread, cost, and fillability are nonconstant.

The combined 90-trade win rate is 28.9% and maximum cumulative drawdown is
-21.402R. Trade probability/net-R correlation is 0.075 and utility-advantage/
net-R correlation is 0.064. These small development samples cannot estimate a
stable correlation, but they provide no evidence that the current decision
ranking delivers net utility.

## Objective and component-link audit

Runtime transmission is intact: eyes feed beliefs, beliefs feed plans and
utilities, risk vetoes model entries, approved orders either fill or expire,
positions produce hold/protect/exit feedback, and original structural
invalidations remain frozen. The failure is semantic rather than a broken
ledger.

Path calibration freezes target-before-original-invalidation when a sequence
first completes. The model may enter the same setup later after its plan
changes, and the managed position may exit through `protect` before either
frozen endpoint. Seven of 37 June action plans and 11 of 53 July action plans
differ from their setup's frozen path plan. In each month, eight trades stop
while that setup's frozen path later resolves target. Thus a well-calibrated
path probability is not yet a calibrated expected return for the actual
enter/hold/protect/exit policy.

The action margin also compares playbook-specific plan variants. In the audited
2024-06-02 18:05 decision, DFP-short and LSR-short share entry and target, with
enter utilities 1.197R and 1.156R. Their 0.041R difference causes `abstain`
despite same-direction agreement. Because their invalidations differ, the next
revision needs an explicitly preregistered action-equivalence and plan-arbitration
rule; it must not be patched from the revealed outcome.

## Warmup, overlap, and complexity

A three-day warmup improves 1H/5m/1m readiness, creates setup funnels and path
tests, and is better than one day for continuity. It still leaves 4H unready in
the bounded comparison and produces no executable action, so it is not promoted
as an optimized parameter. The full replay retains 45 warmup days.

Calibration overlap is material. Failed-auction and liquidity-sweep-reversal
paths co-start 590 times; 582 share the same frozen plan and 589 share the same
outcome. The same relationship persists in development: 25/26 same-direction
June co-decisions and 28/29 July co-decisions share the same plan. They also
have non-overlapping episodes, so these months do not authorize a merge.
Rolling incremental attribution must decide whether to retain both in the next
major registry. No fourth playbook is admitted.

The system avoids state-combination strategies and uses only three ordered
hypotheses plus compact monotone maps. The prior 0.66 double gate was an
overengineered responsibility overlap and has been corrected. Adding more
playbooks now would increase attribution conflict and overfit risk.

## Visual audit

Decision charts show 4H/1H/5m/1m causal candles; directional displacement,
efficiency, swing progression, acceptance/rejection, impulse, first pullback,
reacceptance, compression, 1m path/counter-pressure; event order and
persistence; selected draw IDs; calibrated beliefs/phases; plan provenance;
MBO BBO/depth; action utilities, reasons, and vetoes; and the AI primitive
review section. Only the two most recent event labels are expanded on price to
bound clutter. Off-screen plan levels use edge badges and no longer compress
the 1m/5m axes.

Future candles appear only in a separately sealed reveal image and JSON record
bound to the decision, protocol, config, and model-code hashes. AI review files
may propose only registered, unvalidated sequence primitives; they cannot
supply actions, outcomes, profit labels, or model authority.

The June images confirm that off-screen targets use edge badges rather than
compressing the 1m/5m axes, and that decision and future remain separate. They
also expose the playbook-plan competition described above. The AI primitive
section is present but no AI review was supplied in these runs. The renderer is
functionally complete for the registered fields; audit coverage is not yet
complete because examples are not systematically stratified across every
action, veto, stop, target, and protect outcome.

## Claim boundary

Software, causal plumbing, MBO execution inputs, risk vetoes, sequential
feedback, and the visual audit workflow are implemented and tested. The current
result does **not** establish profitability or absence of overfitting. It
provides evidence of path-label calibration transfer alongside negative net
utility, an objective mismatch between path calibration and the managed trading
policy, playbook-plan competition, substantial playbook overlap, and incomplete
source-wide 2017-2026 continuity. Sealed holdouts remain reserved for a later
genuinely frozen candidate.
