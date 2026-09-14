# Setup first-passage gate — receipt, 2026-09-14

Spec: [2026-09-13-setup-first-passage-gate-design.md](../specs/2026-09-13-setup-first-passage-gate-design.md)
(with its §9 correction). Plan: [2026-09-13-setup-first-passage-gate.md](../plans/2026-09-13-setup-first-passage-gate.md).

## Verdict: FAIL

None of the ten judged cells — {K0, K1, K2} × {zone_return, pool_reversal}
minus K1:zone_return × {logistic, lightgbm} — meets the spec §6 conditions.
No cell's pooled session-block interval for Δ log-loss (M₁ − M₀) lies below
zero; Holm rejects nothing at α = 0.10; the best cell beats the geometry
baseline on two rolling folds of five. Knowing the Setup does not change
the probability that its target is reached before its failure boundary,
beyond what the two distances, the remaining session and the volatility
already say. Under the spec, the conditional proposer, the Setup × target
hypothesis pool and event-driven orchestration do not start.

## The run

| item | value |
| --- | --- |
| run id | `796a4742275648f7` (`outputs/setup_gate/796a4742275648f7/`, ignored directory) |
| source | `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`, the path `configs/data_splits.json` binds |
| atomic identity | `f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c`, unchanged |
| Eye pass | 23 Globex-week blocks, 2022-01-03 → 2022-06-06, warmed seven days each (spec §5.1), `build_gate_blocks.py --record-paths`; 6 workers, 330.7 min wall, 63–91 min per block; 150,690 clocks |
| path log | 38,153 new-step rows, `context_found` on 100.0 % |
| labels | 26,046 labelled; dropped `past_failure` 7,491 (price already beyond the boundary at the step, mostly terminal steps), `no_target` 4,594 (no unswept level ≥ 1R in direction), `no_tape` 14, `no_failure` 8 |
| labelled instances | three-way target / failure / censored 0.215 / 0.735 / 0.049; `hit_target` 0.215; same-bar 2.3 %; d_target p10/50/90 0.13 / 0.48 / 1.63 ATR₆₀, d_failure 0.02 / 0.13 / 0.52; time to resolve 1 / 5 / 78 min |
| sessions | the blocks emit 114 sessions (through 2022-06-10); primary fold train 90 → OOS 20, five rolling folds 60 → 10 step 10; embargo 240 min |
| M₀ | 9 geometry columns (`setup_features.GEOMETRY_COLUMNS`) |
| M₁ | M₀ + 39–46 Setup columns (`setup_matrix`: context kind, zone kind, entry mode, source scale, step strengths and reasons, path age, zone width, penetration, reclaim / hold margins, structure alignment on 5m / 15m / 1h) |
| M₂ | M₁ + the 150 Eye-state components; reported only |
| models | logistic (C ∈ {0.01, 0.1, 1.0} by blocked, time-purged CV on the primary training rows) and LightGBM binary (400 trees, early stop on a time-purged tail); ±10 σ clip; `brain/research/gate_models.py` |
| gate | `python -m brain.scripts.setup_gate --run-id 796a4742275648f7`, under two minutes |
| tables | `results.csv` sha256 `785dded3…` (60 rows), `verdict.csv` `15df6786…` (12), `descriptives.csv` `0cf4fb96…` (6), `instances.parquet` `29989800…` (38,153) |

## Cells

| cell | instances | primary train / OOS | hit_target share | three-way t / f / c | d_f / d_t median (ATR₆₀) | judged |
| --- | --- | --- | --- | --- | --- | --- |
| K0:zone_return | 1,731 | 1,359 / 310 | 0.258 | 0.261 / 0.660 / 0.079 | 0.33 / 0.71 | yes |
| K0:pool_reversal | 7,698 | 6,098 / 1,335 | 0.157 | 0.161 / 0.834 / 0.005 | 0.05 / 0.35 | yes |
| K1:zone_return | 270 | — | 0.281 | 0.281 / 0.656 / 0.063 | 0.26 / 0.59 | **no** — instances on 104 sessions, 110 needed for one fold |
| K1:pool_reversal | 3,546 | 2,802 / 617 | 0.252 | 0.249 / 0.726 / 0.025 | 0.14 / 0.43 | yes |
| K2:zone_return | 1,452 | 1,142 / 244 | 0.239 | 0.244 / 0.722 / 0.034 | 0.19 / 0.52 | yes |
| K2:pool_reversal | 1,386 | 1,129 / 221 | 0.203 | 0.209 / 0.570 / 0.222 | 0.52 / 1.10 | yes |

Every judged cell clears the 5 % share guard on both outcomes and the
200-row OOS floor.

## Primary fold: Δ log-loss, M₁ − M₀ (negative is better)

Session-block bootstrap intervals in brackets (2,000 draws); rolling folds
beating M₀ out of five.

| cell | lightgbm | logistic |
| --- | --- | --- |
| K0:zone_return | −0.0024 [−0.0055, +0.0005] 2/5 | −0.0019 [−0.0104, +0.0055] 1/5 |
| K0:pool_reversal | +0.0002 [−0.0032, +0.0036] 1/5 | +0.0017 [−0.0016, +0.0047] 3/5 |
| K1:pool_reversal | +0.0030 [−0.0028, +0.0084] 3/5 | −0.0033 [−0.0087, +0.0017] 4/5 |
| K2:zone_return | −0.0005 [−0.0044, +0.0033] 4/5 | +0.0039 [−0.0076, +0.0153] 0/5 |
| K2:pool_reversal | −0.0013 [−0.0062, +0.0041] 3/5 | −0.0205 [−0.0850, +0.0264] 2/5 |

**Why every cell fails.** Condition 1 (interval below zero after Holm)
fails everywhere: the smallest one-sided p is 0.059 (K0:zone_return,
lightgbm), unadjusted. Two cells reach four of five rolling folds
(K1:pool_reversal logistic, K2:zone_return lightgbm) and both have
intervals covering zero and primary deltas of −0.0033 and −0.0005 on
baselines of 0.48 and 0.51 — under 1 % of the loss. The one large delta,
K2:pool_reversal logistic −0.0205, has an interval of ±0.06 on 221 rows and
rolling folds of +0.041, +0.008, +0.005, −0.010, −0.034: the linear model
on 46 columns is unstable on that cell, not informative.

## Reported, not judged

**What M₀ and the analytic ratio do.** Primary-fold losses against the
entropy of the training `hit_target` prior:

| cell | prior | analytic `d_f/(d_t+d_f)` | M₀ logistic / lightgbm | M₁ logistic / lightgbm | M₂ logistic / lightgbm |
| --- | --- | --- | --- | --- | --- |
| K0:zone_return | 0.571 | 0.549 | 0.543 / 0.558 | 0.541 / 0.555 | 0.589 / 0.564 |
| K0:pool_reversal | 0.434 | 0.437 | 0.434 / 0.431 | 0.436 / 0.431 | 0.433 / 0.430 |
| K1:pool_reversal | 0.565 | 0.481 | 0.476 / 0.477 | 0.473 / 0.480 | 0.490 / 0.484 |
| K2:zone_return | 0.550 | 0.510 | 0.505 / 0.510 | 0.509 / 0.510 | 0.540 / 0.513 |
| K2:pool_reversal | 0.504 | 0.553 | 0.538 / 0.535 | 0.517 / 0.534 | 0.528 / 0.535 |

Three readings. First, the zero-parameter driftless ratio is within 0.01
of the fitted geometry model on four of five cells: what the geometry
model knows is the ratio of the two distances, and it learns nothing
beyond it. Second, on K0:pool_reversal — the largest cell, the sweep bar
itself, with the failure boundary a median 0.05 ATR₆₀ away and 83 % of
instances failing at a median of one minute — neither the ratio nor M₀
beats the class prior at all (0.434 vs 0.434): at that clock the outcome
is the next bar's sign. Third, adding the Eye state (M₂) costs 0.0005–0.048
on every cell but K0:pool_reversal; 150 columns on a few hundred rows
overfit, as they did in the information-gain gate.

**Accuracy** (argmax, primary fold): M₀ 0.713–0.822, M₁ within ±0.02 of
it on every cell.

**Time to resolve and MAE by outcome** (minutes, ATR₆₀; p10 / p50 / p90):

| cell | target: time | target: MAE | failure: time | failure: MAE | censored share |
| --- | --- | --- | --- | --- | --- |
| K0:zone_return | 4 / 24 / 125 | 0.01 / 0.12 / 0.31 | 2 / 16 / 96 | 0.15 / 0.37 / 0.66 | 7.9 % |
| K0:pool_reversal | 1 / 4 / 26 | 0.00 / 0.02 / 0.08 | 1 / 1 / 8 | 0.04 / 0.11 / 0.25 | 0.5 % |
| K1:zone_return | 5 / 17 / 57 | 0.01 / 0.08 / 0.22 | 2 / 11 / 82 | 0.14 / 0.31 / 0.54 | 6.3 % |
| K1:pool_reversal | 2 / 9 / 48 | 0.00 / 0.05 / 0.16 | 1 / 4 / 26 | 0.09 / 0.20 / 0.38 | 2.5 % |
| K2:zone_return | 2 / 12 / 79 | 0.00 / 0.06 / 0.25 | 1 / 6 / 58 | 0.07 / 0.24 / 0.59 | 3.4 % |
| K2:pool_reversal | 4 / 34 / 155 | 0.01 / 0.13 / 0.49 | 1 / 21 / 131 | 0.12 / 0.48 / 1.11 | 22.2 % |

These are the shapes a Setup × target node would have carried. A target
that pays does so with a median adverse excursion of 0.02–0.13 ATR₆₀; a
failure has already run 0.11–0.48 against the claim. The K2:pool_reversal
clock — after a 1m break in the sweep's direction — is the only one where
a claim lives for hours; every other clock resolves within minutes at the
median, because the Eye's failure boundaries sit a fraction of one hour's
diffusion from the price.

**The Eye's own draw.** On zone_return instances the Eye's
`nearest_visible_draw_distance_points` has a median of 14.75 points where
the ≥ 1R target chosen here is 37.9 points away: the Eye's draw is nearer
than one R at the median, which is the same fact the correction found.

## Deviations from the spec, all disclosed

1. **The target rule and the outcome changed before the first fit** (spec
   §9, commit `ae6f4d6`). The smoke block showed the nearest-level target
   inside a bar's range with a 2 % censored class; the target became the
   first level at least 1R away and the verdict outcome became the binary
   `hit_target`. The choice was made on the label shares of two sessions
   with no model fitted, and it is the reason the guard did not refuse
   every cell.
2. **K1:zone_return is not judged**: `reacceptance_held` on a zone_return
   path occurred in 104 of the sessions and `build_folds` needs 110 for one
   primary fold. It is reported in the cell table only.
3. **The blocks emit 114 sessions, not 110.** The last Globex week runs
   through 2022-06-10; `build_folds` takes the first 110 for the primary
   fold and the rolling folds run over the full set, exactly as the
   information-gain gate's run did.
4. The purge is by clock, not by row, inside `select_logistic_c` and the
   LightGBM early-stopping tail (`gate_models.py`), because instances are
   minutes to hours apart. The every-minute gates keep their row gap.

## What the result says, and does not say

It says: on this tape, at the three milestones the Eye publishes for its
own Setups — the location, the held reacceptance, the aligned micro-break —
knowing which Setup it is, what kind of zone or pool it came from, which
steps it has passed with what strength, how old it is and how the 5m / 15m
/ 1h structure is aligned with it does not change a first-passage forecast
to the Setup's own levels that already sees the two distances, the time
left in the session and the recent volatility. The geometry model itself
is the driftless ratio, and on the sweep bar not even that.

It does not say: that Group-5 paths have no value as *locations* for
sizing and stops — the MAE and time distributions above are conditional on
the Setup and are usable as they stand; that a different target
definition (the Eye's draw, a fixed multiple of R, a higher-timeframe
level only) would fail — those are new gates with their own
pre-registration; or that a longer history or a different year would not
change the picture. This gate is closed.

## Commits

`a41793c` spec · `70262b3` plan · `eff1907` gate_models · `545e881` path
log · `d0aadb5` recorder in the Eye pass · `395185d` labels · `4f9a751`
features · `f43e0a7` block builder flag · `866c095` gate script ·
`ae6f4d6` correction (1R target, binary outcome) · `526243c` run 2 of the
information-gain receipt · `64f94b5` README tables.
