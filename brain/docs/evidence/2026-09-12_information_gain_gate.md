# Information-gain gate — receipt, 2026-09-12

Spec: [2026-09-11-information-gain-gate-design.md](../specs/2026-09-11-information-gain-gate-design.md).
Plans: [eye-per-bar-cost](../plans/2026-09-12-eye-per-bar-cost.md), [information-gain-gate](../plans/2026-09-12-information-gain-gate.md).

## Verdict: FAIL

**Two runs, one verdict.** The tables from "The run" to "Ablation" are
**run 1**, which placed the barriers in ATR₁ₘ; the spec correction of
2026-09-12 (`4facecd`, `322237a`) found those cells degenerate — "neither"
on 0.0 % of rows, so the label was the sign of the first tick — and voided
them. **Run 2** below repeats the gate with the barriers in ATR₆₀ and
non-degenerate classes, and fails all eighteen cells again. Run 1 is kept
here because it was written before the defect was found; it is evidence
of the harness, not of the market.

None of the eighteen verdict cells — {`fp_1.0_1.0`, `fp_1.0_0.5`,
`fp_0.5_1.0`} × {C1, C5, C15} × {logistic, lightgbm} — meets the
pre-registered conditions of spec §5.9. Under the spec, C (the joint
conditional model), D (hypothesis lifecycle on distribution modes) and E
(event-driven orchestration) do not start. The redesign stops at the
measurement, which is the outcome the brief itself allowed for.

## The run

| item | value |
| --- | --- |
| run id | `aa4be1c91244d0c4` (`outputs/information_gain_gate/aa4be1c91244d0c4/`, ignored directory) |
| source | `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`, the path `configs/data_splits.json` binds |
| atomic identity | `f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c`, unchanged |
| Eye pass | 23 Globex-week blocks, 2022-01-03 → 2022-06-06, each warmed from the seven days before its Sunday 18:00 open (spec §5.2); 6 workers, 230 min wall; 44–66 min per block |
| clocks | 149,366 (6,414–6,654 per week; 5,335 in the 2022-04-11 holiday week); 1,050k transition events |
| clock coverage | C1 113,650 (76.1 %), C5 25,270 (16.9 %), C15 5,293 (3.5 %), C60 1,254 (0.8 %), ALL 149,366 |
| primary fold | train 2022-01-03 → 2022-05-09 (90 sessions), OOS 2022-05-10 → 2022-06-06 (20 sessions); OOS rows C1 19,951 / C5 4,391 / C15 937 / C60 219 / ALL 26,026 |
| rolling folds | five, train 60 → holdout 10, step 10; holdouts from 2022-03-28, 04-11, 04-26, 05-10, 05-24 |
| M₀ | 166 columns: the 24 tape features of `predictability_gate.raw_features` plus the Eye's 142 state components |
| M₁ | M₀ + 884 Δₜ columns (`brain/research/event_sequence.py`): 49 kinds × 5 scales × {since, count, dir} over L = 120 min; one K = 4 track per scale; trigger one-hot |
| models | multinomial logistic (C ∈ {0.01, 0.1, 1.0} by blocked purged CV on the primary training rows, reused on the rolling folds) and LightGBM multiclass (400 trees, early stop on a purged tail) |
| tables | `results.csv` sha256 `d97af292…` (200 rows), `verdict.csv` `888bc8d1…`, `ablation.csv` `df298481…` (441 rows) |

## Primary fold: Δ log-loss, M₁ − M₀ (negative is better)

Session-block bootstrap intervals in brackets (2,000 draws).

| clock | model | fp_1.0_1.0 | fp_1.0_0.5 | fp_0.5_1.0 |
| --- | --- | --- | --- | --- |
| C1 | lightgbm | **−0.0024** [−0.0049, −0.0005] | +0.0000 [−0.0005, +0.0006] | −0.0010 [−0.0022, +0.0002] |
| C5 | lightgbm | −0.0001 [−0.0016, +0.0015] | −0.0008 [−0.0018, +0.0002] | +0.0001 [−0.0008, +0.0011] |
| C15 | lightgbm | −0.0012 [−0.0024, +0.0000] | +0.0006 [−0.0025, +0.0040] | +0.0001 [−0.0013, +0.0013] |
| C1 | logistic | +0.0151 [+0.0110, +0.0193] | +0.0110 [+0.0080, +0.0145] | +0.0140 [+0.0085, +0.0203] |
| C5 | logistic | +0.0219 [+0.0139, +0.0311] | +0.0144 [+0.0091, +0.0198] | +0.0151 [+0.0097, +0.0213] |
| C15 | logistic | +0.0250 [+0.0125, +0.0381] | +0.0255 [+0.0079, +0.0446] | +0.0232 [+0.0114, +0.0338] |

M₀ log-loss on these cells is 0.652–0.699, so the largest improvement
anywhere is 0.3 % of the baseline loss.

**Why every cell fails.**

- *Logistic.* M₁ is worse than M₀ on every clock and every target, with the
  whole interval above zero, and beats M₀ on 0 of 5 rolling folds (1 of 5 on
  C1 × fp_0.5_1.0). Clipping the standardised matrices to ±10 σ (see
  deviations) moved C1 × fp_1.0_1.0 from +0.0157 to +0.0151: this is not a
  numerical event, the linear class overfits the 884 sequence columns.
- *LightGBM.* One cell has its primary interval below zero: C1 ×
  fp_1.0_1.0, Δ −0.0024, one-sided p 0.0055, Holm-rejected at α = 0.10
  across the nine tests. Its rolling folds are +0.0001, +0.0002, +0.0005,
  +0.0010, −0.0003: one of five beats M₀, so it fails consistency (≥ 80 %)
  and drop-best robustness. C15 × fp_1.0_1.0 (Δ −0.0012, interval
  touching zero, 3 of 5 folds) and every other cell fail condition 1.

## Reported, not judged

| clock | model | fp_1.0_1.0 | fp_1.0_0.5 | fp_0.5_1.0 |
| --- | --- | --- | --- | --- |
| C60 (219 rows) | lightgbm | +0.0007 [−0.0006, +0.0022] | −0.0004 [−0.0015, +0.0007] | −0.0015 [−0.0059, +0.0025] |
| C60 | logistic | +0.0990 [+0.0372, +0.1618] | +0.0674 [+0.0092, +0.1336] | +0.0823 [+0.0233, +0.1474] |
| ALL (26,026 rows) | lightgbm | −0.0006 [−0.0019, +0.0006] | +0.0005 [−0.0001, +0.0013] | −0.0018 [−0.0030, −0.0007] |
| ALL | logistic | +0.0158 [+0.0123, +0.0198] | +0.0126 [+0.0097, +0.0160] | +0.0129 [+0.0084, +0.0177] |

The every-minute clock shows the same picture as the event clocks: the
event clocks did not un-dilute a signal, there was none to un-dilute.

Continuous targets, primary fold, Δ MSE: `asymmetry_60` lightgbm −0.0032 /
+0.0001 / −0.0053 on C1 / C5 / C15 (every interval covers zero), logistic
+0.0138 / +0.0030 / −0.0009; `range_60` lightgbm −0.17 / +0.26 / +0.56,
logistic +1.94 / +0.90 / +2.49. Δₜ does not help the one target that is
known to be predictable either.

## Ablation (M_Full − Eᵢ, lightgbm, primary fold, 49 kinds × 3 clocks)

`delta_logloss_vs_full` is the loss without the kind minus the loss with it:
positive means the full model was using the kind. Every value is of the
order of the bootstrap half-widths above (10⁻⁴ … 10⁻³).

| clock | mean | sd | min | max | kinds whose removal *improves* the model |
| --- | --- | --- | --- | --- | --- |
| C1 | +0.00004 | 0.00029 | −0.00032 | +0.00223 | 17 of 49 |
| C5 | −0.00019 | 0.00071 | −0.00208 | +0.00213 | 31 of 49 |
| C15 | +0.00024 | 0.00055 | −0.00121 | +0.00182 | 5 of 49 |

Largest mean effect over the three targets: C1 `delivery_phase_updated`
+0.00083, `delivery_phase_exited` +0.00061; C5 `acceptance_confirmed`
+0.00043, `fvg_created` +0.00027, `sweep_confirmed` +0.00026; C15
`acceptance_confirmed` +0.00088, `liquidity_sweep` +0.00080,
`swing_confirmed` +0.00077. `acceptance_confirmed` is the only kind in the
top three on every clock. None of these is distinguishable from the
resampling noise of the cell it was measured in; the table is a ranking of
noise with one weakly recurring name, not a list of events with predictive
value.

## Deviations from the spec, all disclosed

1. **Part 1 stopped at the harness** (spec §4, outcome paragraph). The
   projection memo could not hit because every liquidity candidate is
   rebuilt on every bar (`eyes/core/market_state.py:2374`, `age_bars + 1`);
   the remaining two changes were worth about twenty minutes of wall time
   and were not done. The Eye ran unchanged; the block protocol bounded its
   cost. `eyes/scripts/replay_hash_stream.py` stays.
2. **The gate was restarted once after its first six C1 cells were seen.**
   The first run's C1 logistic cells were +0.0157 / +0.0113 / +0.0152 and
   the lightgbm cells −0.0024 / +0.0000 / −0.0010. It was stopped to (a)
   clip the standardised matrices to ±10 σ for both models — a guard
   against near-constant Δ columns whose first out-of-sample occurrence
   standardises to an arbitrarily large value — and (b) move the ablation
   from logistic (100–170 s per fit) to lightgbm (6 s). The verdict cells
   of the second run match the first run's on every cell both runs
   computed: the lightgbm cells are identical to four decimals and the
   logistic cells moved by at most 0.0012 (+0.0152 → +0.0140 on
   fp_0.5_1.0), all still above zero. The clip did not change the
   conclusion, and it is recorded here so that it cannot be mistaken for a
   pre-registered choice.
3. **Ablation on lightgbm only**, after the verdict was written, per (2b).
4. **`build_dataset` now requires sixty consecutive traded minutes** ahead
   of a sampled clock (`brain/research/trajectory_dataset.py`); it
   previously accepted any sixty rows, so the last hour before the
   maintenance break and before the weekend was labelled with the next
   session's bars. This is a correction of the research path made before
   the run, covered by `brain/tests/test_event_log.py`.
5. **The dataset's future window is offset by one minute**: it is the sixty
   rows after the row at `asof`, i.e. closes at asof+2 … asof+61, a
   property the research path has always had and that this run kept so
   its numbers are comparable to `brain/docs/README.md`. It shifts the
   horizon for M₀ and M₁ alike and does not bias the comparison.

## What the result says, and does not say

It says: on this tape, this Eye's 49 transition kinds — as recency, count,
direction, per-scale order and trigger, over two hours — do not change a
sixty-minute first-passage forecast that already sees the raw tape and the
Eye's state, whether the forecast is read at every minute or only when a
structural event fires, and whether the reader is linear or boosted. The
one linear result is unambiguous in the other direction: the sequence
columns cost the linear class 1.5–2.5 % of its loss out of sample.

It does not say: that events carry no information under a different
encoding, a different horizon, a different asset, or after the Eye's
retention semantics are redesigned (the never-retiring one-minute
liquidity set of spec §2 is a candidate cause of the flat state features
and was bounded, not fixed, by the weekly blocks). Those are new gates,
each with its own pre-registration; this one is closed.

## Run 2 (ATR₆₀ barriers, 2026-09-12 14:00): FAIL on all eighteen cells

Same run id and blocks; `results.csv`, `verdict.csv` and `ablation.csv`
under `aa4be1c91244d0c4/` are run 2's, run 1's are kept under `run1/`.
Barriers at `multiplier · ATR₁ₘ · √60`; class shares on the primary training
rows (neither / upper / lower): `fp_1.0_1.0` 0.668 / 0.154 / 0.177,
`fp_1.0_0.5` 0.401 / 0.139 / 0.460, `fp_0.5_1.0` 0.400 / 0.440 / 0.160.

Primary fold, Δ log-loss M₁ − M₀, session-block bootstrap intervals,
rolling folds beating M₀ out of five:

| clock | model | fp_1.0_1.0 | fp_1.0_0.5 | fp_0.5_1.0 |
| --- | --- | --- | --- | --- |
| C1 | lightgbm | −0.0059 [−0.0221, +0.0088] 2/5 | −0.0063 [−0.0256, +0.0106] 2/5 | +0.0047 [−0.0109, +0.0200] 3/5 |
| C5 | lightgbm | +0.0036 [−0.0084, +0.0153] 0/5 | +0.0128 [−0.0006, +0.0244] 1/5 | −0.0012 [−0.0168, +0.0143] 0/5 |
| C15 | lightgbm | +0.0158 [+0.0022, +0.0296] 0/5 | +0.0052 [−0.0075, +0.0182] 1/5 | +0.0092 [−0.0048, +0.0234] 1/5 |
| C1 | logistic | +0.1658 [+0.0958, +0.2446] 0/5 | +0.1791 [+0.1032, +0.2644] 0/5 | +0.1554 [+0.0965, +0.2180] 0/5 |
| C5 | logistic | +0.0952 [+0.0501, +0.1427] 0/5 | +0.1334 [+0.0753, +0.1967] 0/5 | +0.0961 [+0.0569, +0.1387] 0/5 |
| C15 | logistic | +0.0777 [+0.0334, +0.1199] 0/5 | +0.1417 [+0.0887, +0.2025] 0/5 | +0.0838 [+0.0485, +0.1186] 0/5 |

No LightGBM interval excludes zero on the favourable side; no cell beats
M₀ on four of five rolling folds; Holm rejects nothing. The linear class is
8–18 % worse with the 884 sequence columns on every cell.

What M₀ itself does: its primary-fold log-loss against the class prior's
entropy is 0.815–0.830 vs 0.864 on `fp_1.0_1.0`, 0.952–0.991 vs 0.998 on
`fp_1.0_0.5`, and 0.998–1.023 vs 1.021 on `fp_0.5_1.0` (LightGBM; the
logistic M₀ is 0.829–0.879, 0.961–0.998, 1.020–1.040). The 4–5 % gain on
the symmetric target is the magnitude skill the predictability gate already
measured — whether ±1 ATR₆₀ is reached at all — and it vanishes on the two
asymmetric targets, where the question is which side first. Nothing here
predicts the side, and the event sequence adds nothing to either.

## Commits

`4988dd9` event log · `1441f81` weekly blocks · `a27480f` Δₜ encoding ·
`c0349d3` first-passage labels · `fe9c2e3` bootstrap, Holm, verdict ·
`8e4517c` gate script · `11c0c72` contiguous future · `e97650e` save_block
fix · `295fe44` z-clip and lightgbm ablation. Part 1: `b027e51`, `a52241f`
(harness). Spec and plans: `f4f39e3`, `1a4bade`, `8772588`, `36d66c2`,
`eeb51b8`.
