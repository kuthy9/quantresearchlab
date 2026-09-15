# Eye per-bar growth at bar 20,000 — 2022-02 tape (2026-09-14)

Question: with every tracker's state bounded (`README.md`, "Per-bar cost"),
per-1,000-bar time on the 2022-02 month replay still rose 13.2 s → 96.7 s.
What still grows with the journal, and what has merely reached a plateau?

Method: `cProfile` of `CausalObserver.observe` over bars 20,000–20,500
against bars 0–500 of `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`,
2022-02-01 → 2022-03-01, the registered `configs/model.json`; and a census of
every container on the observer (and its trackers, emitter, memory, audit
store) at bars 500 / 2,500 / 5,000 / 10,000 / 15,000 / 20,000. Scripts were
scratch (`profile_late_month.py`, `state_census.py`); the numbers below are
copied from their logs.

## Before

`observe` cumulative 11.0 s → 47.2 s (4.29×); 38.4 M → 143.8 M function calls
(3.74×). Growth in cumulative seconds, late minus early, by owner:

| owner | early → late | mechanism | container (bar 500 → 20,000) |
|---|---|---|---|
| `EventStore._indexed_eligible_bars` | 0.0 → 5.9 | whole-scale `sorted()` per structural-leg validation | `_eligible_bar_index` 647 → 25,757 |
| `EventMemory.sync_retained_entity_timelines` + `live_entity_keys` | 0.3 → 6.0 | every completed minute walks every retained timeline | `_entity_timelines` 576 → 3,840 keys; `swing:confirmed` 33 → 933, `swing:forming` 33 → 770 |
| emitter bar-root scans (`_swing_window_event_ids`, `_swing_window_geometry`, crossing next-bar, exact clock root) | 0.03 → 1.9 | linear walks over `(clock, event id)` lists | `_real_bar_event_ids_by_timeframe` 642 → 25,752 |
| `EventStore.append_batch` self | 0.0 → 0.9 | `set(...)` copy of the open forward references per batch | `_unresolved_forward_reference_ids` 1,291 → 54,963 |
| reducer (`reduce_timeframe_state`, `_settle_candidate_views`, `_settle_swing_geometry`) | 1.6 → 11.7 | `TimeframeState.__post_init__` validates the whole hierarchy on every `replace` (17 M iterations / 500 bars); ranks dict rebuilt per candidate projection; `view_of` per hierarchy item per bar | `swing_hierarchy` 197 → 2,048 (1m, saturated by bar 5,000), 0 → 1,031 (5m, still filling) |
| Group 5 `on_completed_1m` | 0.3 → 3.3 | `_zone_source` × `can_register` over every FVG/OB state passed in | 5m FVG+OB sources 8.6 → 285 per bar (capacity 256 + 128) |
| `CausalLiquidityTracker.snapshot` | 0.9 → 3.1 | zone tuple rebuilt per snapshot | S/R zones 24 → 88 across scales |

The `_unresolved_forward_reference_ids` entries are `legacy_transport`
typed-state events' `source_ids` — entity identities (`swing:…`, `pool:…`,
raw swing and displacement ids), never event ids — registered as open forward
references that cannot resolve. They cost membership lookups only.

## After (first four owners bounded)

| | before | after |
|---|---|---|
| late window 20,000–20,500 under profile | 47.6 s | 35.5 s |
| late / early ratio | 4.23× | 3.18× |
| `observe` cumulative, late | 47.2 s | 35.1 s |
| function calls, late | 143.8 M | 137.3 M |
| warm-up bars 500 → 20,000 (no profiler) | 819 s | 598 s |

`_indexed_eligible_bars` and `append_batch` no longer appear among the
growing owners; `sync_retained_entity_timelines` +4.1 s → +2.3 s (the swing
keys are gone; what remains is the exposure-bounded set of frame swings, FVGs
and zones, a plateau); `_record_frame_events` +2.9 s → +1.0 s.

What remained, in order: the reducer's hierarchy working set (+5.1 s
`reduce_timeframe_state`, +2.2 s `_settle_candidate_views`, +1.2 s
`_settle_swing_geometry`), Group 5's source scan (+2.4 s), the liquidity
snapshot's zone rebuild (+2.2 s), `_liquidity_state`'s sort over 97 1m
candidates (+1.6 s), and structural-leg projection (+1.3 s; 330 vs 274 folds,
market-dependent). None of these grows with the journal; the first two were
design choices — `swing_hierarchy_hot_retention` (2048 per scale, with the
registry's own evidence that a rank assignment reaches ≤ 4 swings back and the
geometry tree ≤ 447) and Group 3's terminal-until-capacity retention, the
pattern Groups 5 and the liquidity tracker replaced with an exposure-based
retention.

## After the two plateau owners (same day)

The hierarchy is validated once per object and trusted until it changes, the
rank map is carried by the validated tuple, and the geometry settle re-views
only the Swings that moved (`ValidatedSwingHierarchy`); Group 3 terminal
states are exposed once, then compacted (`terminal_state_retention_native_bars`,
registered as `zone_terminal_state_retention`).

| | before | after four | after six |
|---|---|---|---|
| late window 20,000–20,500 under profile | 47.6 s | 35.5 s | 25.3 s |
| late / early ratio | 4.23× | 3.18× | 2.36× |
| `observe` cumulative, late | 47.2 s | 35.1 s | 25.0 s |
| function calls, late | 143.8 M | 137.3 M | 97.2 M |
| warm-up bars 500 → 20,000 (no profiler) | 819 s | 598 s | 415 s |

Remaining late-minus-early growth: `reduce_timeframe_state` +2.5 s,
`_liquidity_state` +1.6 s and `_project_candidate_views` +1.6 s (97 1m
candidates against ~30 early — the candidate-retirement plateau), the
liquidity snapshot +1.5 s (88 S/R zones against 24), Group 5 +1.3 s, and
structural-leg projection +1.2 s. All are bounded by registered retentions
or by the market, none by the journal.
