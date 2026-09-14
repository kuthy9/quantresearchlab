# Eye Contract Repairs (2026-09-13)

**Status (end of 2026-09-13):** Tasks 0, 4, 1, 5, 2 and 7 done as written. Task 3
done for displacement and zones; the dealing range stays 1H (its Group 4
parameters are written in 1H/1m units, so a per-scale range is a new definition,
not the same one applied more widely). Task 6 bounded the candidate set and
replaced three full rebuilds; the per-bar curve is flatter, not flat, and the
remaining owners are named in [README.md](../README.md). The Foundation v2.1
stack is kept and marked as the next removal. Details per task in the README
sections written beside each change.

Branch: `eyes` (worktree `.claude/worktrees/eyes`). Merge to `main` when every task
below is green. Ordered by dependency, not by size. Every task is TDD: failing test,
minimal change, `eyes/tests` green, one commit. The atomic identity and event
fingerprints are expected to change from Task 4 onward; the hash-stream harness
(`eyes/scripts/replay_hash_stream.py`) is used to *measure* the change, not to
forbid it.

Evidence for each item is in the audit that preceded this plan (2022 gate event
log, 1,026,436 events; one-session replay 2022-02-01, 17,532 events).

## Task 0 — make the Eye suite collectable
10 modules fail collection on orphan imports of the retired Brain. Eye-side fixes
only: retire the checkpoint-store test onto `pickle`, lazy-import the retired
validation loader in the two authority scripts, skip the engine-bound Eye test with
a stated reason. `shares/core/engine.py` itself stays broken — it is orchestration.

## Task 4 — one crossing, one touch
Root cause: two emitters (`_record_frame_events` swing-break path;
`_append_inventory_crossing_event`) with disjoint dedup registries. Fix at the
source: the swing-break path registers its touch/penetration in the same
`(level_id, crossed_at)` registries the inventory path consults, and the inventory
path reuses them. Populate `entity_id` on every crossing event so two levels at one
price stay distinguishable. Acceptance: exact-duplicate share on a one-session
replay 9.9 % → 0 for `level_touched` / `level_penetrated`.

## Task 1 — one formation clock
Atomic events carry formation in `event_time` and leave `formed_at` empty; legacy
state transports fill `formed_at`. Fill `formed_at` (and `entity_id`) on the atomic
emitter's single choke point `_append_semantic_atomic`.

## Task 5 — target outcomes on the target's own timeframe
Inventory crossings for a 5m/15m/1H level are published on `timeframe=1m`.
Publish `LEVEL_REACHED` / `LEVEL_INVALIDATED` derived at the emitter from the
existing touch / accepted-outside facts, on `item.timeframe`, with `entity_id`.

## Task 2 — 1m leaves the main event clock
`ObserverConfig.published_timeframes`; the store keeps everything for provenance,
`semantic_events_this_update` / `recent_events` carry only published timeframes,
1m transitions move to `MarketObservation.microstructure_events`.

## Task 3 — no per-timeframe roles
Displacement / zone / dealing-range trackers become per-timeframe dicts like
structure / liquidity already are; protocol `timeframe` becomes `timeframes`.
Acceptance: the twelve 100 %-NaN Brain features become finite where the scale is
ready.

## Task 6 — candidate retirement
Retire liquidity candidates by age and by distance (parameters in
`semantics/parameters_v1_3.yaml`), emit `LIQUIDITY_RETIRED`, and stop the
inventory / emitter maps from growing with them. Acceptance: per-bar cost flat
across 4,000 bars in `replay_hash_stream --timing-only`.

## Task 7 — remove what the runtime never loads
Foundation v2.1 stack (`semantic_foundation`, `semantic_lifecycle`,
`semantic_zones`, `structural_outcome`), `eye_statistics`, the six `study_*.py`
scripts and their tests; keep `foundation_registry` only if a snapshot field still
needs it.
