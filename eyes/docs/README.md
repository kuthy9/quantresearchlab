# Trading Eye

The Eye normalizes bars, detects the registered semantic primitives, emits
canonical events, stores the complete atomic history and reduces it to one
current market view. It imports no downstream module — no `eyes/core/` module
imports `brain`, `execution`, or the orchestration half of `shares`, and
`eyes/tests/test_eye_module_boundary.py` enforces that.

## Documents here

- [smc_semantic_specification_v1.3.md](smc_semantic_specification_v1.3.md) —
  the registered atomic semantic definitions (`smc_semantics_v1.3`).
- [canonical_semantic_foundation_v2.1.md](canonical_semantic_foundation_v2.1.md) —
  the additive Foundation projection (`smc_semantic_foundation_v2.1`), which
  declares v1.3 as its parent. This pairing is not a unified full-stack v2.
- [evidence/](evidence/) — bounded-study receipts for the v1.3 reading, the
  balance-range candidate hypotheses and the structural range-width strata.

## Where the Eye's own inputs live

The Eye's protocol files are the eight the atomic semantic identity hashes, so
they stay at the repository root in `configs/`: `model.json`,
`data_splits.json`, and `primitives_structure_liquidity.json`,
`primitives_displacement.json`, `primitives_zones.json`, `primitives_range.json`,
`primitives_entry.json`, `primitives_interaction.json`. Moving them would change
`atomic_definition_identity`, because that identity hashes the reference strings
as well as the file contents.

`semantics/` stays at the repository root for the same reason, by a different
route: `registry_v1_3.yaml` names its parameters file by the root-relative string
`semantics/parameters_v1_3.yaml`, and the registry's own bytes are
`registry_sha256`. Relocating the directory rewrites that line and moves the
identity to `fb7d604d…3b75`. See the provenance-mapping note in
[AGENTS.md](../../AGENTS.md).

## The cost of one bar

Driving the Eye degraded within a single run: over 2022-02-01 the same bar cost
10.0 ms at bar 200 and 71.4 ms at bar 1,800. The cause was not memory or GC —
it was the liquidity candidate collection, which is effectively append-only by
design. A sweep disarms a level and keeps it so a later re-approach is
recognisably the same level, so over 2022-02 the Eye created 15,000 levels and
retired 84.

`reduce_timeframe_state` folded the snapshot-derived candidate projections —
IRL/ERL membership, `normalized_location_in_range`, and the Swing `rank` — back
into the stored candidate at the tail of *every* reduced event. Each fold
walked the whole collection and re-sorted it, so per-bar cost was linear in
accumulated history and a run's total cost quadratic in its length. Measured
over 1,400 bars: `_liquidity_state` ran 27.1 times per bar for 18.6% of
runtime, and `dataclasses.replace` was 34.8%, three quarters of it rebuilding
candidates.

The registry binds those fields as "snapshot-derived only … no canonical
membership event is claimed", so they now run where a hierarchy is
materialized. `_settled_candidate_state` is the single choke point, used by
`MarketSnapshotPublisher` and by `replay_atomic_market_snapshot` alike, so a
checkpoint-restored view and a log-rebuilt one cannot disagree about a field
the reducer no longer folds in.

Measured on the same window, per-bar cost is `15.18 + 0.1529·N` ms before and
`15.17 + 0.1074·N` ms after, where `N` is the candidate population: the fixed
term is untouched and the per-candidate term falls 30%. A 5,160-bar replay of
2022-02-01→05 went from 455.7 s to 348.4 s and emitted the identical event
store — same 67,026 events, same fingerprint `f18ce950…ee78f`, zero differing
keys across all fifteen census sections.

**This lowers the constant; it does not change the asymptotics.** The
collection still grows without bound, so per-bar cost is still linear in
history. Bounding the population — the only asymptotic fix — would stop an
evicted level from ever re-arming and so would change the event stream.

`reduce_timeframe_state` is exported, and its contract moved with this change:
a `TimeframeState` it returns no longer carries those projections. A direct
caller that reads `range_role`, `normalized_location_in_range` or candidate
`rank` must settle the state first.

## Scripts — `eyes/scripts/`

Bounded, outcome-blind Eye studies and scans. `run_eye_authority_scan.py` is the
registered 2023 authority scan (its `RUNTIME_CODE_FILES` list is hashed into the
scan's provenance, so it must be updated whenever an Eye module moves);
`audit_eye_authority_cases.py` replays sampled cases with images;
`scan_eye_event_statistics.py` and `scan_mature_ranges.py` are census scans; the
six `study_*.py` files are the balance-range and structure-reading studies whose
receipts live in [evidence/](evidence/). Throwaway probes belong here too.
