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

## Scripts — `eyes/scripts/`

Bounded, outcome-blind Eye studies and scans. `run_eye_authority_scan.py` is the
registered 2023 authority scan (its `RUNTIME_CODE_FILES` list is hashed into the
scan's provenance, so it must be updated whenever an Eye module moves);
`audit_eye_authority_cases.py` replays sampled cases with images;
`scan_eye_event_statistics.py` and `scan_mature_ranges.py` are census scans; the
six `study_*.py` files are the balance-range and structure-reading studies whose
receipts live in [evidence/](evidence/). Throwaway probes belong here too.
