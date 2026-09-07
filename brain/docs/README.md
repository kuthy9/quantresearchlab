# Trading Brain

The Brain interprets the Eye's published facts. It owns no market state and no
history: it reads `MarketSnapshot` and the canonical physical interaction facts,
and it produces belief, admission and action.

## Core modules — `brain/core/`

| module | owns |
| --- | --- |
| `brain_entry_sequence.py` | Brain-side interpretation of canonical physical interaction facts |
| `path_belief.py` | the mutually exclusive, exhaustive path hypothesis manager |
| `dol_ranking.py` | deterministic shadow-only DOL candidate ranking |
| `dol_probability.py` | path-marginalized, shadow-only DOL probability boundary |
| `signal_policy.py` | fail-closed, shadow-only signal policy at the Phase 7 boundary |
| `playbooks.py` / `playbook_registry.py` | the preregistered playbook set and its versioned protocol loader |
| `calibration.py` / `brain_calibration.py` / `calibration_replay.py` | monotone reliability maps, event-level causal targets, and bounded resumable replay |
| `decision.py` → `risk.py` | the sole runtime action authority |
| `shadow_outcome.py` | outcome-blind shadow candidates for one frozen replay |

## Protocols — `brain/configs/`

`playbooks.json`, `path_hypotheses.json`, `dol_probability.json` and
`signal_policy.json`. None of these is hashed into the atomic semantic identity,
which is why they could move here; `configs/model.json` records each one's path
and its `protocol_fingerprint`, so a content edit must recompute that
fingerprint in the same change.

`dol_probability.json` carries the provenance strings
`brain.core.dol_ranking.rank_dol_candidates` and
`brain.core.dol_ranking.DOLObstructionViewFact`, which
`brain/core/dol_probability.py` compares fail-closed against its own constants.
The two must always move together.

## Tests — `brain/tests/`

21 files. `test_v4_typed_vertical.py` is the shared typed-vertical fixture that
`brain/tests/test_brain_conflict_routing.py`,
`brain/tests/test_brain_connection_memo.py`,
`brain/tests/test_calibration_replay.py` and
`execution/tests/test_sequential_replay.py` import from.

## Authority documents

The Brain's current implementation-versus-plan authority is
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md);
the facts it consumes are defined in
[eyes/docs/smc_semantic_specification_v1.3.md](../../eyes/docs/smc_semantic_specification_v1.3.md).
No Brain surface carries economic or live-trading authority: `configs/model.json`
holds `release_readiness.live_execution_allowed = false`.

## Scripts — `brain/scripts/`

`fit_typed_brain_calibration.py` fits the typed Brain maps from resolved causal
recorder rows; `evaluate_typed_brain_calibration.py` evaluates one frozen
artifact on registered validation rows and imports the fitter. Throwaway probes
belong here too.
