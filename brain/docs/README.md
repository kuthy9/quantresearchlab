# Trading Brain

The Brain interprets the Eye's published facts. It owns no market state and no
history: it reads `MarketSnapshot` and the canonical physical interaction facts,
and it produces belief, admission and action.

**The typed Brain was retired on 2026-09-07 and no belief producer has replaced
it yet.** `playbooks.py`, `playbook_registry.py`, `dol_ranking.py`,
`dol_probability.py`, `signal_policy.py` and `shadow_outcome.py` are gone, and
with them the only producer of `MarketBelief`. `decision.py` and `risk.py`
survive with their contracts intact but have no input; `shares/core/engine.py`
still imports the retired modules and cannot be imported at all. Rebinding that
orchestration to a new belief producer is the next piece of work — see
[Current gaps](#current-gaps).

## Core modules — `brain/core/`

| module | owns |
| --- | --- |
| `brain_entry_sequence.py` | Brain-side interpretation of canonical physical interaction facts |
| `market_belief.py` | the mutually exclusive, exhaustive path hypothesis manager — the real-time path-probability component |
| `calibration.py` | monotone reliability maps for preregistered playbook beliefs |
| `brain_calibration.py` | event-level causal targets; it records and resolves, it does not fit |
| `calibration_replay.py` | bounded, resumable sequential replay primitives |
| `validation.py` | the data-split, holdout, OOF and MBO-manifest protocol that guards every replay boundary |
| `decision.py` → `risk.py` | the sole runtime action authority |

`market_belief.py` is self-contained: nothing else in the repository imports its
`PathKind` vocabulary any more, and `contract/brain/belief.py` takes only
`PathBeliefUpdateRecord` and `PathCompetitionSetState` from it. Its six paths —
continuation, deeper retracement, reversal, balance, failed breakout and
residual unknown — remain a frozen enum; making that set protocol-driven is a
separate task.

## Research — `brain/research/`

Reserved for the Brain's own study projections: recorders and replay harnesses
that observe Brain output and resolve it against later completed bars. Nothing
placed here may become a runtime authority. The directory is currently empty
apart from its `__init__.py`; the calibration recorder and replay harness stay
in `brain/core/` until they are deliberately moved.

## Protocols — `brain/configs/`

`path_hypotheses.json` only. It is not hashed into the atomic semantic identity,
which is why it can live here; `configs/model.json` records its path and its
`path_protocol_fingerprint`, which is `sha256` over the whole file, so a content
edit must recompute that fingerprint in the same change.

The retired `playbooks.json`, `dol_probability.json` and `signal_policy.json`
were removed with their modules, and `path_hypotheses.json` lost its dead
`dol_diagnostic_ranking` block. That edit moved the fingerprint from
`d897635c…b0482` to `61417d9f…3152e0`, which `configs/model.json` now records.

## Tests — `brain/tests/`

Six files. `test_market_belief_update.py` and `test_phase7_hypothesis_manager.py`
cover the path competition set and its reducer. `test_data_splits.py` covers
`validation.py` and the sealed data boundaries it protects; it needs the
gitignored `data/` payload, so a fresh worktree must link or materialize it.
`test_risk.py` and `test_brain_calibration_recorder.py` cover the surviving
Risk and calibration-recorder surfaces. `test_calibration_replay.py` cannot be
collected while `shares/core/engine.py` is broken.

## Authority documents

The Brain's current implementation-versus-plan authority is
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md);
the facts it consumes are defined in
[eyes/docs/smc_semantic_specification_v1.3.md](../../eyes/docs/smc_semantic_specification_v1.3.md).
No Brain surface carries economic or live-trading authority: `configs/model.json`
holds `release_readiness.live_execution_allowed = false`.

## Current gaps

1. `shares/core/engine.py` imports `brain.core.playbooks`,
   `brain.core.playbook_registry`, `brain.core.dol_probability` and
   `brain.core.signal_policy`. Until those import blocks and the code paths
   behind them are removed or rebound, 8 test modules (135 tests) cannot be
   collected, and `shares.ContinuousSMCEngine` is unavailable.
2. `decision.py` reads `MarketBelief.thesis_candidates` and
   `position_management_candidates`. A new belief producer has to fill those, or
   Decision needs a new input contract.
3. `validation.py` sits in `brain/core/` but is imported by
   `eyes/scripts/run_eye_authority_scan.py`,
   `eyes/scripts/scan_mature_ranges.py`,
   `shares/scripts/audit_market_clock.py` and
   `execution/scripts/materialize_mbo_execution.py`. That makes Eye and
   execution scripts depend on the Brain package; moving it back to
   `shares/core/` would remove the inversion.

## Scripts — `brain/scripts/`

Empty apart from its `__init__.py`. `fit_typed_brain_calibration.py` and
`evaluate_typed_brain_calibration.py` were removed with the typed playbooks they
fitted. Throwaway probes belong here.
