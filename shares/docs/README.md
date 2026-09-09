# Shared components

`shares/` holds what more than one subsystem depends on: data access
(`io.py`), the session clock, the scale
registry, the `ContinuousSMCEngine` orchestration that wires
Eye → Brain → Decision/Risk, and the outcome-blind study projections
(`scene_graph.py`, `market_cases.py`, `visualization.py`).

The case-library chain (`case_retrieval.py`, `causal_cases.py`,
`market_representation.py`) was retired on 2026-09-07 with the typed Brain whose
output it recorded, and `validation.py` moved to `brain/core/`.
`core/engine.py` still imports the retired Brain modules and cannot currently be
imported; `shares/__init__.py` therefore no longer exports
`ContinuousSMCEngine`, `ExecutionFSM` or `RiskApprovedTradeIntent`.

The immutable type contracts moved out of `core/model.py` into the
[contract](../../contract/README.md) package on 2026-09-08, one package per
subsystem boundary. `shares/__init__.py` is still the aggregate public surface
and resolves every export lazily so the facade never pulls the engine in.

## Documents here

- [architecture.md](architecture.md) — runtime authority boundaries.
- [current_implementation_status.md](current_implementation_status.md) — the
  implementation-versus-plan authority for the whole repository.
- [self_review_checklist.md](self_review_checklist.md) — the pre-completion
  self-review pass.

## Protocols — `shares/configs/`

`market_case_input_profiles_v2.json`, the `authority: current` registry of the
twelve input profiles. It points its historical source back at
`configs/data_splits.json`, which stays sealed at the repository root. It is
retained as a frozen preregistration artifact: its readers went with the retired
replay runner, so nothing loads it today.

## Scripts — `shares/scripts/`

The cross-cutting materializers: `prepare_causal_front.py`,
`materialize_ohlcv_preholdout.py` and `audit_market_clock.py`. A script that
drives more than one subsystem belongs here rather than in the subsystem it
happens to start from. Throwaway probes belong here too.

`run_continuous_replay.py` — the full-stack causal replay — was removed on
2026-09-07 along with `train_market_representation.py`,
`evaluate_market_episode_retrieval.py` and `query_causal_cases.py`. All four
were built on the retired Brain; the replay runner in particular was 7,519 lines
of episode, open-thesis and shadow-outcome funnels over `PlaybookBrain` output.
A new full-stack runner has to be written against the new belief producer rather
than patched out of the old one.
