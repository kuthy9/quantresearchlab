# Shared components

`shares/` holds what more than one subsystem depends on: the immutable contracts
in `core/model.py`, data access (`io.py`), the session clock, the scale
registry, the data-split authority, the `ContinuousSMCEngine` orchestration that
wires Eye → Brain → Decision/Risk, and the outcome-blind study projections
(`scene_graph.py`, `market_representation.py`, `case_retrieval.py`,
`causal_cases.py`, `market_cases.py`, `visualization.py`).

`shares/__init__.py` is the aggregate public surface. It resolves every export
lazily, because `shares.core.model` is imported by all four subsystems and an
eager facade would turn the existing module-level cycles into import errors.

## Documents here

- [architecture.md](architecture.md) — runtime authority boundaries.
- [current_implementation_status.md](current_implementation_status.md) — the
  implementation-versus-plan authority for the whole repository.
- [self_review_checklist.md](self_review_checklist.md) — the pre-completion
  self-review pass.

## Protocols — `shares/configs/`

`market_case_input_profiles_v2.json`, the `authority: current` registry of the
twelve input profiles. It points its historical source back at
`configs/data_splits.json`, which stays sealed at the repository root.

## Scripts — `shares/scripts/`

The cross-cutting runners and materializers: `run_continuous_replay.py` (the
full-stack causal replay), `train_market_representation.py`,
`evaluate_market_episode_retrieval.py`, `query_causal_cases.py`,
`prepare_causal_front.py`, `materialize_ohlcv_preholdout.py` and
`audit_market_clock.py`. A script that drives more than one subsystem belongs
here rather than in the subsystem it happens to start from. Throwaway probes
belong here too.
