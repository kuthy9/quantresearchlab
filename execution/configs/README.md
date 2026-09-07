# Execution protocols

This directory is intentionally empty of protocol files. Execution carries no
protocol of its own:

- Its cost, spread and fillability gates live in the `risk` block of
  `configs/model.json` (`maximum_spread_ticks`, `maximum_cost_R`,
  `minimum_fillability`, `minimum_target_R`, `same_bar_resolution`).
- Its MBO data windows live in `configs/data_splits.json`.

Both files are hashed into `atomic_definition_identity`, so they stay at the
repository root. See [eyes/configs/README.md](../../eyes/configs/README.md) for
why that seal fixes their location.

Directory kept so every subsystem has the same four-way shape.
