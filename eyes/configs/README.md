# Eye protocols

This directory is intentionally empty of protocol files.

Every protocol the Eye reads is hashed into `atomic_definition_identity`, so all
of them stay at the repository root in `configs/`:

- `configs/model.json` — the runtime entry config, which declares
  `semantic_selection` and mounts the six primitives protocols.
- `configs/data_splits.json` — the data authority.
- `configs/primitives_structure_liquidity.json`
- `configs/primitives_displacement.json`
- `configs/primitives_zones.json`
- `configs/primitives_range.json`
- `configs/primitives_entry.json`
- `configs/primitives_interaction.json`

`SemanticDefinitionIdentity` hashes the reference *strings* — `data_split_registry`
and the `primitive_protocol_sha256` map keyed by `configs/primitives_*.json` —
alongside the file contents, and `semantics/parameters_v1_3.yaml` (itself hashed)
carries those strings. Moving any of these files would therefore change
`f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c` and break the
fail-closed check in `eyes/core/semantics.py`. Leave them where they are; refreeze
a new identity rather than relocating one.

Directory kept so every subsystem has the same four-way shape.
