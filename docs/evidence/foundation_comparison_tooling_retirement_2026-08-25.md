# Foundation Comparison Tooling Retirement Receipt

Retired on 2026-08-25 from preimage commit
`b3b05508f5d47abbdc43532d775a387ea6e6c099`. This receipt removes only the
inert Phase 4/5 and Phase 6 comparison orchestration layer. It does not alter a
frozen manifest, result, semantic authority, generic replay runner, or research
primitive.

## Retired files

| File | SHA-256 | Git blob | Bytes / lines |
|---|---|---|---:|
| `scripts/compare_foundation_v2_results.py` | `9a30549e6e9a5633ea697780811ff244962b462456a651bf3a327d89d6b74702` | `6b539c020154d93e062953a74d35e61b9f4ed1e2` | 44,753 / 860 |
| `scripts/run_phase45_foundation_comparison.py` | `fa6d8a382097b4c5d79a48e6f23228e84a3e962ee6706bfae3128fd515563cd5` | `037456393b35216b8c1617ee2d9eed6a56392a8b` | 21,187 / 534 |
| `scripts/run_phase6_mbo_foundation_comparison.py` | `dbadfe06181def3b93c1624d2b03a8fde5e8b6cefd8da23559b200090bcea047` | `6727c6ef17b05828d6f7dfb9024090961fdf4755` | 17,565 / 452 |
| `tests/test_phase45_foundation_comparison.py` | `631e8c078bc0ae14023823eab0bc26abdb4116940b5a18a898b274bc095981dd` | `08b994c55587df5c11f59dd10bdbe5b9da303412` | 8,155 / 218 |
| `tests/test_phase6_mbo_foundation_comparison.py` | `617a3a060f9e29c92fecfb1767d94fc487cdb985b8a63dcf25a06d6eb8a70631` | `97725ebd6a6f0adfcacac10088b899938fc3f897` | 12,991 / 380 |

## Reference-closure evidence

Repository-wide exact-name search found each wrapper referenced only by itself,
its two frozen manifests, and its dedicated test. The comparator had no code or
test consumer outside itself; the inventory was its only documentation mention.
No matching `experiments/results/foundation_v2_*comparison*` file exists, so the
comparator had no complete input set. The wrappers were also hash-bound to their
historical source identities and therefore fail closed after the canonical
Zone/Range Auction owner migration.

The four frozen manifests remain byte-for-byte untouched:

| Historical manifest | SHA-256 |
|---|---|
| `foundation_v2_2024_06_phase45_w1_development_comparison_v1.yaml` | `4d03649ceaea9a337fb8a95ed586c80e8b735f763c57cba8c635c73860d5bbc6` |
| `foundation_v2_2024_06_phase45_w2_historical_validation_comparison_v1.yaml` | `945f12fa366c982d1430ddc596556e990371a505142dcbaf10618e1efd7317f4` |
| `foundation_v2_2024_06_phase6_mbo_w1_development_comparison_v1.yaml` | `1b832e72084684735bfe95b827c83282d0285ecdc1a5bdd83643036019b68b98` |
| `foundation_v2_2024_06_phase6_mbo_w2_historical_validation_comparison_v1.yaml` | `4c6595c19bab5ed1d547edc4306e8c0647413242894ce7ffbf497df44be22584` |

They remain historical governance records, not runnable current identities.
Generic owners `run_semantic_signal_research.py`,
`run_mbo_mechanism_research.py`, `signal_research.py`, and
`mbo_mechanism_research.py` remain available for a newly frozen study.

## Recovery

Recover the exact retired bytes without changing the manifests:

```bash
git restore --source=b3b05508f5d47abbdc43532d775a387ea6e6c099 -- \
  scripts/compare_foundation_v2_results.py \
  scripts/run_phase45_foundation_comparison.py \
  scripts/run_phase6_mbo_foundation_comparison.py \
  tests/test_phase45_foundation_comparison.py \
  tests/test_phase6_mbo_foundation_comparison.py
```

Recovery alone does not make the old contracts current. Any renewed comparison
must receive a new manifest identity and produce new no-clobber result paths.
