# Neutral-B2 and Prompt Document Retirement Receipt

Date: 2026-08-25
Preimage commit: `2ef40cbd7342837738d63996e00d077fc8d7557c`

The optional Neutral-B2 audit island had no runtime, current-manifest, result,
or evidence consumer. Its three configurations, three CLIs, and three
self-contained tests were retired as one unit. The non-authoritative prompt PDF
had one documentation link and no executable consumer; repository-owned
architecture and semantic documents remain authoritative.

| Retired path | Preimage SHA-256 |
|---|---|
| `configs/neutral_b2_eye_market_episode_transport_audit.json` | `0f5e441243e1327d45acebcfde2b2e99b9df2b26ca8e37755f589edfa419eda9` |
| `configs/neutral_b2_mechanism_semantic_audit.json` | `a77e04ed2113f3cec9e612e0b8c6ab8b7dcadf9000fc7ecc4c5572d651bf71a6` |
| `configs/neutral_b2_semantic_audit.json` | `512287428d86aa975df9c558e005a686d135bcedb1696e9a0ad224526a734a44` |
| `scripts/audit_neutral_b2_eye_market_episode_transport.py` | `eb1f169e298077af5d62b9e0ecb3ed1ee1e84c43f177f1f5a12dd72c04bdb189` |
| `scripts/audit_neutral_b2_mechanism_semantics.py` | `e496c4c4160007e226c1ff0d81d9997889185fc55080dd728250ad5a40098b92` |
| `scripts/audit_neutral_b2_semantic_signal.py` | `de0a656c0edf609f18f6dd88f3e8bc70ca3c791566ef426d35fc915f587768c1` |
| `tests/test_neutral_b2_eye_market_episode_transport_audit.py` | `46714fcae4d461ca93911e8167b1fe511a9ffd3d95e08ba1849fe1c8d155fa08` |
| `tests/test_neutral_b2_mechanism_semantic_audit.py` | `8f548286d03dac7707c97dd0e767899388fd7213a35590c85f269d5fe263a81b` |
| `tests/test_neutral_b2_semantic_audit.py` | `62c11f9a8f00ed444decca75c607cef61d6dc9530e0cfeead0bd676046148c68` |
| `docs/codex提示词.pdf` | `8963efcf69dd7179987b6b765a7d855a3e3ffd9acbb0f27a7d2bcde4c0248356` |

No file under `data/`, `inputs/`, `outputs/`, `experiments/`, `semantics/`, or
`docs/evidence/` was removed. The frozen inventory rows remain as the preimage
record. Recover any retired byte stream with
`git show 2ef40cbd7342837738d63996e00d077fc8d7557c:<path>` and verify it against
the SHA-256 above.
