# MarketEpisode input stream V1

The production V1 artifact is an input-only, playbook-neutral stream. It is a
downstream consumer of the existing continuous replay and never starts or
advances a second replay loop.

The current fail-closed contract is protocol
`market-episode-input-only-1.2.0`, NeutralMarketState schema 2, neutral Engine
checkpoint schema 3, and market-input runner state schema 7. Schema-1 neutral
state, pre-v3 Engine checkpoints, protocol 1.1.0, and runner-state schemas 4/5/6
cannot resume or materialize under this contract. Engine checkpoint schema 3
adds the production canonical-foundation projection; it does not change the
18-field MarketEpisode row schema. Runner schema 6 added one run-level
repository commit identity; schema 7 additionally binds the exact replay data
continuity policy once in the run manifest. Neither is repeated in input rows.
The recorder schema remains 1 and the Arrow row schema remains the same 18
fields.

## Grain and admission

One row is one physical MarketEpisode milestone reached at one observable
clock. DFP, LSR, ContextThesis, EntryEpisode, Brain, Decision, and Risk do not
admit rows and are not serialized. Multiple physical facts reached on the same
clock are represented once in the ordered `transition_kinds_json` list:

1. `episode_created`
2. `zone_registered`
3. `first_pullback`
4. `trigger`
5. `successful_pulse`
6. `terminal`

Claim and binding relation-only transitions still pass transport, custody,
identity, and clock validation and update the recorder's current episode state,
but never create a row or advance a revision index. A relation update sharing a
clock with a physical milestone is absorbed into that single physical row and
is not named as a transition kind. An unchanged episode produces no heartbeat
row. `successful_pulse` is a typed physical lifecycle fact, not a playbook plan
or action signal.

## Row schema

Rows contain only:

- stable `revision_id` and per-episode `revision_index`;
- the runtime `market_epoch_id` and physical `market_episode_id`;
- `asof`, direction, lifecycle, EntryLocation ID, and EntryPath ID;
- the transition stage and ordered transition kinds;
- the same-clock typed Eye transition and Scene delta;
- the same-clock neutral GlobalMarketContext;
- five canonical OHLCV prefix references;
- replay ordinal and synthetic-bar facts.

Source path, source hash, split/config identity, and other run constants belong
in the run manifest, not every row. OHLCV bars are not copied. Synthetic
updates retain the last consumed real-row prefix boundary.

There is no case ID, model version, playbook diagnostic payload, fingerprint
pair, future outcome stream, Shadow join, MFE/MAE, target-first label, library
pair manifest, or dual-parity artifact in this production contract.

## Safety and continuity

The recorder independently verifies NeutralMarketState schema 2, exact
same-clock transition transport, aware clocks, physical episode identity,
immutable physical custody, append-only milestones, monotonic lifecycle, and
terminal/retirement non-revival. A contract or data-reset anomaly must advance
the runtime market epoch; IDs may be reused only after that epoch boundary.
The input-only runner synthesizes at most five registered open minutes for a
same-contract no-trade gap. A larger same-contract gap emits no synthetic run:
the next real bar carries `data_gap_history_reset` and advances the market
epoch. Cross-contract gaps remain fail-closed. This is a runner continuity fix
under the existing protocol 1.2.0 reset semantics, not a row/protocol change.

Every structured input JSON tree rejects non-finite values, future/outcome key
markers, and future-dated evidence clocks. The sole `outcome` key allowed in an
input is the protocol-bound `MicroBOSReference.outcome` inside the same-clock
Eye `group5_micro_bos_transitions_this_update` collection, with one of its four
causal structural enum values; the same key anywhere else remains forbidden.
Prefix cutoffs must be no later than the row `asof`. Revision IDs hash the
canonical row payload, so checkpoint and resume produce the same identity
without storing a second fingerprint.

## Memory and materialization

Memory is bounded to current active episode states, lightweight hashes for
terminal/retired identities, and rows waiting to be drained. Draining releases
the pending rows; the recorder does not retain a year of prior rows or
fingerprints.

The runner's generic Arrow stream machinery owns shard schema, hash, key-bound,
checkpoint, and manifest integrity. V1 deliberately has no separate case
library finalizer.

## One-time migration parity

The one-time five-day common-fact migration check completed with `PASS` on
2026-08-17. It compared 13,800 completed updates from the full and neutral
Engine paths over the 2022-03 contract-roll window, including the natural
contract reset; a separate injected data-gap fixture also advanced the epoch
identically. The comparison covered bar order, Eye identities, Scene revisions,
physical EntryLocation/EntryPath facts, MarketEpisode formation/terminal clocks,
and reset anomalies. It did not compare legacy and neutral case counts or any
outcome. The temporary dual-Engine parity script was deleted after this result
and is not part of the production or daily validation path.
