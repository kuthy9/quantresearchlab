# Playbook registration policy

Schema: **1**

## Current set

The runtime keeps exactly three typed causal mechanisms:

1. `displacement_first_pullback` (DFP);
2. `liquidity_sweep_reversal` (LSR);
3. `failed_auction_value_return` (FAVR).

Long and short are directions of one mechanism, not separate playbooks.
Session, volatility, FVG, order block, support/resistance, premium/discount and
microstructure are context, location or execution evidence. Their combinations
do not become new strategies.

Each registration in [`../configs/playbooks.json`](../configs/playbooks.json)
must freeze:

- one falsifiable mechanism and causal start event;
- ordered required events and completed-bar clocks;
- lifecycle, terminal and rearm conditions;
- supporting, opposing and missing evidence groups;
- one structural invalidation source;
- decision-time visible draw/targets;
- valid entry location, trigger, deadline and chase/path-space rejection;
- which typed dimensions each observation may update.

Numeric calibration mappings and action margins are not playbook identities.
Changing a number must not create a new playbook name.

## Mechanism boundaries

DFP requires directional higher-timeframe structure, continuation acceptance or
BOS, a qualified displacement-created zone, its first return and an aligned 1m
trigger. Direction alone or a 50% pullback proxy is insufficient.

LSR starts from one visible liquidity pool. A sweep alone is insufficient; the
opposite displacement/MSS, location and trigger must form before executable.
The sweep extreme freezes the invalidation and the opposite visible liquidity
supplies the draw.

FAVR requires a mature range with duration, repeated two-sided tests, internal
crossing, inside acceptance and usable value. It then requires a boundary
sweep, failed outside acceptance, re-entry and displacement back inside. If
natural mature-range authority is absent, FAVR is parked. General wick
rejection cannot stand in for a failed auction.

If FAVR and LSR repeatedly share the same start, invalidation and destination,
the appropriate response is to merge or retire one mechanism, not preserve two
names.

## Change rule

Model development changes one concept family at a time. A repeated issue across
a fixed stratified audit batch may authorize one computable semantic fix. The
fix is checked on a different development window. A single loss, PnL search or
AI action opinion cannot authorize a change.

A fourth playbook is considered only when a recurring mechanism cannot be
represented as a primitive, phase or evidence group inside the existing three;
its causal definition is frozen before calibration, and it shows distinct
behavior on later untouched data. Sparse coverage is not itself a reason to add
one.

The validation order is fixed by
[`../configs/data_splits.json`](../configs/data_splits.json): definition work,
Brain calibration, one rolling OOF evaluation, MBO execution stability and one
sealed holdout reveal. Failure with a correctly connected model may mean the
hypothesis has no usable edge; it does not authorize repeated threshold tuning.
