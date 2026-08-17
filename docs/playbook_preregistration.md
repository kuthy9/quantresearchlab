# Playbook registration policy

Registry schema: **1**. DFP protocol schema: **5**, LSR protocol schema:
**8**, and FAVR protocol schema: **1**.

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

Path outcomes belong to the calibration/shadow contract rather than the
runtime playbook registry. Every candidate revision uses only geometry visible
at that clock, same-bar collisions resolve invalidation first, and no future
price extreme may retrospectively rewrite a revision. DFP schema 5 may
re-evaluate its primary target on a later pre-Risk revision; Risk approval then
freezes entry, invalidation, draw, target and deadline. The mechanism-specific
primary outcomes are:

- DFP: the frozen primary deliverable target is reached before the frozen
  entry-zone invalidation and deadline; the 4H external draw remains thesis
  and terminal context rather than this empirical delivery outcome;
- LSR: after a child freezes as the execution owner with a complete plan, its
  frozen opposing primary target is reached before the original frozen sweep
  extreme and deadline;
- FAVR: the opposite frozen range-boundary liquidity is reached before the
  original frozen sweep extreme and deadline. FAVR remains parked.

Numeric calibration mappings and action margins are not playbook identities.
Changing a number must not create a new playbook name.

Calibration and replay artifacts bind the complete registry fingerprint and
per-playbook schema versions. DFP schema-1/2/3/4 and LSR schema-1/2/3/4/5/6/7 outputs
therefore remain historical evidence only and cannot represent or calibrate
the current Context-Thesis/Entry-Episode lifecycle contract.

## Open-thesis candidate boundary

`OpenMarketThesis` is a playbook-neutral description, not an order permission.
All open theses, including unmatched and incomplete ones, remain analysis
candidates for Focus and unexplained-episode diagnostics. Each compatible root
is evaluated independently against DFP or LSR; the resulting root-specific
projection is an action candidate only when its exact graph identity, causal
hard gates and frozen plan are complete. Evidence, entry zones, triggers, draws
and terminal state cannot be borrowed across roots.

FAVR remains parked and does not receive action-candidate authority. A root
that matches neither DFP nor LSR remains descriptive only. An entered
candidate's frozen root projection may be retained solely for HOLD/PROTECT/EXIT
management until terminal or boundary exit; it cannot re-authorize ENTER.

The lifecycle has two explicit levels. A descriptive `ContextThesisState`
freezes the market epoch, authority structure, direction, context draw and
structural invalidation; it can remain active with zero current entry children.
Each `EntryEpisodeState` freezes one local displacement/sweep path, zone,
trigger, plan and short deadline beneath that Context. A child terminal does
not close its parent or siblings. A causal Context terminal closes every child.
If a root temporarily leaves the bounded Scene Graph without such a terminal,
root absence alone does not clear its causal gates. When the current observation
uniquely resolves the identical frozen setup, location and active path, that
same root-specific episode continues normal typed evaluation and may retain
action authority after all gates pass; it cannot borrow any identity from a
different root. If the frozen path is missing or ambiguous, the episode is
retained as dormant lifecycle state: it can resolve calibration or an existing
position, but is absent from Focus and the action-candidate map until the exact
path becomes independently resolvable again. `child_episode_ids` is the bounded
current-child projection; historical counts belong in diagnostics rather than
live state. The frozen `entry_path_id` is explicit on both the typed belief and
the child Episode before a trigger or plan exists; it is never inferred from
the setup/root identity.
DFP's runtime Context has no entry/session deadline: its frozen structure,
context draw, market epoch and data/contract boundary own its terminal. Its
thesis calibration target uses a separate registered session-close observation
horizon; that label clock is not written back into runtime lifecycle state.
The Context freezes exactly three terminal-authority roles when it is created:
the confirmed H4 structure, its protected raw swing and the context draw.
Current H4 high/low legs and `swing_projection` nodes remain supporting evidence
and may create a new evidence revision, but their rollover cannot invalidate a
child Episode, close the parent Context or force a position exit. Local zone,
path, trigger and stop facts retain authority over their own Entry Episode only.

## Mechanism boundaries

DFP requires directional higher-timeframe structure and draw, a qualified
displacement-created zone, its first return and an exact rejection, held
reacceptance or aligned 1m micro BOS on that path. H1 continuation acceptance
is supporting evidence, not a mandatory duplicate gate. Direction alone or a
50% pullback proxy is insufficient. Its visible same-direction H4-timeframe
draw establishes the thesis and remains the context/terminal draw; "external"
here means higher-timeframe relative to the M5 setup and does not add a new
`structural_rank == external` gate. That draw is not automatically the trade
target. The primary deliverable target is a visible, unconsumed item from the
registered structural-liquidity kinds, strictly between planned entry and that
terminal draw, before every relevant hard barrier, with both planned target R
and current remaining path R at least 1. Before Risk approval this target may
be re-evaluated from current causal geometry. Risk approval freezes it, and a
position cannot replace it with a later or farther draw.

LSR starts from one visible liquidity pool. A sweep alone is insufficient; the
failed outside acceptance and one exact reverse displacement establish a
longer-lived reversal Context. That Context freezes the manipulation root,
pool-path provenance, displacement identity and causal clocks, direction,
sweep extreme and Context deadline. Every graph-connected displacement-linked
FVG or OB formed after reacceptance is a separate Entry Episode with its own
zone, first pullback, trigger, plan, deadline and terminal state. A closed or
bounded-away Group5 path does not erase the frozen Context or prevent a later
eligible zone from becoming a child. A local zone/path failure closes only its
child. Accepted outside remains a hard-gate failure before the Context is
established, but cannot retrospectively erase a frozen reacceptance and reverse
displacement. After establishment, only the exact sweep-extreme breach, exact
source invalidation, Context deadline or boundary closes all children.
A terminal, closed, deadline-expired, or formerly known but no-longer-live
Context cannot create a later zone child; retained children may only resolve
their already-frozen lifecycle. If two still-live Contexts claim the same
physical location or entry path, neither receives an Entry Episode or action
identity. Only a causally new manipulation root can rearm after that ambiguity.

Each zone must supply its own first pullback and its own later rejection, held
reacceptance or aligned micro BOS. Triggers, paths and terminal facts cannot be
borrowed between siblings. The uniquely earliest completed-clock child whose
hard gates and plan feasibility are valid freezes the execution owner before
action publication. That clock freezes the child's entry, original sweep stop,
primary target, deadline, liquidity route, trigger and first-executable clock;
only current remaining-path, target-visibility and hard-obstruction diagnostics
may weaken it. Later or stronger zones, triggers or targets cannot replace it,
and a same-clock tie fails closed until a new manipulation Context. Completed
or invalidated terminal resolution changes only phase, reason and lifecycle
clocks: the original owner plan, full liquidity route and first trigger remain
materialized as non-actionable frozen history. Before that
owner freeze, a child's route target is provisional: consuming it cannot mark a
zone still waiting for its own first pullback or trigger as completed. This does
not change Context, position-management or Risk target semantics. An accepted
M5 OPPOSED MSS remains supporting evidence, not a mandatory gate or an
independent entry mechanism. These identity changes do not lower a hard gate,
alter Risk geometry or grant broader 1m root authority.

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
