# v2 causal continuous architecture

## Runtime flow

```text
newly completed 1m OHLCV bar
        ↓
causal 1m/5m/1H/4H reader
        ↓
descriptive primitives + ordered event memory + MBO execution reality
        ↓
Belief(t) = update(Belief(t-1), Observation(t))
        ↓
three playbook-specific ordered sequences and continuous phases
        ↓
enter / wait / hold / protect / exit / abstain utility comparison
        ↓
independent structural-risk, cost, deadline, data, and fillability vetoes
        ↓
later-bar conservative execution → position/trade feedback
```

The model does not first predict a future price path and then select a strategy.
It updates competing causal hypotheses from the latest completed information,
constructs only currently visible plans, and compares the utility of acting or
waiting. Future paths exist only in a separate post-decision validation stream.

## Eyes

Each completed minute refreshes:

| Frame | Descriptive state |
|---|---|
| 4H | directional displacement, path efficiency, structure progression/age, range position, external liquidity |
| 1H | swing progression, acceptance/rejection, dealing range, path obstruction |
| 5m | impulse, first-pullback depth/completeness, reacceptance, compression |
| 1m | path order, acceleration, counter-pressure, trigger level/hold |

Event memory records confirmed swings, sweeps, breaks, rejection, impulse,
reacceptance, compression, and trigger changes with causal clocks and state
persistence. Execution reality contains manifest-bound MBO BBO, displayed
sizes, depth imbalance, spread, cost, freshness, remaining time, and
fillability. It describes the market; it cannot choose an action.

## Brain

Exactly three hypotheses are preregistered:

1. displacement → first pullback → reacceptance;
2. external-liquidity sweep → rejection/reacceptance → reversal trigger;
3. failed auction → acceptance back to value → ordered 1m return path.

The admission and versioning rules for any future hypothesis are frozen in
`docs/playbook_preregistration.md`. No fourth playbook is currently admitted;
location tools such as FVG, order block, breaker, OTE, or premium/discount
remain evidence primitives unless they demonstrate a distinct causal mechanism.

Each direction uses:

`inactive → forming → armed → waiting_pullback → executable → entered →
weakening/delivering → completed/invalidated`

The ordered event sequence prevents state-combination explosion. Beliefs carry
raw and calibrated probability, supporting/contradicting primitives,
uncertainty, phase duration, structural invalidation, visible targets, remaining
path, and frozen protocol/setup identities.

## Decision, risk, and execution

Flat states compare `enter`, `wait`, and `abstain`; open states compare `hold`,
`protect`, `exit`, and `abstain`. Utility includes delivery belief, available
R, cost, uncertainty, deadline, fillability, and phase readiness. A small
best-versus-second-best advantage produces `abstain`.

Risk is independent. It can veto stale/constant data, spread, cost, insufficient
depth, deadline, account risk, invalid structural provenance, or consumed
liquidity. Entry freezes the thesis. Later structure may tighten a separate
protection stop but cannot rewrite the original invalidation.

Approved entry is evaluated no earlier than the next bar. Stop/target ambiguity
is adverse-first, and a same-entry-bar favorable target is not credited.
Position state and one-decision terminal feedback return to the brain.

## Validation and visual audit

Every full sequence creates a frozen path test before later bars are read.
Calibration uses only raw beliefs from its registered historical window.
Development, calibration, rolling validation, OHLCV holdout, MBO development,
and MBO holdout are separate.

Decision charts contain no future candles. They show all four frames, causal
primitives, probabilities, phases, sequence, evidence, plan provenance,
utilities, risk vetoes, event persistence, BBO, and AI primitive proposals.
Future reveal is a separate file and JSON record bound to the decision,
protocol, config, and model-code hashes.

AI review is a deterministic two-pass workflow. A first replay emits sealed
decision hashes. Review files may then be placed in a separate directory as
`<decision_hash>.json`; a second identical replay accepts only registered
diagnostic issue codes through `--ai-review-directory`. The adapter rejects
action, outcome, profit, and future-path fields, maps each issue to an
`unvalidated` causal primitive proposal, and displays that proposal in the
decision and later reveal. The proposal ledger has no model authority until a
separately preregistered path test validates it.
