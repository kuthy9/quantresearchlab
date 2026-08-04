# Mandatory pre-test self-review — v2 current revision

Completed: 2026-07-25  
Scope: `smc_trader/`, active `scripts/`, v2 configs, and tests  
Authorization: syntax, unit, bounded integration, clock audit, development-MBO
materialization, and development/rolling replay only

## Trading logic

- [x] Observations are descriptive and action-free.
- [x] Every update starts from one newly completed 1m bar.
- [x] 4H/1H/5m frames expose completed candles only.
- [x] Swing pivots require right-side confirmation.
- [x] Event clocks preserve when structure, sweeps, rejection, impulse,
  reacceptance, compression, and trigger state first became observable.
- [x] Stateful event persistence is distinct from instantaneous-event age.
- [x] A pullback is continuous evidence, not automatic invalidation.
- [x] Late displacement extension with shallow pullback is contradictory evidence.
- [x] Exactly three playbooks and one common phase vocabulary are registered.
- [x] Required sequence steps are ordered and latched within one setup identity.
- [x] A complete active setup cannot be replaced by a later impulse before it
  reaches a terminal state.
- [x] A complete setup plan is frozen; later extrema cannot rewrite its
  entry, invalidation, target, or deadline.
- [x] Targets originate from visible, confirmed, unswept liquidity.
- [x] Consuming a frozen target before entry invalidates the setup.
- [x] Stops originate from the setup-side causal swing, sweep, or failure event.
- [x] Raw belief is updated once, then calibration is applied once.
- [x] Actions are compared in common net-R utility and ambiguous advantages
  return `abstain`.
- [x] Same-bar stop/target ambiguity resolves to the stop.

## Data, execution, and risk

- [x] Strict previous-session contract selection is hash-bound before MBO joins.
- [x] Duplicate, reversed, and unknown market gaps fail closed.
- [x] Contract changes reset reader, observer memory, and playbook state.
- [x] The registered historical calendar is independent of observed data.
- [x] macOS dataless files are rejected before content reads.
- [x] The August-December MBO holdout remains sealed and requires a separate
  explicit reveal flag.
- [x] MBO is streamed in bounded batches/records and never converted wholesale
  to a DataFrame.
- [x] Databento Add/Cancel/Modify/Clear semantics update the L3 book; Trade,
  Fill, and None are no-ops as required by the vendor normalization.
- [x] Publisher/instrument books are isolated and receive-time order is enforced.
- [x] BBO snapshots never postdate their decision clock.
- [x] No constant slippage is invented from BBO. Insufficient displayed depth
  fails closed because price impact is unknowable from one level.
- [x] Constant CLI execution assumptions are tagged research-only and hard-veto
  entry.
- [x] Spread, cost, freshness, fillability, deadline, account risk, structural
  stop provenance, and liquidity-target provenance are non-bypassable risk gates.
- [x] A protection candidate must remain visible and can only tighten the stop.
- [x] Sequential execution uses prior-decision execution inputs at the next bar;
  current bar-end MBO cannot leak into its own open.

## Calibration, validation, and visualization

- [x] Development, calibration, rolling-validation, and sealed-holdout windows
  are disjoint and preregistered.
- [x] Path tests freeze a complete causal sequence before consuming later bars.
- [x] Path artifacts carry protocol, config, model-code, calibration, decision,
  setup, target, invalidation, and timing identities.
- [x] Calibration uses raw identity beliefs from the calibration window only.
- [x] Fixed quantile bins and monotone PAVA are used without threshold search.
- [x] Insufficient or collapsed calibration samples retain identity calibration.
- [x] Decision and future-reveal images are physically separate and
  decision-hash bound.
- [x] Plan lines cannot expand the price-panel y-axis.
- [x] Four-timeframe primitives, sequence steps, event order/persistence,
  liquidity source IDs, utilities, vetoes, BBO, and AI primitive proposals are
  visible in the decision audit.
- [x] A bounded number of frozen path tests can automatically create paired
  decision/reveal audit artifacts during replay.
- [x] AI action/outcome labels are rejected; AI proposals remain unvalidated
  until separately registered path evidence exists.

The detailed findings and remaining claim boundary are recorded in
`docs/self_review_v2.md`. No current-revision automated test had been run when
this checklist was completed.
