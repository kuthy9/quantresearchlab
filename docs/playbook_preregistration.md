# Playbook preregistration policy

## Current frozen set

Version 2 keeps exactly three causal hypotheses:

1. `displacement_first_pullback`
2. `liquidity_sweep_reversal`
3. `failed_auction_value_return`

Long and short are symmetric directions of one hypothesis, not separate
playbooks. Session, volatility, news, premium/discount, FVG, order block,
breaker, and OTE are context, evidence, location, or execution primitives; they
do not become separate strategies merely by appearing in a different
combination.

Each playbook registration must freeze:

- one falsifiable market thesis and its causal start event;
- an ordered sequence of observable steps and their event clocks;
- the shared continuous phase vocabulary;
- supporting and contradicting evidence;
- a structural invalidation source and the clock at which it freezes;
- target selection from decision-time visible liquidity;
- deadline and expiry conditions;
- action eligibility, without hard-coding a buy/sell label;
- a future-blind path-test outcome and same-bar resolution rule;
- data scope, calibration method, protocol version, and content hash.

Numeric belief and utility thresholds are calibration parameters, not playbook
identities. Changing a threshold must not create a nominally new playbook.

## Overlap, competition, and retirement

The registry is not an exclusive classifier. The same completed minute may
support more than one directional hypothesis. Each hypothesis keeps its own
phase, probability, evidence, invalidation, target, and deadline; the decision
layer compares their action utilities. If two hypotheses remain close enough
that neither action has a clear registered advantage, the result is
`abstain`--not a new combination playbook.

Overlap is measured on frozen path episodes. If two registered playbooks
repeatedly start from the same causal event, use the same invalidation and
destination, and produce no stable incremental calibration or utility, the
default remedy is to merge or retire one in the next major version. Adding a
fourth playbook is not a remedy for poor separation among the existing three.

## Admission gate for a new playbook

A candidate is admitted only in a new major registry version after all of the
following are true:

1. **Structural non-redundancy.** Its start event, ordered mechanism, frozen
   invalidation, or causal destination cannot be represented as a phase or
   primitive inside one of the three current hypotheses.
2. **Outcome-free recurrence.** Development data contains at least 100 complete
   causal candidate sequences across at least three calendar years, both
   directions, and at least two volatility regimes. Discovery counts may not
   use realized PnL or future target labels.
3. **Frozen specification.** The sequence, evidence, invalidation, target,
   deadline, overlap rule, and path test are written and hash-frozen before
   calibration or rolling validation is examined.
4. **Adequate independent evidence.** At least 200 resolved calibration/path
   episodes exist overall, including at least 50 per direction. Sparse strata
   remain explicitly uncalibrated rather than being pooled after seeing their
   results.
5. **Incremental value.** On untouched rolling folds, the candidate improves
   hypothesis calibration or net decision utility after costs relative to the
   same system without it. The effect must recur across time and cannot depend
   on one month, one volatility regime, or one threshold.
6. **Overlap and attribution audit.** Before admission, report co-start rate,
   shared frozen invalidation/draw rate, outcome agreement, and incremental
   action-utility contribution against every registered playbook. A candidate
   that mostly recreates an existing episode or cannot receive stable causal
   attribution is rejected, merged, or expressed as an evidence primitive.
7. **Complexity budget.** The new hypothesis replaces a documented coverage
   gap. Only one candidate is promoted per major-version validation cycle.
8. **Fresh final test.** The existing sealed holdout may not authorize the
   expansion. Adding a playbook creates a new major protocol and requires a
   later untouched holdout.

## Candidate handling

Candidate ideas remain in a discovery ledger with four statuses:
`observed → specified → path_validated → admitted/rejected`. Until admission,
they have no model, decision, or trading authority. Rejected candidates and
their tested definitions remain recorded so that the same idea is not silently
reintroduced under a new name.

The present model has no evidence-backed coverage gap that clears these gates.
Therefore the correct current action is to calibrate and falsify the three
registered playbooks, not to add more.

## Post-freeze June-July finding

This section records a finding; it does not rewrite the frozen v2 rule above.
Across June-July development paths, failed-auction and liquidity-sweep
co-decisions remain highly overlapping. In the audited 2024-06-02 18:05
decision, DFP-short and LSR-short share entry and target but have different
structural invalidations. Their playbook-specific enter utilities are separated
by only 0.041R, so the frozen action-margin rule abstains.

Before another profitability test, a successor protocol must preregister two
separate questions:

1. **Action equivalence.** Define whether two same-direction entries with
   different invalidations are distinct actions or evidence for one action.
   If they are one action, freeze the aggregation and plan-arbitration rule. If
   they are distinct, retain the current competition and report the abstention
   as plan uncertainty.
2. **Managed-policy utility.** Keep target-before-original-invalidation as a
   playbook-belief label, but do not equate it with expected trade return.
   Decision utility must be calibrated to the full frozen policy--fill/expiry,
   enter, hold, protect, exit, costs, and deadline--at the actual action clock.

These corrections simplify responsibility boundaries; they do not justify a
fourth playbook. Any revised action-equivalence or managed-policy protocol must
receive a new version and hash before the sealed MBO holdout is opened.
