# v2 pre-test self-review

Review completed: 2026-07-25  
Result: **passed for testing; not passed for an economic-edge claim**

This review was completed before syntax compilation, unit tests, historical
clock audits, MBO materialization, or current-revision replays.

## Architecture and causal logic

The runtime is one directed pipeline:

`completed 1m bar → causal multi-timeframe reader → descriptive observation
and event memory → six playbook/direction beliefs → action utilities →
independent risk veto → later-bar execution and feedback`

The observer has no action method. The brain maintains exactly three
preregistered playbooks in both directions and never emits a buy/sell training
label. Sequence steps are causal, ordered, and latched. Once a sequence is
complete, its plan remains fixed until terminal completion, invalidation,
deadline, or contract change. A later impulse or extreme cannot replace the
active thesis.

The decision layer compares explicit alternatives. The risk engine can veto
entry regardless of belief probability. Original invalidation and targets are
immutable after risk approval; protection is stored separately and must use a
still-visible, newly confirmed level.

## MBO and execution reality

The development MBO reader is memory-bounded. Parquet is scanned in batches and
DBN is iterated record by record. Books are keyed by publisher and instrument.
Databento `A/C/M/R` order-state semantics are implemented; `T/F/N` are no-ops
because normalized fills are accompanied by the cancel record that changes the
displayed book. Snapshot, event-end, bad-clock, aggregate-record, and
maybe-bad-book flags are handled fail-closed.

The observer now receives actual BBO, displayed sizes, top-five depth
imbalance, book time, spread, cost, and fillability inputs. It does not invent a
fixed MBO slippage value. If displayed best-side size cannot cover the requested
quantity, the unknown price impact is a hard veto. Commission remains an
explicit configured execution cost rather than an alpha feature.

Every materialized MBO-minute file is bound to a manifest, strict causal-OHLCV
hash, validation role, source interval, and output hash. The sealed
August-December source cannot be read without a distinct explicit reveal.

## Calibration and leakage boundary

Path tests are registered only after all sequence steps and a causal plan exist.
The decision snapshot, setup clock, sequence completion clock, raw and
calibrated beliefs, plan, protocol hash, model-config hash, model-code hash, and
future resolution are recorded. Later bars are consumed only after the frozen
decision clock. Stop/target ambiguity is adverse-first.

Belief calibration accepts identity/unvalidated **raw** probability only from
the preregistered calibration window. It uses fixed quantile bins, Beta(1,1)
smoothing, and weighted monotone PAVA. It refuses sparse, single-level, mixed
protocol, mixed config, mixed code, or holdout data. No threshold is selected
from realized return.

The latest OHLCV interval and the August-December MBO interval remain sealed.
Earlier project generations inspected other historical periods, so they are
development or rolling validation rather than pristine holdouts.

## Visualization and AI boundary

Decision images show completed 4H, 1H, 5m, and 1m candles with their registered
primitives; event order and persistence; raw/calibrated beliefs and phase
duration; ordered setup steps; frozen entry/invalidation/target sources;
action utilities, reasons, vetoes; and MBO BBO/depth. Off-screen plan levels are
rendered as edge labels, so they cannot compress the price axis.

The audited hypothesis is explicit and separate from the final action. Thus an
`abstain` decision still exposes the leading candidate and why it failed.
Future bars are absent from decision images. A replay may seal the first N path
tests and later emit a different reveal image and JSON audit record tied to the
same decision hash.

AI review accepts diagnostic issue codes only. Each issue maps to a causal
sequence-primitive proposal and remains `unvalidated`; direct action or outcome
labels are rejected.

## Static defects found and corrected before tests

1. Complete sequences could be replaced by a later impulse before terminal
   resolution. Active complete setup identity is now retained.
2. A complete plan was recomputed each minute, allowing target and invalidation
   drift. It is now frozen at sequence completion; target consumption invalidates.
3. Calibration artifacts could train on already calibrated probability and
   could outlive changed model code. They now use raw identity beliefs and carry
   a model-code fingerprint.
4. Empty funnel/path outputs had unstable schemas. Typed empty schemas are now
   written.
5. DBN enum integers were initially at risk of being interpreted as ASCII.
   Action and side enum normalization are now separate and vendor-defined.
6. One invalid publisher event could remove another publisher's valid book.
   Latest books and invalid states are now publisher/instrument scoped.
7. BBO slippage initially contained a constant one-tick assumption. It was
   removed; insufficient visible depth now fails closed.
8. A consumed protection level could remain eligible for stop tightening. Risk
   now rechecks current visibility and side.
9. Event “duration” was actually event age. Stateful persistence is now closed
   explicitly; instantaneous events report zero persistence.
10. `abstain`-first visuals could hide the leading candidate plan and sequence.
    The audited hypothesis is now selected independently from the final action.
11. MBO-derived replay files were not manifest/hash/role bound at load time.
    The replay now verifies all three before reading the table.
12. The declared Python minimum was incompatible with the code's union syntax.
    It is now Python 3.10+.
13. The first real MBO materialization rejected every vendor snapshot because
    it required the leading `R` clear record to carry the snapshot flag. In the
    source protocol the clear precedes snapshot-flagged `A` records, with LAST
    on the final add. Validation now requires exactly that ordered shape while
    still rejecting mixed actions, missing adds, partial events, and bad books.
14. The first full-month attempt revealed a throughput defect: top-five depth
    was sorted and copied after every L3 event even though the consumer needs
    one observation only at each completed minute. Event replay and book
    validity remain sequential and unchanged, but BBO/depth capture is now
    deferred to minute boundaries. A public `snapshot()` performs the same
    calculation, a regression test covers deferred capture, and the original
    two-hour materialization is retained for exact output comparison before
    restarting a month. Partition-completion messages add observability without
    changing data.
15. Deferred capture alone left full-month throughput impractical. The legacy
    Parquet source is already partitioned by UTC receive date and each daily
    partition begins with a complete vendor snapshot. Parquet materialization
    may now use up to four deterministic day workers, but every worker first
    verifies the ordered `R` plus snapshot-flagged event. A missing partition
    produces invalid execution rows; a missing daily baseline aborts. Results
    are merged and checked against the complete OHLCV decision-clock set.
    Native DBN remains sequential because it is one continuous file.
16. The first end-to-end MBO smoke replay found a reveal-only bug: a path test
    can terminate at its decision boundary on contract change or deadline,
    leaving zero completed future price bars. The visualizer previously crashed.
    It now permits an empty future panel only for a bound terminal result after
    the decision and before the first future bar completes, labels that
    condition explicitly, and keeps the
    reveal in a physically separate file. It never includes a bar completed
    after the registered resolution merely to make a chart.
17. The corrected smoke replay then exposed inconsistent interval endpoints:
    decisions included `end`, while MBO rows used `[start, end)`. Replay output
    is now explicitly decision-clock half-open. MBO materialization loads one
    extra prior OHLCV minute so a bar completing exactly at `start` is eligible,
    and retains only clocks `start <= decision_time < end`. Replay asserts the
    same invariant before writing any result.
18. Direct inspection of the decision image exposed a state-machine defect:
    completed/invalidated hypotheses retained their latched sequence indefinitely
    whenever no new candidate existed, and could display an unrelated fresh
    plan thousands of minutes later. Terminal phases now acknowledge one causal
    update, then clear sequence and plan and return inactive unless a strictly
    later setup clock produces a different setup identity. Open-position state
    still takes precedence. This infrastructure correction changes the
    registry fingerprint, so the registry is frozen anew as
    `2.0.0-preregistered.2` and each unchanged causal playbook protocol is
    `1.0.1`; prior path/calibration artifacts cannot silently mix with it.
19. Parallel MBO progress originally waited for results in task submission
    order, hiding already completed later days. Completion is now reported with
    `as_completed`; rows are still sorted by decision clock and paths remain
    manifest-sorted, so display order cannot change materialized content.
20. Visual inspection also found clustered 1m/5m labels and overlapping
    off-screen plan badges. All recent event points remain plotted and the full
    ordered memory remains in the audit panel, but only the four latest visible
    events are text-annotated per price panel. Off-screen entry, invalidation,
    and target badges now receive separate edge slots. This is display-only and
    does not filter model inputs.
21. Warmup comparison was not auditable from the decision table because it
    stored only the leading hypothesis. Replay rows now also record each
    timeframe's ready flag, observation anomalies, all-hypothesis phase counts,
    active setup count, complete sequence count, and plan count. These are
    diagnostics only and cannot affect actions.
22. The AI primitive adapter and chart layer were present, but replay had no
    manifestable input path for pre-reveal reviews. The optional
    `--ai-review-directory` now looks up only an exact
    `<decision_hash>.json`, converts it before a sealed path audit is created,
    and writes a proposal ledger. The existing adapter rejects direct action,
    outcome, profit, and future-path fields; accepted proposals remain
    `unvalidated` and are never passed to the observer, brain, decision, risk,
    or execution components. Missing files mean no AI proposal and do not alter
    replay behavior.
23. The visual renderer was corrected to support a terminal path outcome at the
    decision boundary, but replay still skipped an audit whenever its
    post-decision candle list was empty. That stale guard is removed. The
    renderer itself accepts an empty reveal only for registered boundary
    outcomes and otherwise fails closed, so the change closes the audit gap
    without permitting a fabricated future path.
24. A full calibration replay previously retained every minute-level decision
    row even though the reliability fitter needs only frozen path results. The
    new `run_calibration_paths.py` keeps the engine and path recorder in one
    chronological stream, observes setups only inside the preregistered
    calibration window, and writes only path artifacts. Warmup bars can
    update causal state but cannot register training episodes. The constant
    execution object is explicitly path-only, has zero execution authority, and
    continues to hard-veto entries; profitability is not evaluated. Source,
    contract-selection, config, validation-protocol, and model-code identities
    are verified before output.
25. Empty future-reveal rendering accepted only terminal outcome names, but it
    did not require the registered resolution clock to equal the sealed decision
    clock. A manually omitted later path could therefore be rendered as if it
    ended at the boundary. Empty reveals now require exact clock equality in
    addition to a boundary-eligible outcome and decision-hash/plan binding.
26. The first full calibration stream failed closed on 2021-11-26 at 13:00 ET.
    The generic CMES calendar ended the post-Thanksgiving equity-index session
    at 13:00, while the NQ source continued through 13:14. CME's published 2021
    holiday schedule specifies a 12:15 CT / 13:15 ET early close. Explicit
    post-Thanksgiving 13:15 ET overrides are now registered for every
    2017–2026 sample year. The final 13:00–13:15 partial H1 candle must contain
    exactly 15 observed/expected minutes and may not emit before 13:15.
27. Full-month image inspection showed that four annotated recent events could
    still collide near the right edge of dense 1m/5m panels. All eight recent
    event markers and all ordered-memory rows remain visible, but only the last
    two markers now receive on-price text, using fixed above-left and below-left
    offsets. This is display-only and cannot change model evidence.
28. The first corrected calibration retry exposed a performance defect:
    `ContinuousSMCEngine.on_bar` serialized and hashed the complete observation,
    six beliefs, action utilities, and risk assessment on every minute even
    though calibration consumes only eyes, brain, and newly frozen paths. The
    calibration stream now calls the same reader, observer, and brain directly.
    It computes a full observation/belief content hash only on a minute with a
    previously unseen complete setup, then passes that sealed snapshot to the
    unchanged `FrozenPathTestRecorder`; minutes without a new setup do not call
    `observe` because later-bar resolution already occurs through `on_bar`.
    Decision, risk, execution, and model runtime code are unchanged.
29. Warmup state can contain a complete sequence that began before the
    calibration boundary. On the first in-window minute, the generic recorder
    would otherwise freeze that carry-in setup as a training episode. The fast
    stream now presents the path recorder only hypotheses whose
    `sequence.started_at` is inside the calibration window. Warmup still
    initializes market state, but cannot contribute an episode identity.
30. The completed calibration exposed a deterministic phase-gate mismatch:
    every fitted monotone reliability map ended below the provisional global
    `executable_probability=0.66`, so attaching the artifact would make
    `executable` unreachable for all three playbooks. No profitability or
    rolling-validation threshold search is used to correct this. The gate is
    frozen at `0.50`, meaning only that the registered target-first path is
    calibrated as more likely than invalidation/deadline. Reward distance,
    cost, uncertainty, remaining time, fillability, and best-action advantage
    remain exclusively in the decision layer; risk retains independent veto
    authority. The calibration artifact is bound to the unchanged registry and
    model-code hashes, uses 10,695 eligible 2022-2023 paths, and declares
    `holdout_used=false`.
31. The first calibrated rolling replay showed 80 risk-approved entry
    instructions but only 53 filled-and-closed trades. Review of
    `SequentialPortfolio` confirmed that each instruction is a one-next-bar
    limit attempt: an untouched order is cleared, not silently converted to a
    fill. The replay report previously omitted those order outcomes and labeled
    only the highest-probability hypothesis, which can differ from the
    utility-selected action hypothesis. The reporting layer now writes both
    identities plus a separate `entry_attempts.parquet` with filled,
    unfilled/expired, or end-censored outcomes. This is audit instrumentation
    only: it does not change eyes, beliefs, utilities, risk, fills, registry,
    model code, calibration, or the sealed holdout. Approval rows are appended
    only after the registered decision start, so warmup attempts cannot enter
    the reported order ledger; fills are identified from either an open
    position or a same-bar closed trade.
32. Before the second development-MBO month, source integrity was reviewed
    independently of the output manifest. The partitioned materializer
    previously trusted filenames under the allowed root and bound only the
    resulting minute file. It now requires `legacy_parquet/manifest.json`,
    resolves every selected path below that root, hashes every selected
    partition before worker launch, and rejects an absent or mismatched entry.
    The output manifest records the source-manifest hash and verified partition
    count. This adds no market feature and cannot alter a valid replay; it makes
    corrupted input fail before any new test result exists.
33. The first full June materialization exposed 1,370 invalid book minutes,
    all inside the UTC partition that spans the strict NQM4→NQU4 roll. The
    source was intact. Root cause: instrument filtering happened before
    `F_LAST` packet grouping, while one complete vendor packet can contain
    records for multiple books. That discarded packet delimiters and fed mixed
    keys to a single-book state machine. The materializer now reads each
    partition in complete source order, groups the original vendor packet
    first, then partitions that packet by `(publisher_id, instrument_id)` and
    restores `F_LAST` on each atomic book subevent. Each selected book must
    still begin with its own complete snapshot. No record is reordered, and no
    invalid minute is imputed. The change is confined to source
    materialization; model code and the probability calibration hash are
    unchanged. The future DBN branch uses the same full-packet-before-selection
    rule, but remains unopened behind the holdout seal.
34. The corrected full-packet replay then failed on legitimate zero and
    negative prices belonging to calendar-spread instruments in the same NQ
    feed packet. The original record parser incorrectly imposed an
    outright-contract positive-price invariant before instrument selection.
    Materialization now inspects every raw row's `F_LAST`, parses only selected
    outright records, and reconstructs the selected subevent boundary when the
    complete vendor packet closes. Calendar-spread prices are neither parsed
    nor used, while the NQ outright still rejects every nonpositive observed
    price. No price is replaced, clipped, or synthesized. A batch-boundary
    test proves that a negative spread may close the packet without losing the
    selected outright's `F_LAST`; the complete suite passes 57 tests before the
    retry.
35. The next full June retry exposed the Databento Sunday snapshot protocol:
    at 11:04 UTC the feed emits a complete clear-only packet, followed at 12:00
    UTC by a separate complete packet of snapshot-flagged adds, both before the
    first tradable OHLCV minute. Requiring reset and snapshot adds in one vendor
    packet incorrectly rejected 9, 16, 23, and 30 June. Daily replay now holds
    exactly one clear-only preamble and initializes only if the next accepted
    snapshot packet consists entirely of snapshot-flagged adds. An ordinary
    update, a missing snapshot, or a malformed reset still fails closed. The
    real 9 June partition produces 119/119 valid book minutes, the 17 June
    dual-contract partition remains 1,380/1,380 valid, and all 58 tests pass
    before the full-month retry.

## Remaining claim boundary

- All belief values and decision coefficients are provisional until the
  preregistered calibration and rolling-validation gates have adequate episodes.
- BBO gives actual spread and displayed liquidity, but not latency, hidden
  liquidity, or price impact beyond the visible best level. Unknown impact is
  vetoed rather than guessed.
- OHLCV cannot reveal intrabar event order. It remains adverse-first even when
  MBO is present; an MBO event-order extension would require a separately
  registered execution protocol.
- The exchange calendar implementation must still pass a full 2017-2026 gap
  audit before full-period results are accepted.
- AI primitive proposals have no model authority until a later registered path
  test accepts them.
- Passing this review authorizes tests only. It does not establish profitability,
  absence of overfitting, or readiness for capital.

## Post-suite addendum: clock-audit harness

After the initial 43-test suite passed, `scripts/audit_market_clock.py` was
added for the next validation stage. It is read-only, requires the exact
preregistered source hash, routes every yielded minute through the production
reader, accounts for synthetic no-trade minutes, and writes a new report without
overwriting an existing one. Its static review passed before its first run.

The first full-history run stopped at a two-minute open-market hole on
2017-01-03. This was a valid audit failure: Databento OHLCV omits intervals
without a trade, while the first implementation filled exactly one missing
minute and recognized closures only at exact boundary timestamps.

Before rerunning tests, the correction was statically reviewed against these
fail-closed rules:

1. An independent registered exchange calendar classifies every missing
   interval minute as open or closed; source-row absence does not define the
   market calendar.
2. Only unchanged-contract, open-market runs of at most five minutes are
   synthesized with prior-close OHLC and zero volume.
3. Closed-market minutes are never synthesized.
4. More than five open minutes, a contract change across an open gap, or any
   residual unregistered discontinuity raises `DataContinuityError`.
5. Ordinary 17:00-18:00 maintenance, the historical 16:15-16:30 pause,
   registered early closes, abbreviated Good Friday release sessions, and the
   explicit 2026 Juneteenth close are represented independently of observed
   trades.
6. The five-minute bound is a data-integrity limit, not a trading parameter and
   is not tuned against profitability.

Targeted tests now cover a three-minute no-trade run, a six-minute hard
failure, a weekend closure with no synthetic bars, and holiday/Juneteenth
membership. This addendum authorizes the corrected clock suite and full audit;
it does not authorize filtering or silently accepting off-session source rows.

The first all-gap diagnostic then exposed a DST implementation defect before
producing statistics: flooring a timezone-aware New York timestamp inside the
repeated fall-back hour asked pandas to infer an ambiguous offset. The source
instant was already unambiguous. Calendar rounding now happens on the UTC
timeline before conversion to New York, and both occurrences of the repeated
hour are covered by a targeted test. This is a clock correctness fix, not a
change to registered session hours.

The full gap distribution also showed that the generic `CMES` calendar omitted
historical NQ early closes on 3 July and Juneteenth closes in 2022-2025.
Explicit equity-index close overrides were added only where the observed
boundary agrees with the registered holiday schedule; they are calendar facts,
not inferences from later price outcomes. Unknown intervals such as the
2019-02-26 and 2020-02-28 source-wide holes remain unresolved and fail closed.
