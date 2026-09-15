# Multi-scale dealing range (2026-09-14)

**Status:** done on `eyes` 2026-09-14 (1243 Eye/contract/shares/brain tests green). Same semantic version (`smc_semantics_v1.3`);
the atomic identity moves because the range protocol's bytes move.

## What changes

The Group 4 tracker keeps one manipulation funnel (one live manipulation at a
time, over pool sweeps and mature-range boundary crossings of every scale) and
runs its *range* side once per registered scale:

- `configs/primitives_range.json` gains `"timeframes": ["15m", "1H", "4H"]`.
  Every `*_h1_bars` parameter keeps its name and value and counts bars of the
  range's own scale (the 15m range needs 8 completed 15m bars, the 4H range 8
  completed 4H bars). Field names on `DealingRangeState`
  (`candidate_real_h1_bars`, `age_h1_bars`) are kept so journals still reduce;
  they mean native bars.
- `CausalRangeAuctionTracker` holds per-scale ATR windows, prior closes,
  clocks and idempotency memos; `_apply_native(candle, zones)` replaces
  `_apply_h1`; at most one live range per scale; a range, its boundary
  inventory and its manipulation source carry the range's scale.
- `RangeFormationFunnelSnapshot` carries `timeframe`; the observation keeps
  one funnel per (scale, clock).
- The observer feeds every registered scale's completed candle and that
  scale's support/resistance, publishes each scale's ranges into its own
  frame, and marks cold pairs per scale.
- Emitter and store contracts use the range's scale instead of `Timeframe.H1`;
  the DTO validators reject 1m only.

## Order (TDD, one commit)

1. RED: `eyes/tests/test_ranges_on_every_scale.py` — protocol names the
   scales; a two-session replay forms a 15m range in the 15m frame with 15m
   sources, its events and boundary inventory carry 15m, one live range per
   scale, and the H1 range is unchanged in kind.
2. Protocol + tracker per-scale state.
3. Observer wiring, contract, emitter, store contracts.
4. Existing Group 4 tests green; identity and pins re-pinned; docs.
