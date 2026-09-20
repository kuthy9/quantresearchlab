# Direction fix — Brain layer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make the Brain's direction explicit (`bias`), judged on the scale whose delivery is live, with the opportunity bound to it by code; trim the calls that carry nothing; measure direction against the tape.

**Architecture:** `bias` joins the LLM reply and the `BrainState` (schema 2); reducer rule 4b drops an opportunity against the bias; the prompt gains the bias section and the expression rule; the controller (schema 4) debounces relation triggers per alias in the runtime; `summarize_run.py` reports `direction_accuracy_60m`; `audit_scales.py --triggers` replays a run's triggers under the configured rule.

**Tech Stack:** Python 3.12, pytest (`env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider`, never `-q`).

**Spec:** [brain/docs/specs/2026-09-18-direction-eye-brain-execution-design.md](../specs/2026-09-18-direction-eye-brain-execution-design.md) §2 (§2.3's phase-bookkeeping item was done in the Eye layer).

## Global Constraints

- Starts only after run E (`5c491789ccef7367`) is summarized against `72ea13c7fcbc1cff` and its receipt is written; the prompt's wording may adjust to what run E showed the model doing with the new facts, the design does not.
- No prices in the LLM contract; `basis` is one sentence of text.
- Every scripted test client builds from `LLM_UPDATE_EXAMPLE`, so the example carries a valid `bias`; a client that proposes a LONG sets `bias.direction = "LONG"`.
- Amended 2026-09-19 after run B (spec §2.6): `Bias.scale` validates against `BIAS_SCALES` (`4H | 1H | 15m`), not `GOVERNING_TIMEFRAMES`; the prompt's bias section carries the hysteresis bullet. The tasks below are as first executed.
- Old journals: `BrainState.from_dict` reads schema 1 (no `bias`) as NEUTRAL on 15m; `summarize_run.py` reads the opportunity's direction when a state has no bias.

---

### Task 1: `bias` in the state and the reply contract

**Files:**
- Modify: `contract/brain/state.py` (`BiasDirection`, `Bias`, `BrainState.bias`, schema 2, `from_dict` upgrade), `contract/brain/llm.py` (`LLMUpdate.bias`, `_BIAS_KEYS`, `LLM_UPDATE_REQUIRED_KEYS`, the example, `parse_update`)
- Test: `brain/tests/test_brain_state.py`, `brain/tests/test_llm_contract.py`

**Interfaces:**
- Produces: `BiasDirection(str, Enum)` `LONG | SHORT | NEUTRAL`; `Bias(direction=BiasDirection.NEUTRAL, scale="15m", basis="")` with `to_dict` / `from_dict` and validation (`scale in GOVERNING_TIMEFRAMES`); `BrainState.bias: Bias`; `LLMUpdate.bias: Bias`; `BRAIN_STATE_SCHEMA_VERSION = 2`.

- [ ] **Step 1: Write the failing tests**

`brain/tests/test_brain_state.py`:

```python
def test_bias_round_trips_and_a_schema_1_state_reads_as_neutral() -> None:
    state = make_state(bias=Bias(BiasDirection.LONG, "15m", "15m active leg long past one ATR after the MSS"))
    again = BrainState.from_json(state.to_json())
    assert again.bias == state.bias and again.schema_version == 2
    payload = json.loads(state.to_json())
    del payload["bias"]
    payload["schema_version"] = 1
    old = BrainState.from_dict(payload)
    assert old.bias == Bias() and old.schema_version == 2


@pytest.mark.parametrize("bad", [dict(scale="1m"), dict(scale="4h"), dict(direction="UP")])
def test_bias_refuses_a_bad_scale_or_direction(bad) -> None:
    with pytest.raises(ValueError):
        Bias(**{"direction": BiasDirection.LONG, "scale": "15m", "basis": "", **bad})
```

`brain/tests/test_llm_contract.py`:

```python
def test_the_example_carries_a_bias_and_a_reply_without_one_is_refused() -> None:
    assert LLM_UPDATE_EXAMPLE["bias"] == {"direction": "NEUTRAL", "scale": "15m", "basis": LLM_UPDATE_EXAMPLE["bias"]["basis"]}
    payload = dict(LLM_UPDATE_EXAMPLE)
    del payload["bias"]
    with pytest.raises(MalformedReply):
        parse_update(json.dumps(payload), evidence_ids=("ev_<id from new_evidence>",), object_ids=("FVG_5m_3", "BSL_1H_1"))
    payload = json.loads(json.dumps(LLM_UPDATE_EXAMPLE))
    payload["bias"] = {"direction": "SHORT", "scale": "1H", "basis": "1H MSS short with the active leg short"}
    update = parse_update(json.dumps(payload), evidence_ids=("ev_<id from new_evidence>",), object_ids=("FVG_5m_3", "BSL_1H_1"))
    assert update.bias == Bias(BiasDirection.SHORT, "1H", "1H MSS short with the active leg short")
```

- [ ] **Step 2: Run to verify they fail** — ImportError on `Bias` / `BiasDirection`.

- [ ] **Step 3: Implement**

`contract/brain/state.py`:

```python
class BiasDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class Bias:
    """The Brain's direction (2026-09-18): one side (or none), the scale
    whose live delivery sets it, and the one-sentence basis."""

    direction: BiasDirection = BiasDirection.NEUTRAL
    scale: str = "15m"
    basis: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _enum(BiasDirection, self.direction, name="bias.direction"))
        if self.scale not in GOVERNING_TIMEFRAMES:
            raise ValueError(f"bias.scale must be one of {list(GOVERNING_TIMEFRAMES)}")
        _text(self.basis, name="bias.basis")

    def to_dict(self) -> dict[str, Any]:
        return {"direction": self.direction.value, "scale": self.scale, "basis": self.basis}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Bias":
        return cls(payload["direction"], payload["scale"], payload.get("basis", ""))
```

`BrainState`: `bias: Bias = field(default_factory=Bias)` before `schema_version`; `BRAIN_STATE_SCHEMA_VERSION = 2`; `to_dict` adds `"bias": self.bias.to_dict()`; `from_dict` accepts `schema_version in (1, 2)`, tolerates a missing `bias` for schema 1 (`missing -= {"bias"}`), and constructs with `bias=Bias.from_dict(payload["bias"]) if "bias" in payload else Bias()` and `schema_version=BRAIN_STATE_SCHEMA_VERSION`. Export `Bias`, `BiasDirection`.

`contract/brain/llm.py`: `LLM_UPDATE_REQUIRED_KEYS` += `"bias"`; `_BIAS_KEYS = frozenset({"direction", "scale", "basis"})`; `LLMUpdate.bias: Bias` (after `opportunity`), in `to_dict`; the example gains `"bias": {"direction": "NEUTRAL", "scale": "15m", "basis": "one sentence: the live delivery that sets the bias, or why no scale is live"}`; `parse_update` reads it after the opportunity:

```python
    raw_bias = _require_keys(payload["bias"], _BIAS_KEYS, name="bias")
    try:
        bias = Bias(
            _enum(BiasDirection, raw_bias["direction"], name="bias.direction"),
            _str(raw_bias["scale"], name="bias.scale"),
            _str(raw_bias["basis"], name="bias.basis"),
        )
    except ValueError as error:
        raise _fail(f"bias: {error}") from None
```

- [ ] **Step 4: Run** `brain/tests/test_brain_state.py brain/tests/test_llm_contract.py` — all passed; then `brain/tests` and `execution/tests/test_stack_e2e.py` to see what the contract change broke (expected: nothing yet, since the example carries the bias).

---

### Task 2: Reducer rule 4b and the state

**Files:**
- Modify: `brain/core/reducer.py` (`empty_state`, `_carry_forward`, the draft, rule 4b), `brain/core/main_brain.py` (nothing: `_prior_view` uses `to_dict`)
- Test: `brain/tests/test_reducer.py`

- [ ] **Step 1: Write the failing tests**

```python
def _opp(direction=TradeDirection.LONG, tf="5m"):
    return Opportunity(OpportunityState.DEVELOPING, direction, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe=tf)


def test_an_opportunity_against_the_bias_is_dropped_and_the_bias_kept() -> None:
    result = apply(quiet(), episode_id="EP", evidence=(), update=upd(opportunity=_opp(TradeDirection.SHORT), bias=Bias(BiasDirection.LONG, "15m", "b")), ctx=ctx())
    assert result.state.opportunity == Opportunity() and result.state.bias.direction is BiasDirection.LONG
    assert "opportunity_against_bias:SHORT" in result.rejections


def test_no_opportunity_under_a_neutral_bias() -> None:
    result = apply(quiet(), episode_id="EP", evidence=(), update=upd(opportunity=_opp(), bias=Bias()), ctx=ctx())
    assert result.state.opportunity == Opportunity() and "opportunity_against_bias:NEUTRAL" in result.rejections


def test_a_thesis_above_the_bias_scale_is_dropped() -> None:
    update = upd(opportunity=_opp(tf="1H"), bias=Bias(BiasDirection.LONG, "15m", "b"))
    result = apply(quiet(), episode_id="EP", evidence=(), update=update, ctx=ctx())
    assert result.state.opportunity == Opportunity() and "opportunity_scale_above_bias:1H" in result.rejections
    update = upd(opportunity=_opp(tf="5m"), bias=Bias(BiasDirection.LONG, "15m", "b"))
    assert apply(quiet(), episode_id="EP", evidence=(), update=update, ctx=ctx()).state.opportunity.state is OpportunityState.DEVELOPING
```

(`upd()` needs `bias=Bias()` in its defaults; `make_state` in `test_brain_state.py` gets `bias=Bias()` too.)

- [ ] **Step 2: Run to verify they fail** — `TypeError: unexpected keyword 'bias'` / rejections missing.

- [ ] **Step 3: Implement** — in rule 4, before the coherence check:

```python
        bias = update.bias
        if bias.direction is BiasDirection.NEUTRAL:
            rejections.append("opportunity_against_bias:NEUTRAL")
        elif opportunity.direction is not None and opportunity.direction.value != bias.direction.value:
            rejections.append(f"opportunity_against_bias:{opportunity.direction.value}")
        governing = opportunity.governing_timeframe
        if governing is not None and _SCALE_LADDER.index(governing) < _SCALE_LADDER.index(bias.scale):
            rejections.append(f"opportunity_scale_above_bias:{governing}")
```

(the existing `any(r.startswith("opportunity_") …)` then drops the opportunity); the draft and `_carry_forward` carry `bias=update.bias` / `prev.bias`; `empty_state` uses `Bias()`.

- [ ] **Step 4: Run** `brain/tests/test_reducer.py brain/tests/test_main_brain.py` — all passed.

---

### Task 3: The prompt

**Files:**
- Modify: `brain/configs/prompts/main_brain_system.md`
- Test: `brain/tests/test_main_brain.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_prompt_defines_the_bias_and_how_the_facts_set_it() -> None:
    text = CONFIG.system_prompt
    for word in ("## Bias", "`bias`", "active_leg_direction", "forming_leg_atr", "displacement_age_bars", "`reset`", "drift_atr", "contains_price", "live delivery"):
        assert word in text, word
    assert "The thesis is judged on its governing scale" not in text
    assert "judged on the bias scale" in text and "judged on the governing scale" in text
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Write the prompt.** Insert before "## The opportunity — a thesis, expressed":

```markdown
## Bias — which scale sets the direction

`bias` is your direction: `LONG`, `SHORT` or `NEUTRAL`, the `scale` whose
delivery sets it, and one sentence of `basis`. Code drops any opportunity
whose direction is not the bias, any opportunity under a NEUTRAL bias, and
any thesis whose `governing_timeframe` is above the bias scale — to change
side, change the bias and say why.

Every scale's `delivery` now says which leg price is in: `active_leg_direction`
is the leg forming from the last confirmed swing (`forming_leg_atr` its
size in that scale's ATRs, signed), `last_leg_direction` the confirmed leg
it left; `phase` follows the active leg. `displacement_direction` and
`displacement_age_bars` date the last displacement on that scale. A
`structure.reset` says an acceptance broke the protected swing on that
side and no structure has confirmed since. `session.drift_atr` is the
session's own drift in 1m ATRs.

- **Live delivery** on a scale: its active leg has travelled at least one
  ATR of that scale (`forming_leg_atr` beyond ±1.0) *and* either a
  displacement in that direction at most three bars old on that scale, or
  an MSS / BOS in that direction as its latest structural event. Anything
  else is location, not direction: a `phase` printed for a leg the active
  leg has left, a displacement twelve bars old, an external direction
  whose protected swing is far away.
- The bias scale is the **15m** unless the 1H or the 4H delivery is live in
  its own right; then the highest live scale sets the bias, and the scales
  above it are premium / discount and the draw on liquidity, never the
  direction.
- A `reset` in a direction makes that side live on that scale until a
  structure confirms. `external_direction: long` with `internal_direction:
  short`, the protected low intact and the active leg long is a pullback
  in an uptrend, not a counter-trend bounce. `drift_atr` and the forming
  legs are evidence of direction; a reading that fights both needs a
  structural event on the bias scale to stand.
- `NEUTRAL` when no scale is live and the 15m active leg disagrees with
  the 15m structure. Say so and propose nothing.

**Expression.** After an MSS or BOS on the bias scale in the bias direction
with the active leg past one ATR, express at the object that
`contains_price` — the order fills now. Name a retracement object below
price (for a LONG) only once the 5m active leg has turned against the
bias; a limit at a level the tape is leaving expires unfilled and the
thesis dies with it.
```

Replace the hard rule "The thesis is judged on its governing scale …" with:

```markdown
- **The invalidation is judged on the governing scale; the direction on
  the bias scale.** `watch_next` names objects on the governing scale or
  one below, with the question each one answers about the thesis; a 5m
  pool crossing price is not a reason to re-examine a 1H reading, and you
  are not woken for it.
```

Add `bias` to "Incremental update — what every reply must answer" ("- The bias: direction, scale, basis (see Bias).") and to the output-contract note (`bias.direction` ∈ LONG | SHORT | NEUTRAL; `bias.scale` ∈ 4H | 1H | 15m | 5m).

- [ ] **Step 4: Run** `brain/tests/test_main_brain.py` — all passed (fix `test_the_prompt_explains_the_execution_view`'s word list only if a word moved).

---

### Task 4: Controller schema 4 — the relation debounce

**Files:**
- Modify: `brain/core/sleep_controller.py` (`SLEEP_CONTROLLER_SCHEMA_VERSION = 4`, `relation_change_debounce_bars: int`), `brain/configs/sleep_controller.json` (`"schema_version": 4`, `"relation_change_debounce_bars": 15` after `relation_change_timeframes`), `brain/core/runtime.py` (`_relation_triggered`, the skip in `_relation_changes`, the record after an UPDATE, the reset with the episode)
- Test: `brain/tests/test_sleep_controller.py`, `brain/tests/test_runtime.py`

- [ ] **Step 1: Write the failing tests**

`test_sleep_controller.py`: `assert CFG.relation_change_debounce_bars == 15` and the schema-3 refusal (a tmp json with `"schema_version": 3` raises `ValueError`), mirroring the existing schema test.

`test_runtime.py`:

```python
def test_a_relation_flip_of_the_same_alias_triggers_once_per_debounce_window() -> None:
    from types import SimpleNamespace
    from brain.tests.test_brain_state import make_state
    from contract.brain.state import RegisteredObject, WatchItem

    runtime = BrainRuntime(controller=CONTROLLER, brain=MainBrain(client=EchoClient(), config=CONFIG, ledger=InMemoryPositionLedger(), sleep=lambda s: None), journal=None, ledger=InMemoryPositionLedger(), tick=0.25)
    registry = {"BSL_15m_1": RegisteredObject("b" * 24, "bsl", "15m")}
    state = make_state(watch_next=(WatchItem("BSL_15m_1", "?"),), destination_candidates=(), object_registry=registry)
    t0 = pd.Timestamp("2022-01-04T15:00:00Z")
    above = lambda at: SimpleNamespace(relation_of=lambda alias: "above_price", known_at=at)
    below = lambda at: SimpleNamespace(relation_of=lambda alias: "below_price", known_at=at)
    runtime._remember_relations(state, above(t0))
    assert runtime._relation_changes(below(t0 + pd.Timedelta(minutes=1))) == ("BSL_15m_1",)
    runtime._relation_triggered_at(("BSL_15m_1",), t0 + pd.Timedelta(minutes=1))
    runtime._remember_relations(state, below(t0 + pd.Timedelta(minutes=1)))
    assert runtime._relation_changes(above(t0 + pd.Timedelta(minutes=5))) == (), "a second flip within the window does not trigger"
    assert runtime._relation_changes(above(t0 + pd.Timedelta(minutes=16))) == ("BSL_15m_1",)
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** — `ControllerConfig.relation_change_debounce_bars` from the JSON (`int`, ≥ 0); runtime keeps `self._relation_triggered: dict[str, pd.Timestamp]`, `_relation_changes` skips an alias whose last trigger is less than `debounce_bars` minutes before `context.known_at`, `_relation_triggered_at(aliases, known_at)` records them, called in the ACTIVE step when the decision is UPDATE with the aliases among `decision.reasons` (those not starting with `ev_`); the dict resets where `self._relations = {}` does. `README` controller table row: schema 4.

- [ ] **Step 4: Run** `brain/tests` — all passed.

---

### Task 5: `audit_scales.py --triggers` and the summarizer's direction metrics

**Files:**
- Modify: `brain/scripts/audit_scales.py` (`replay_triggers`, the `--triggers` mode), `brain/scripts/summarize_run.py` (`direction_accuracy`, the `bias` section)
- Test: `brain/tests/test_audit_scales.py`, `brain/tests/test_summarize_run.py`

**Interfaces:**
- `replay_triggers(calls, *, config) -> list[pd.Timestamp]` — `calls` rows `(known_at, kind, reasons, kinds_by_id)`; keeps a WAKE, an UPDATE with an evidence reason whose kind is not bookkeeping, or a relation alias not triggered within `relation_change_debounce_bars` minutes.
- `direction_accuracy(readings, bars, *, horizon_minutes=60) -> dict` — `readings` `(known_at, "LONG"|"SHORT")`; returns `{"readings": n, "agreed": k, "accuracy": k/n or None}` using the close at `known_at` (the bar ending there) and the close `horizon` later.

- [ ] **Step 1: Write the failing tests**

```python
def test_replay_triggers_keeps_wakes_reactions_and_undebounced_relations() -> None:
    t = pd.Timestamp("2022-01-03T15:00:00Z")
    kinds = {"ev_a": "sweep_confirmed", "ev_b": "delivery_phase_entered"}
    calls = [
        (t, "WAKE", ["ev_a"], kinds),
        (t + pd.Timedelta(minutes=1), "UPDATE", ["ev_b"], kinds),          # bookkeeping only → dropped
        (t + pd.Timedelta(minutes=2), "UPDATE", ["FVG_15m_9"], kinds),     # first relation → kept
        (t + pd.Timedelta(minutes=4), "UPDATE", ["FVG_15m_9"], kinds),     # within 15 min → dropped
        (t + pd.Timedelta(minutes=20), "UPDATE", ["FVG_15m_9"], kinds),    # after → kept
    ]
    kept = replay_triggers(calls, config=CFG)
    assert kept == [t, t + pd.Timedelta(minutes=2), t + pd.Timedelta(minutes=20)]


def test_direction_accuracy_reads_the_close_an_hour_later() -> None:
    bars = [Bar(start=pd.Timestamp("2022-01-03T14:00:00Z") + pd.Timedelta(minutes=i), open=1.0, high=1.0, low=1.0, close=100.0 + i, volume=1.0) for i in range(130)]
    readings = [(pd.Timestamp("2022-01-03T14:10:00Z"), "LONG"), (pd.Timestamp("2022-01-03T14:20:00Z"), "SHORT"), (pd.Timestamp("2022-01-03T15:50:00Z"), "LONG")]
    result = direction_accuracy(readings, bars)
    assert result == {"readings": 2, "agreed": 1, "accuracy": 0.5}
```

(the third reading has no bar an hour later and is not counted; check the `Bar` constructor in `contract/market` and adjust the fields.)

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — in `summarize()`, collect `(known_at, direction)` per state revision: `state["bias"]["direction"]` when present and not NEUTRAL, else the opportunity's direction when its state is not NONE; count `bias_changes` per episode; the summary gains `"bias": {"changes": …, "neutral_revisions": …, "direction_accuracy_60m": direction_accuracy(readings, bars) if bars else None, "opportunities_against_bias": rejections.get("opportunity_against_bias", 0)}`; `render` prints `bias.direction_accuracy_60m.accuracy`, `bias.changes`. `--triggers <run-dir>` in `audit_scales.py` reads the run's `llm_call` records into call rows, prints calls kept under the loaded controller config, and `sharp_move_coverage(bars, call_times=kept)` with `load_bars_for(run)`.
- [ ] **Step 4: Run** `brain/tests/test_audit_scales.py brain/tests/test_summarize_run.py` — all passed; then `audit_scales.py --triggers outputs/brain_journal/5c491789ccef7367` and `… 72ea13c7fcbc1cff` for the receipt.

---

### Task 6: Scripted clients, docs

- [ ] `execution/tests/test_stack_e2e.py` `LongAtTheNearestZone` sets `payload["bias"] = {"direction": "LONG", "scale": "5m", "basis": "scripted"}` when it proposes; `brain/tests/test_main_brain.py` replies that propose set the bias likewise (search `"state": "ACTIONABLE"` / `DEVELOPING` in the tests). Run `execution/tests/test_stack_e2e.py brain/tests/test_summarize_run.py brain/tests/test_main_brain.py`.
- [ ] Docs: `brain/docs/README.md` (BrainState schema 2 + `bias`, reducer rule 4b row, controller schema 4 and the debounce, prompt section list, summarizer `bias` section, `audit_scales.py --triggers`), `brain/docs/specs/2026-09-16-llm-brain-design.md` §6 / §7.2 / §8 amendments (one paragraph each), `AGENTS.md` if it names the schema, the spec's §2 stays the design. `git diff --check`.

---

### Task 7: Verification and run B

- [ ] Full suite: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider` — all passed.
- [ ] `audit_scales.py --triggers` on run E and `72ea13c7` — calls kept and coverage under schema 4, for the receipt.
- [ ] Launch run **B** with the frozen command; summarize against run E and `72ea13c7`:

```bash
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/<run B> --run-dir outputs/brain_journal/5c491789ccef7367 --run-dir outputs/brain_journal/72ea13c7fcbc1cff --write
```

- [ ] Receipt `brain/docs/evidence/2026-09-18_brain_bias_day_run_2022-01-03.md` (bias changes and their bases, `direction_accuracy_60m` across the three runs, `opportunity_against_bias` count, calls and coverage under the debounce, orders, fills, P&L, what the prompt's thresholds bound), indexed in `brain/docs/README.md`; memory updated. Then the execution-layer plan.
