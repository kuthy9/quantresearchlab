"""Memory-bounded, resumable sequential replay primitives for calibration.

This module deliberately preserves the production causal order while replacing
the expensive full-object snapshot hash with a small rolling state commitment.
The hash has calibration lineage authority only; it is not a substitute for a
full visual/audit snapshot hash.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any, Iterator, Mapping

import pandas as pd

from .engine import ContinuousSMCEngine
from .io import iter_completed_bars
from .model import AccountState, Bar, EngineSnapshot
from .observation import ExecutionRealityInput
from .simulation import ReplayStep, SequentialPortfolio


CHECKPOINT_FORMAT_VERSION = 1
HASH_MODE = "rolling_causal_state_commitment_v1"


def _rolling_commitment(previous: str, snapshot: EngineSnapshot) -> str:
    """Commit to the small action/risk identity that matters to calibration."""

    thesis_hash = (
        ""
        if snapshot.risk.frozen_thesis is None
        else snapshot.risk.frozen_thesis.thesis_hash
    )
    fields = (
        previous,
        snapshot.observation.asof.isoformat(),
        snapshot.observation.symbol,
        str(snapshot.observation.instrument_id),
        snapshot.decision.selected_action.value,
        snapshot.risk.final_action.value,
        snapshot.decision.best_hypothesis_key or "",
        thesis_hash,
    )
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class CalibrationSequentialReplay:
    """Exact sequential semantics with a lightweight rolling snapshot hash."""

    def __init__(
        self,
        engine: ContinuousSMCEngine | None = None,
        portfolio: SequentialPortfolio | None = None,
        *,
        rolling_commitment: str = "0" * 64,
        simulate_execution: bool = True,
    ) -> None:
        if len(rolling_commitment) != 64:
            raise ValueError("rolling commitment must be a 64-character digest")
        if not simulate_execution and portfolio is not None:
            raise ValueError(
                "a portfolio cannot be supplied when execution simulation is disabled"
            )
        self.engine = engine or ContinuousSMCEngine()
        self.simulate_execution = bool(simulate_execution)
        self.portfolio = (
            (portfolio or SequentialPortfolio())
            if self.simulate_execution
            else None
        )
        self.rolling_commitment = rolling_commitment

    def on_bar(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput,
    ) -> ReplayStep:
        """Run the same causal operations as :class:`SequentialReplay`."""

        if self.portfolio is None:
            closed = ()
            account = AccountState(equity=100_000.0)
            belief_position = account.position
        else:
            closed = self.portfolio.before_bar(bar)
            account = self.portfolio.account(bar.end)
            belief_position = (
                self.portfolio.lifecycle_position
                if self.portfolio.lifecycle_position is not None
                else account.position
            )

        update = self.engine.reader.on_bar(bar)
        observation = self.engine.observer.observe(update, execution)
        if {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(observation.anomalies):
            self.engine.brain.reset()
        belief = self.engine.brain.update(
            observation,
            position=belief_position,
            scene_graph=self.engine.observer.scene_graph,
            scene_delta=self.engine.observer.last_scene_delta,
        )
        decision = self.engine.decision.decide(observation, belief, account)
        risk = self.engine.risk.review(decision, observation, account)
        provisional = EngineSnapshot(
            observation=observation,
            belief=belief,
            decision=decision,
            risk=risk,
            snapshot_hash="",
        )
        self.rolling_commitment = _rolling_commitment(
            self.rolling_commitment,
            provisional,
        )
        snapshot = EngineSnapshot(
            observation=observation,
            belief=belief,
            decision=decision,
            risk=risk,
            snapshot_hash=self.rolling_commitment,
        )
        self.engine._last_snapshot = snapshot

        if self.portfolio is None:
            position = None
        else:
            self.portfolio.after_decision(snapshot)
            self.portfolio.clear_lifecycle_position()
            position = self.portfolio.account(bar.end).position
        return ReplayStep(
            snapshot=snapshot,
            closed_trades=closed,
            position=position,
            account_state=account,
            belief_position_input=belief_position,
        )


def iter_after_source_checkpoint(
    frame: pd.DataFrame,
    last_source_start: pd.Timestamp | None,
    *,
    allow_data_gap_reset: bool = False,
) -> Iterator[Bar]:
    """Resume the registered bar iterator after a committed real source row.

    The checkpoint row is included when constructing the iterator so it becomes
    its causal ``prior``. It is then skipped before yielding. This preserves any
    bounded no-trade densification between that row and the next source row.
    """

    if last_source_start is None:
        yield from iter_completed_bars(
            frame,
            allow_data_gap_reset=allow_data_gap_reset,
        )
        return
    timestamp = pd.Timestamp(last_source_start)
    if timestamp.tzinfo is None:
        raise ValueError("checkpoint source timestamp must be timezone-aware")
    matches = frame.index.get_indexer([timestamp])
    position = int(matches[0])
    if position < 0:
        raise ValueError("checkpoint source row is absent from the bound OHLCV source")
    iterator = iter_completed_bars(
        frame.iloc[position:],
        allow_data_gap_reset=allow_data_gap_reset,
    )
    try:
        prior = next(iterator)
    except StopIteration as exc:
        raise ValueError("checkpoint source row cannot seed the resumed iterator") from exc
    if prior.synthetic_no_trade or prior.start != timestamp:
        raise ValueError("checkpoint does not identify an exact real OHLCV source row")
    yield from iterator


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class ReplayCheckpointStore:
    """Hash-bound atomic local checkpoint storage.

    Pickle is loaded only after the manifest, exact run bindings, regular-file
    checks, and state digest have passed. Checkpoints remain trusted local
    artifacts and must never be accepted from an untrusted source.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    @property
    def exists(self) -> bool:
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            state_file = str(manifest.get("state_file", ""))
        except (OSError, ValueError, TypeError):
            return False
        if re.fullmatch(r"state-[0-9a-f]{64}\.pkl", state_file) is None:
            return False
        state_path = self.root / state_file
        return not state_path.is_symlink() and state_path.is_file()

    def save(
        self,
        state: Mapping[str, Any],
        *,
        bindings: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
        state_hash = hashlib.sha256(raw).hexdigest()
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_sha256": state_hash,
            "state_file": f"state-{state_hash}.pkl",
            "bindings": dict(bindings),
            "last_source_start": (
                None
                if state.get("last_source_start") is None
                else pd.Timestamp(state["last_source_start"]).isoformat()
            ),
            "processed_bars": int(state.get("processed_bars", 0)),
            "decision_rows": int(state.get("decision_rows", 0)),
            "next_shard_index": int(state.get("next_shard_index", 0)),
            "committed_shards": list(state.get("committed_shards", ())),
        }
        state_path = self.root / manifest["state_file"]
        _atomic_bytes(state_path, raw)
        _atomic_bytes(self.manifest_path, _canonical_json_bytes(manifest))
        # Once the new manifest is durable, older content-addressed checkpoint
        # blobs are unreachable.  Keep checkpoint disk use bounded without
        # touching any file outside this exact private checkpoint directory.
        for stale in self.root.glob("state-*.pkl"):
            if (
                stale != state_path
                and not stale.is_symlink()
                and stale.is_file()
                and re.fullmatch(
                    r"state-[0-9a-f]{64}\.pkl",
                    stale.name,
                )
                is not None
            ):
                stale.unlink()
        return manifest

    def load(
        self,
        *,
        expected_bindings: Mapping[str, Any],
        expected_replay_type: type | tuple[type, ...] = CalibrationSequentialReplay,
    ) -> dict[str, Any]:
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise ValueError(
                "checkpoint manifest is not a trusted regular file: "
                f"{self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        if manifest.get("bindings") != dict(expected_bindings):
            raise ValueError("checkpoint bindings differ from the requested replay")
        state_file = str(manifest.get("state_file", ""))
        if re.fullmatch(r"state-[0-9a-f]{64}\.pkl", state_file) is None:
            raise ValueError("checkpoint manifest has an unsafe state filename")
        state_path = self.root / state_file
        if state_path.is_symlink() or not state_path.is_file():
            raise ValueError(
                f"checkpoint state is not a trusted regular file: {state_path}"
            )
        raw = state_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest.get("state_sha256"):
            raise ValueError("checkpoint state hash is invalid")
        state = pickle.loads(raw)
        if not isinstance(state, dict):
            raise ValueError("checkpoint state root must be a mapping")
        for field in (
            "processed_bars",
            "decision_rows",
            "next_shard_index",
            "committed_shards",
        ):
            if state.get(field) != manifest.get(field):
                raise ValueError(f"checkpoint manifest disagrees on {field}")
        if not isinstance(state.get("replay"), expected_replay_type):
            raise ValueError("checkpoint does not contain the expected replay state")
        return state


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CalibrationSequentialReplay",
    "HASH_MODE",
    "ReplayCheckpointStore",
    "iter_after_source_checkpoint",
]
