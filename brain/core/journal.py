"""The BrainJournal: an episode's full reasoning record, hash-chained.

One JSONL file per episode under ``episodes/``; every line carries ``seq``,
``known_at``, ``prev_hash`` and its own ``hash`` over the previous hash and
the canonical payload, so an edited line breaks the chain.  The writer refuses
a record that steps back in time or a state whose revision skips, and the
reader recomputes the chain on demand.  ``state`` and ``tick`` records both
carry the revision they produced, so revisions are contiguous across the two.  Nothing in a record may postdate its
``known_at``: the payloads are built by ``eye_view`` / ``main_brain`` from the
current bar alone, and the replay tool rebuilds them from the Eye to prove it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from brain.core.llm_client import LLMReply
from contract.brain.llm import canonical_json
from contract.brain.state import isoformat_utc, parse_utc
from contract.market.primitives import FrozenDict, aware_timestamp

RECORD_KINDS: tuple[str, ...] = (
    "episode_opened",
    "wake",
    "llm_call",
    "state",
    "tick",
    "opportunity",
    "trade",
    "incident",
    "sleep",
)
RUN_FILE = "run.json"
INDEX_FILE = "index.jsonl"
EPISODES_DIR = "episodes"


class JournalError(RuntimeError):
    """The journal would become inconsistent, or already is."""


def record_hash(prev_hash: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(payload)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JournalRecord:
    seq: int
    record: str
    episode_id: str
    known_at: pd.Timestamp
    prev_hash: str
    hash: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "known_at", aware_timestamp(self.known_at, name="journal.known_at"))
        object.__setattr__(self, "payload", FrozenDict(dict(self.payload)))

    def hashed_body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "record": self.record,
            "episode_id": self.episode_id,
            "known_at": isoformat_utc(self.known_at),
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.hashed_body(), "prev_hash": self.prev_hash, "hash": self.hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JournalRecord":
        return cls(
            seq=int(payload["seq"]),
            record=str(payload["record"]),
            episode_id=str(payload["episode_id"]),
            known_at=parse_utc(payload["known_at"]),
            prev_hash=str(payload["prev_hash"]),
            hash=str(payload["hash"]),
            payload=payload["payload"],
        )


@dataclass
class _EpisodeCursor:
    last_hash: str
    last_known_at: pd.Timestamp
    first_known_at: pd.Timestamp
    seq: int
    last_revision: int | None
    closed: bool


class BrainJournal:
    def __init__(self, run_dir: Path, *, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        (self.run_dir / EPISODES_DIR).mkdir(parents=True, exist_ok=True)
        self._cursors: dict[str, _EpisodeCursor] = {}

    def write_run(self, run: Mapping[str, Any]) -> None:
        (self.run_dir / RUN_FILE).write_text(
            json.dumps(dict(run), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _episode_path(self, episode_id: str) -> Path:
        return self.run_dir / EPISODES_DIR / f"{episode_id}.jsonl"

    def open_episode(self, episode_id: str, known_at: pd.Timestamp) -> JournalRecord:
        if episode_id in self._cursors or self._episode_path(episode_id).exists():
            raise JournalError(f"episode {episode_id!r} already exists")
        known_at = aware_timestamp(known_at, name="journal.known_at")
        self._cursors[episode_id] = _EpisodeCursor(self.run_id, known_at, known_at, -1, None, False)
        return self._append("episode_opened", episode_id, known_at, {"episode_id": episode_id, "run_id": self.run_id})

    def write(
        self, record: str, *, episode_id: str, known_at: pd.Timestamp, payload: Mapping[str, Any]
    ) -> JournalRecord:
        if record not in RECORD_KINDS or record in ("episode_opened", "sleep"):
            raise JournalError(f"record kind {record!r} cannot be written directly")
        return self._append(record, episode_id, aware_timestamp(known_at, name="journal.known_at"), payload)

    def close_episode(
        self, episode_id: str, known_at: pd.Timestamp, payload: Mapping[str, Any]
    ) -> JournalRecord:
        record = self._append("sleep", episode_id, aware_timestamp(known_at, name="journal.known_at"), payload)
        cursor = self._cursors[episode_id]
        cursor.closed = True
        revisions = 0 if cursor.last_revision is None else cursor.last_revision + 1
        line = {
            "episode_id": episode_id,
            "first_known_at": isoformat_utc(cursor.first_known_at),
            "last_known_at": isoformat_utc(cursor.last_known_at),
            "revisions": revisions,
            "records": cursor.seq + 1,
        }
        with (self.run_dir / INDEX_FILE).open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(line) + "\n")
        return record

    def _append(
        self, record: str, episode_id: str, known_at: pd.Timestamp, payload: Mapping[str, Any]
    ) -> JournalRecord:
        cursor = self._cursors.get(episode_id)
        if cursor is None:
            raise JournalError(f"episode {episode_id!r} is not open")
        if cursor.closed:
            raise JournalError(f"episode {episode_id!r} is closed")
        if known_at < cursor.last_known_at:
            raise JournalError(
                f"{record} at {isoformat_utc(known_at)} precedes the last record at {isoformat_utc(cursor.last_known_at)}"
            )
        if record in ("state", "tick"):
            revision = payload.get("revision")
            expected = 0 if cursor.last_revision is None else cursor.last_revision + 1
            if revision != expected:
                raise JournalError(f"{record} revision {revision!r} is not the expected {expected}")
        body = {
            "seq": cursor.seq + 1,
            "record": record,
            "episode_id": episode_id,
            "known_at": isoformat_utc(known_at),
            "payload": json.loads(canonical_json(payload)),
        }
        digest = record_hash(cursor.last_hash, body)
        entry = JournalRecord(
            seq=body["seq"], record=record, episode_id=episode_id, known_at=known_at,
            prev_hash=cursor.last_hash, hash=digest, payload=body["payload"],
        )
        with self._episode_path(episode_id).open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry.to_dict()) + "\n")
        cursor.last_hash = digest
        cursor.last_known_at = known_at
        cursor.seq = body["seq"]
        if record in ("state", "tick"):
            cursor.last_revision = int(payload["revision"])
        return entry


class JournalReader:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def run(self) -> Mapping[str, Any]:
        path = self.run_dir / RUN_FILE
        if not path.exists():
            return FrozenDict({})
        return FrozenDict(json.loads(path.read_text(encoding="utf-8")))

    def run_id(self) -> str:
        run = self.run()
        if "run_id" in run:
            return str(run["run_id"])
        raise JournalError("run.json carries no run_id")

    def episode_ids(self) -> tuple[str, ...]:
        directory = self.run_dir / EPISODES_DIR
        if not directory.exists():
            return ()
        return tuple(sorted(path.stem for path in directory.glob("*.jsonl")))

    def records(self, episode_id: str) -> tuple[JournalRecord, ...]:
        path = self.run_dir / EPISODES_DIR / f"{episode_id}.jsonl"
        if not path.exists():
            raise JournalError(f"episode {episode_id!r} has no journal file")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(JournalRecord.from_dict(json.loads(line)))
        return tuple(rows)

    def verify_chain(self, episode_id: str, *, run_id: str | None = None) -> None:
        records = self.records(episode_id)
        if not records:
            raise JournalError(f"episode {episode_id!r} is empty")
        expected_prev = run_id if run_id is not None else records[0].prev_hash
        last_known_at = None
        for index, record in enumerate(records):
            if record.seq != index:
                raise JournalError(f"{episode_id}: seq {record.seq} at position {index}")
            if record.prev_hash != expected_prev:
                raise JournalError(f"{episode_id}: prev_hash mismatch at seq {index}")
            if record_hash(record.prev_hash, record.hashed_body()) != record.hash:
                raise JournalError(f"{episode_id}: hash mismatch at seq {index}")
            if last_known_at is not None and record.known_at < last_known_at:
                raise JournalError(f"{episode_id}: known_at steps back at seq {index}")
            if record.record not in RECORD_KINDS:
                raise JournalError(f"{episode_id}: unknown record kind {record.record!r} at seq {index}")
            expected_prev = record.hash
            last_known_at = record.known_at

    def recorded_replies(self) -> Mapping[str, LLMReply]:
        replies: dict[str, LLMReply] = {}
        for episode_id in self.episode_ids():
            for record in self.records(episode_id):
                if record.record != "llm_call" or record.payload.get("reply") is None:
                    continue
                reply = record.payload["reply"]
                replies[str(record.payload["input_sha"])] = LLMReply(
                    content=reply["content"],
                    reasoning_content=reply.get("reasoning_content"),
                    usage=reply.get("usage", {}),
                    latency_ms=int(reply.get("latency_ms", 0)),
                    model=str(reply.get("model", "")),
                )
        return FrozenDict(replies)

    def recorded_incidents(self) -> Mapping[str, str]:
        """``input_sha`` → incident kind, for every llm_call that ended in one."""
        incidents: dict[str, str] = {}
        for episode_id in self.episode_ids():
            for record in self.records(episode_id):
                if record.record == "incident" and "input_sha" in record.payload:
                    incidents[str(record.payload["input_sha"])] = str(record.payload["kind"])
        return FrozenDict(incidents)

    def index(self) -> tuple[Mapping[str, Any], ...]:
        path = self.run_dir / INDEX_FILE
        if not path.exists():
            return ()
        return tuple(
            FrozenDict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )


__all__ = [
    "EPISODES_DIR",
    "INDEX_FILE",
    "RECORD_KINDS",
    "RUN_FILE",
    "BrainJournal",
    "JournalError",
    "JournalReader",
    "JournalRecord",
    "record_hash",
]
