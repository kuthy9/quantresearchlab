"""What a Brain run is made of, and how the Eye is driven over its window.

``run_identity`` hashes everything a journal's meaning depends on — the tape,
the window, the model, the prompt, the two configs and the Eye's identities —
into a ``run_id``; ``drive`` feeds the registered Eye every completed bar of
the window (warm-up included) and hands each published observation to a
callback, flagging whether it falls inside the emit window."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import pandas as pd

from contract.eye import MarketObservation
from shares.core.eye_factory import build_eye
from shares.core.io import iter_completed_bars, load_ohlcv

ROOT = Path(__file__).resolve().parents[2]
MARKET_TIMEZONE = "America/New_York"
DEFAULT_SOURCE = "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
DEFAULT_MODEL = "configs/model.json"


@dataclass(frozen=True)
class RunWindow:
    source: Path
    warmup_start: str
    emit_start: str
    end: str

    def to_dict(self) -> dict:
        return {
            "source": str(self.source),
            "warmup_start": self.warmup_start,
            "emit_start": self.emit_start,
            "end": self.end,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> tuple[str | None, bool]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return rev, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False


def run_identity(
    *,
    window: RunWindow,
    model: str,
    prompt_sha256: str,
    controller_sha256: str,
    config_sha256: str,
    atomic_identity: str,
    scale_registry_id: str,
    source_sha256: str,
    root: Path = ROOT,
) -> tuple[str, dict]:
    identity = {
        "window": window.to_dict(),
        "source_sha256": source_sha256,
        "model": model,
        "system_prompt_sha256": prompt_sha256,
        "sleep_controller_sha256": controller_sha256,
        "main_brain_config_sha256": config_sha256,
        "atomic_definition_identity": atomic_identity,
        "scale_registry_id": scale_registry_id,
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    rev, dirty = git_revision(root)
    payload = {
        "run_id": run_id,
        **identity,
        "git_revision": rev,
        "git_dirty": dirty,
        "started_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return run_id, payload


def eye_identities(model_path: Path, *, root: Path = ROOT) -> tuple[str, str]:
    """The Eye's atomic definition identity and scale registry id."""
    from eyes.core.semantics import load_semantic_selection
    from shares.core.scale_registry import parse_scale_specs, scale_registry_id

    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    selection = load_semantic_selection(model["semantic_selection"], root=root)
    specs = parse_scale_specs(model["scales"])
    return selection.atomic_definition_identity, scale_registry_id(specs)


def drive(
    window: RunWindow,
    *,
    model_path: Path,
    root: Path,
    on_observation: Callable[[MarketObservation, bool], None],
    progress_every: int = 0,
    log: Callable[[str], None] = print,
) -> int:
    """Feed the Eye every completed bar of the window; return the bars seen.

    The Eye spills cold events to a journal it never empties, so the journal
    lives in a temporary directory that dies with this pass."""
    loaded = load_ohlcv(window.source, start=window.warmup_start, end=window.end)
    frame = loaded.frame
    emit_from = pd.Timestamp(window.emit_start, tz=MARKET_TIMEZONE)
    seen = 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="eye_journal_brain_run_") as journal:
        reader, observer = build_eye(model_path, root=root, audit_journal_dir=journal)
        for bar in iter_completed_bars(frame):
            observation = observer.observe(reader.on_bar(bar))
            seen += 1
            if observation.market_snapshot is not None:
                emitting = observation.asof.tz_convert(MARKET_TIMEZONE) >= emit_from
                on_observation(observation, emitting)
            if progress_every and seen % progress_every == 0:
                log(f"{seen} bars, {observation.asof}, {(time.monotonic() - started) / 60:.1f} min")
        del observer, reader
        gc.collect()
    return seen


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SOURCE",
    "MARKET_TIMEZONE",
    "ROOT",
    "RunWindow",
    "drive",
    "eye_identities",
    "git_revision",
    "run_identity",
    "sha256_file",
]
