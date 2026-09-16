"""``build_eye`` is the one constructor of the registered Eye outside the Eye."""
from __future__ import annotations

from pathlib import Path

from contract.market import Timeframe
from shares.core.eye_factory import CONFIGURED, build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "configs" / "model.json"


def test_build_eye_publishes_a_snapshot_over_one_synthetic_session(tmp_path: Path) -> None:
    reader, observer = build_eye(MODEL, root=ROOT, audit_journal_dir=tmp_path / "journal")
    published = 0
    for bar in session_bars(1):
        observation = observer.observe(reader.on_bar(bar))
        if observation.market_snapshot is not None:
            published += 1
            assert Timeframe.M1 in observation.market_snapshot.timeframe_states
    assert published > 0


def test_configured_sentinel_is_the_default() -> None:
    assert CONFIGURED == "<configured>"
    assert build_eye.__kwdefaults__["audit_journal_dir"] is CONFIGURED
