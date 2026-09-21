"""The Risk gate: the Brain proposes a plan, the account says what it can bear.

``RiskGate.assess`` reads a ``TradePlan`` (the geometry code resolved from the
objects the LLM named), an ``AccountSnapshot`` (what the broker reports) and
the executor's own open positions, and returns a ``RiskVerdict``: a quantity
and tick-rounded prices when every check passes, named vetoes when one does
not.  Since 2026-09-17 (Risk v2) the budget is a fraction per thesis grade
(the larger one only at or above ``preferred_reward_risk``), the contracts
are also capped by ``max_leverage`` (open notional over equity, counting the
contracts already open), positions are counted from the executor (the
account nets contracts per symbol) and must share a direction, and two account-level stops are kept by ``observe``: the
session's loss limit (``daily_loss_fraction`` of the session's opening
equity, held for the rest of the session date) and the run's drawdown halt
(``max_drawdown_fraction`` from the equity peak, latched for good).

It never reads the Eye, never adjusts a price the geometry did not give it,
and its config file's sha256 is part of a run's identity.  A sizing veto
means the contract is too large for this stop — it is never a request to
move the stop."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from brain.core.position_ledger import PositionRecord
from contract.brain.state import ThesisGrade, TradeDirection, isoformat_utc
from contract.execution import AccountSnapshot
from contract.risk import RiskVerdict, TradePlan, VetoCode

RISK_SCHEMA_VERSION = 3  # 3 (2026-09-20): order_ttl_bars counts bars of the entry object's scale
SESSION_TIMEZONE = "America/New_York"
# A CME futures session opens at 18:00 New York the evening before its date.
SESSION_OFFSET = pd.Timedelta(hours=6)


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    exchange: str
    currency: str
    point_value: float
    tick_size: float


@dataclass(frozen=True)
class ThesisConfig:
    """How many orders one thesis may place in an episode, and how many 1m
    bars every new expression waits after a stop-out."""

    max_expressions: int
    stop_cooldown_bars: int


@dataclass(frozen=True)
class RiskConfig:
    risk_fraction: Mapping[str, float]  # equity fraction at risk per trade, by ThesisGrade value
    max_open_positions: int
    min_reward_risk: float
    preferred_reward_risk: float  # the ratio at which an A_PLUS thesis earns its larger budget
    daily_loss_fraction: float  # of the session's opening equity; no new entries below it
    max_drawdown_fraction: float  # from the equity peak; the run halts and flattens at it
    max_leverage: float  # open notional over equity, counting the contracts already open
    max_quantity: int
    order_ttl_bars: int  # bars of the entry object's own scale (2026-09-20); the machine converts to 1m bars
    account_max_age_s: float
    # The initial margin one contract holds (an approximation of the
    # exchange's); the available funds cap the quantity at it.
    margin_per_contract: float
    thesis: ThesisConfig
    contract: ContractSpec
    sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "RiskConfig":
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("schema_version") != RISK_SCHEMA_VERSION:
            raise ValueError("unsupported risk schema_version")
        contract = payload["contract"]
        thesis = payload["thesis"]
        fractions = {str(grade): float(value) for grade, value in payload["risk_fraction"].items()}
        config = cls(
            risk_fraction=fractions,
            max_open_positions=int(payload["max_open_positions"]),
            min_reward_risk=float(payload["min_reward_risk"]),
            preferred_reward_risk=float(payload["preferred_reward_risk"]),
            daily_loss_fraction=float(payload["daily_loss_fraction"]),
            max_drawdown_fraction=float(payload["max_drawdown_fraction"]),
            max_leverage=float(payload["max_leverage"]),
            max_quantity=int(payload["max_quantity"]),
            order_ttl_bars=int(payload["order_ttl_bars"]),
            account_max_age_s=float(payload["account_max_age_s"]),
            margin_per_contract=float(payload["margin_per_contract"]),
            thesis=ThesisConfig(max_expressions=int(thesis["max_expressions"]), stop_cooldown_bars=int(thesis["stop_cooldown_bars"])),
            contract=ContractSpec(
                symbol=str(contract["symbol"]), exchange=str(contract["exchange"]), currency=str(contract["currency"]),
                point_value=float(contract["point_value"]), tick_size=float(contract["tick_size"]),
            ),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        grades = {item.value for item in ThesisGrade}
        if set(config.risk_fraction) != grades or not all(0.0 < value < 1.0 for value in config.risk_fraction.values()):
            raise ValueError(f"risk_fraction needs one fraction in (0, 1) per grade {sorted(grades)}")
        if (
            config.max_open_positions < 1 or config.max_quantity < 1 or config.contract.tick_size <= 0.0 or config.margin_per_contract <= 0.0
            or config.min_reward_risk <= 0.0 or config.preferred_reward_risk < config.min_reward_risk
            or not 0.0 < config.daily_loss_fraction < 1.0 or not 0.0 < config.max_drawdown_fraction < 1.0 or config.max_leverage <= 0.0
            or config.thesis.max_expressions < 1 or config.thesis.stop_cooldown_bars < 0
        ):
            raise ValueError("risk config values out of range")
        return config


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10)


class RiskGate:
    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        # The account-level memory ``observe`` keeps: the session's opening
        # equity, the session date the daily stop is latched for, the equity
        # peak and the halt record.
        self._session: date | None = None
        self._session_open_equity: float | None = None
        self._daily_stop_session: date | None = None
        self._peak: float | None = None
        self._halt: dict[str, Any] | None = None

    @property
    def config(self) -> RiskConfig:
        return self._config

    # ------------------------------------------------------------ account-level stops

    @staticmethod
    def session_date(asof: pd.Timestamp) -> date:
        return (pd.Timestamp(asof).tz_convert(SESSION_TIMEZONE) + SESSION_OFFSET).date()

    @property
    def halted(self) -> bool:
        return self._halt is not None

    @property
    def halt_record(self) -> dict[str, Any] | None:
        return None if self._halt is None else dict(self._halt)

    def daily_stopped(self, asof: pd.Timestamp) -> bool:
        return self._daily_stop_session is not None and self._daily_stop_session == self.session_date(asof)

    def observe(self, account: AccountSnapshot, asof: pd.Timestamp) -> None:
        """Read the account's equity once per bar: the session's opening
        equity, the daily stop, the peak and the drawdown halt."""
        asof = pd.Timestamp(asof).tz_convert("UTC")
        equity = float(account.equity)
        session = self.session_date(asof)
        if session != self._session:
            self._session = session
            self._session_open_equity = equity
        assert self._session_open_equity is not None
        if self._daily_stop_session != session and equity <= self._session_open_equity * (1.0 - self._config.daily_loss_fraction):
            self._daily_stop_session = session
        self._peak = equity if self._peak is None else max(self._peak, equity)
        if self._halt is None and equity <= self._peak * (1.0 - self._config.max_drawdown_fraction):
            self._halt = {
                "at": isoformat_utc(asof), "equity": equity, "peak": self._peak,
                "drawdown": round((self._peak - equity) / self._peak, 6),
            }

    # ------------------------------------------------------------ the plan

    def assess(
        self, plan: TradePlan | None, account: AccountSnapshot, *, asof: pd.Timestamp, positions: Sequence[PositionRecord] = ()
    ) -> RiskVerdict:
        cfg = self._config
        contract = cfg.contract
        if plan is None:
            return RiskVerdict(False, (VetoCode.NO_PLAN,), ("no actionable plan",))
        if self._halt is not None:
            return RiskVerdict(False, (VetoCode.HALTED,), (f"the run halted at {self._halt['at']} ({self._halt['drawdown']:.1%} below the equity peak)",))
        if self.daily_stopped(asof):
            return RiskVerdict(False, (VetoCode.DAILY_STOP,), (f"the session of {self._daily_stop_session} lost {cfg.daily_loss_fraction:.1%} of its opening equity; no new entries today",))

        vetoes: list[VetoCode] = []
        reasons: list[str] = []
        age = (pd.Timestamp(asof).tz_convert("UTC") - account.asof).total_seconds()
        if age > cfg.account_max_age_s:
            vetoes.append(VetoCode.STALE_DATA)
            reasons.append(f"account snapshot is {age:.0f}s old (limit {cfg.account_max_age_s:.0f}s)")
        ours = tuple(positions)
        if ours:
            if len(ours) >= cfg.max_open_positions:
                vetoes.append(VetoCode.EXPOSURE)
                reasons.append(f"{len(ours)} open position(s) (limit {cfg.max_open_positions})")
            elif any(position.direction is not plan.direction for position in ours):
                vetoes.append(VetoCode.EXPOSURE)
                reasons.append(f"an open position in the opposite direction to {plan.direction.value}")
        elif account.net_position(contract.symbol) != 0:
            vetoes.append(VetoCode.EXPOSURE)
            reasons.append(f"a position in {contract.symbol} that this session did not open")
        if account.open_entry_orders():
            vetoes.append(VetoCode.WORKING_ORDER)
            reasons.append("an entry order is already working")
        if vetoes:
            return RiskVerdict(False, tuple(vetoes), tuple(reasons))

        geometry = plan.geometry
        long = plan.direction is TradeDirection.LONG
        entry, stop, target = geometry.entry_price, geometry.stop_price, geometry.target_price
        if (long and not stop < entry) or (not long and not stop > entry):
            vetoes.append(VetoCode.INVALID_STOP)
            reasons.append(f"stop {stop} is not on the losing side of entry {entry} for {plan.direction.value}")
        if (long and not target > entry) or (not long and not target < entry):
            vetoes.append(VetoCode.INVALID_TARGET)
            reasons.append(f"target {target} is not on the winning side of entry {entry} for {plan.direction.value}")
        if vetoes:
            return RiskVerdict(False, tuple(vetoes), tuple(reasons))

        limit_price = round_to_tick(entry, contract.tick_size)
        stop_price = round_to_tick(stop, contract.tick_size)
        target_price = round_to_tick(target, contract.tick_size)
        risk_points = abs(limit_price - stop_price)
        reward_points = abs(target_price - limit_price)
        if risk_points <= 0.0:
            return RiskVerdict(False, (VetoCode.INVALID_STOP,), ("stop rounds onto the entry",))
        reward_risk = reward_points / risk_points
        if reward_risk < cfg.min_reward_risk:
            return RiskVerdict(False, (VetoCode.REWARD_RISK,), (f"reward-to-risk {reward_risk:.2f} < {cfg.min_reward_risk}",))

        grade = ThesisGrade.A_PLUS if plan.grade is ThesisGrade.A_PLUS and reward_risk >= cfg.preferred_reward_risk else ThesisGrade.BASE
        fraction = cfg.risk_fraction[grade.value]
        risk_budget = account.equity * fraction
        per_contract = risk_points * contract.point_value
        by_budget = int(math.floor(risk_budget / per_contract)) if per_contract > 0.0 else 0
        if by_budget < 1:
            return RiskVerdict(False, (VetoCode.POSITION_SIZE,), (
                f"risk budget {risk_budget:.2f} ({fraction:.1%} of equity) buys no contract at {per_contract:.2f} per contract: "
                f"the contract is too large for this stop",
            ))
        notional = limit_price * contract.point_value
        already_open = abs(account.net_position(contract.symbol)) if ours else 0
        by_leverage = (int(math.floor(account.equity * cfg.max_leverage / notional)) if notional > 0.0 else 0) - already_open
        if by_leverage < 1:
            return RiskVerdict(False, (VetoCode.LEVERAGE,), (
                f"{cfg.max_leverage:g}× leverage on equity {account.equity:.2f} holds no further contract ({notional:.0f} notional each; "
                f"{already_open} already open)",
            ))
        by_margin = int(math.floor(account.available_funds / cfg.margin_per_contract))
        if by_margin < 1:
            return RiskVerdict(False, (VetoCode.POSITION_SIZE,), (
                f"available funds {account.available_funds:.2f} hold no contract at margin {cfg.margin_per_contract:.2f}",
            ))
        quantity = min(by_budget, cfg.max_quantity, by_leverage, by_margin)
        return RiskVerdict(
            True, (), (),
            quantity=quantity, limit_price=limit_price, stop_price=stop_price, target_price=target_price,
            risk_amount=quantity * per_contract, reward_risk=reward_risk, equity=account.equity,
            grade_applied=grade.value, risk_fraction=fraction,
        )


__all__ = ["RISK_SCHEMA_VERSION", "SESSION_TIMEZONE", "ContractSpec", "RiskConfig", "RiskGate", "ThesisConfig", "round_to_tick"]
