from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any


class AcceptanceStatus(StrEnum):
    # Ruff S105 is a false positive here: this is an acceptance status, not a credential.
    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


class RuntimeStage(StrEnum):
    BASELINE = "BASELINE"
    REST_ONLY = "REST_ONLY"
    STREAM_STARTING = "STREAM_STARTING"
    STREAM_STALE = "STREAM_STALE"
    RECOVERING = "RECOVERING"
    READY = "READY"
    HALT = "HALT"
    STOPPED = "STOPPED"


class ReconciliationDepth(StrEnum):
    FAST = "FAST"
    DEEP = "DEEP"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True, slots=True)
class HardMetrics:
    long_short_identity_mismatch: int = 0
    duplicate_fill_effects: int = 0
    duplicate_funding_effects: int = 0
    rest_db_unexplained_position_diff: int = 0
    rest_memory_unexplained_diff: int = 0
    unknown_unrecovered_orders: int = 0
    restart_state_loss: int = 0
    ws_reconnect_without_reconciliation: int = 0
    new_risk_while_stale: int = 0
    unexplained_wallet_drift: int = 0

    def failures(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items() if int(value) != 0}

    @property
    def passed(self) -> bool:
        return not self.failures()


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    max_clock_skew_ms: float = 1_000.0
    max_clock_rtt_ms: float = 5_000.0
    quantity_tolerance: Decimal = Decimal("0.00000001")
    financial_tolerance: Decimal = Decimal("0.00000001")
    wallet_drift_tolerance: Decimal = Decimal("0.000001")
    stale_after: timedelta | None = timedelta(seconds=90)
    fast_reconciliation_interval: timedelta = timedelta(minutes=1)
    deep_reconciliation_interval: timedelta = timedelta(minutes=15)
    reconnect_requires_reconciliation: bool = True

    def __post_init__(self) -> None:
        if self.max_clock_skew_ms <= 0 or self.max_clock_rtt_ms <= 0:
            raise ValueError("clock thresholds must be positive")
        for name in ("quantity_tolerance", "financial_tolerance", "wallet_drift_tolerance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.stale_after is not None and self.stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive or None")


@dataclass(frozen=True, slots=True)
class RoundEvidence:
    round_id: str
    title: str
    status: AcceptanceStatus
    started_at: datetime
    completed_at: datetime
    checks: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status is AcceptanceStatus.PASS


@dataclass(frozen=True, slots=True)
class LegIdentity:
    account_id: str
    symbol: str
    position_side: str
    present_in_rest: bool
    quantity: Decimal
    leverage: int
    margin_mode: str

    @property
    def key(self) -> str:
        return f"{self.account_id}:{self.symbol}:{self.position_side}"


@dataclass(frozen=True, slots=True)
class PositionValue:
    account_id: str
    symbol: str
    position_side: str
    quantity: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal

    @property
    def key(self) -> str:
        return f"{self.account_id}:{self.symbol}:{self.position_side}"


@dataclass(frozen=True, slots=True)
class BalanceValue:
    account_id: str
    asset: str
    wallet_balance: Decimal
    available_balance: Decimal
    unrealized_pnl: Decimal

    @property
    def key(self) -> str:
        return f"{self.account_id}:{self.asset}"


@dataclass(frozen=True, slots=True)
class OrderValue:
    account_id: str
    symbol: str
    position_side: str
    exchange_order_id: str
    client_order_id: str
    status: str
    cumulative_filled_quantity: Decimal
    active: bool

    @property
    def key(self) -> str:
        return f"{self.account_id}:{self.symbol}:{self.exchange_order_id}"


@dataclass(frozen=True, slots=True)
class FillValue:
    account_id: str
    symbol: str
    position_side: str
    exchange_trade_id: str
    exchange_order_id: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    realized_pnl: Decimal

    @property
    def key(self) -> str:
        return f"{self.account_id}:{self.symbol}:{self.exchange_trade_id}"


@dataclass(frozen=True, slots=True)
class IncomeValue:
    account_id: str
    identity: str
    income_type: str
    asset: str
    symbol: str
    amount: Decimal
    event_time_ms: int

    @property
    def key(self) -> str:
        return f"{self.account_id}:{self.income_type}:{self.identity}"


@dataclass(frozen=True, slots=True)
class FactPlane:
    account_id: str
    observed_at: datetime
    positions: Mapping[str, PositionValue] = field(default_factory=dict)
    balances: Mapping[str, BalanceValue] = field(default_factory=dict)
    active_orders: Mapping[str, OrderValue] = field(default_factory=dict)
    fills: Mapping[str, FillValue] = field(default_factory=dict)
    income: Mapping[str, IncomeValue] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = {
            "account_id": self.account_id,
            "positions": {
                key: _json_ready(asdict(value)) for key, value in sorted(self.positions.items())
            },
            "balances": {
                key: _json_ready(asdict(value)) for key, value in sorted(self.balances.items())
            },
            "active_orders": {
                key: _json_ready(asdict(value)) for key, value in sorted(self.active_orders.items())
            },
            "fills": {key: _json_ready(asdict(value)) for key, value in sorted(self.fills.items())},
            "income": {
                key: _json_ready(asdict(value)) for key, value in sorted(self.income.items())
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotSet:
    rest: FactPlane
    memory: FactPlane
    database: FactPlane


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    depth: ReconciliationDepth
    entity_type: str
    entity_key: str
    reason: str
    expected: Any
    observed: Any
    explained: bool = False


@dataclass(frozen=True, slots=True)
class ReconciliationOutcome:
    depth: ReconciliationDepth
    checked_at: datetime
    issues: tuple[ReconciliationIssue, ...]

    @property
    def unexplained(self) -> tuple[ReconciliationIssue, ...]:
        return tuple(issue for issue in self.issues if not issue.explained)

    @property
    def passed(self) -> bool:
        return not self.unexplained


@dataclass(frozen=True, slots=True)
class RuntimeAcceptanceReport:
    schema: str
    generated_at: datetime
    baseline_version: str
    rounds: tuple[RoundEvidence, ...]
    hard_metrics: HardMetrics
    live_evidence: bool
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(round_result.passed for round_result in self.rounds) and self.hard_metrics.passed

    def to_dict(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        payload["passed"] = self.passed
        return payload

    def sha256(self) -> str:
        rendered = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return sha256(rendered.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value
