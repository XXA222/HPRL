"""Adapters from existing Hedge runtime facts into the Production Readiness Spine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from .risk_envelope import AccountRiskView


@dataclass(frozen=True, slots=True)
class RiskSnapshotAdapterPolicy:
    max_age: timedelta = timedelta(seconds=15)
    require_liquidation_complete: bool = True
    require_maintenance_complete: bool = True

    def __post_init__(self) -> None:
        if self.max_age <= timedelta(0):
            raise ValueError("risk snapshot max_age must be positive")


def risk_view_from_account_snapshot(
    snapshot: Any,
    *,
    now: datetime | None = None,
    policy: RiskSnapshotAdapterPolicy | None = None,
) -> AccountRiskView:
    """Adapt freqtrade.hedge.risk.models.AccountRiskSnapshot without importing it here."""
    required = (
        "equity", "available_balance", "gross_long_notional", "gross_short_notional",
        "initial_margin", "maintenance_margin", "pending_long_notional", "pending_short_notional",
    )
    missing = [name for name in required if not hasattr(snapshot, name)]
    if missing:
        raise TypeError("account risk snapshot missing fields: " + ",".join(missing))
    active_policy = policy or RiskSnapshotAdapterPolicy()
    if hasattr(snapshot, "effective_risk_data_valid") and not bool(snapshot.effective_risk_data_valid):
        raise ValueError("account risk snapshot is invalid")
    if active_policy.require_liquidation_complete and hasattr(snapshot, "liquidation_data_complete"):
        if not bool(snapshot.liquidation_data_complete):
            raise ValueError("liquidation data is incomplete")
    if active_policy.require_maintenance_complete and hasattr(snapshot, "maintenance_margin_complete"):
        if not bool(snapshot.maintenance_margin_complete):
            raise ValueError("maintenance margin data is incomplete")
    if now is not None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        observed_ms = getattr(snapshot, "exchange_time_ms", None) or getattr(snapshot, "observed_at_ms", None)
        if observed_ms is None:
            raise ValueError("risk snapshot timestamp is required for production freshness check")
        observed = datetime.fromtimestamp(int(observed_ms) / 1000, tz=UTC)
        age = now.astimezone(UTC) - observed
        if age < timedelta(0):
            raise ValueError("risk snapshot timestamp is in the future")
        if age > active_policy.max_age:
            raise ValueError("risk snapshot is stale")
    return AccountRiskView(
        equity=Decimal(getattr(snapshot, "equity")),
        available_balance=Decimal(getattr(snapshot, "available_balance")),
        long_notional=Decimal(getattr(snapshot, "gross_long_notional")),
        short_notional=Decimal(getattr(snapshot, "gross_short_notional")),
        initial_margin=Decimal(getattr(snapshot, "initial_margin")),
        maintenance_margin=Decimal(getattr(snapshot, "maintenance_margin")),
        pending_long_notional=Decimal(getattr(snapshot, "pending_long_notional")),
        pending_short_notional=Decimal(getattr(snapshot, "pending_short_notional")),
    )


def execution_gate_is_fail_closed(gate: Any) -> bool:
    snapshot = gate.snapshot()
    return bool(getattr(snapshot, "blocking_reasons", ())) or not bool(getattr(snapshot, "armed", False))


def readiness_allows_new_risk(readiness: Any) -> bool:
    method = getattr(readiness, "allows_new_risk", None)
    if not callable(method):
        raise TypeError("readiness gate must expose allows_new_risk")
    return bool(method())
