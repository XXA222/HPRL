"""Port for consuming authoritative account facts from exchange/ledger direction two."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from freqtrade.hedge.risk.models import PendingOrderRisk
from freqtrade.hedge.risk.portfolio import (
    PositionRiskLeg,
    RiskPortfolioSnapshot,
    build_risk_portfolio,
)


@dataclass(frozen=True, slots=True)
class AccountRiskFacts:
    exchange: str
    account_id: str
    equity: Decimal
    wallet_balance: Decimal
    available_balance: Decimal
    positions: tuple[PositionRiskLeg, ...]
    pending_orders: tuple[PendingOrderRisk, ...]
    initial_margin: Decimal
    maintenance_margin: Decimal
    snapshot_id: str
    source_version: int
    exchange_time_ms: int
    observed_at_ms: int
    reconciliation_converged: bool
    source_complete: bool = True
    source_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_complete, bool):
            raise ValueError("source_complete must be a boolean.")
        if not isinstance(self.reconciliation_converged, bool):
            raise ValueError("reconciliation_converged must be a boolean.")
        if not isinstance(self.source_errors, tuple):
            raise ValueError("source_errors must be a tuple.")
        normalized = tuple(
            item.strip()
            for item in self.source_errors
            if isinstance(item, str) and item.strip()
        )
        if len(normalized) != len(self.source_errors):
            raise ValueError("source_errors must contain non-empty strings only.")
        object.__setattr__(self, "source_errors", tuple(dict.fromkeys(normalized)))

    def to_portfolio(self) -> RiskPortfolioSnapshot:
        portfolio = build_risk_portfolio(
            exchange=self.exchange,
            account_id=self.account_id,
            equity=self.equity,
            wallet_balance=self.wallet_balance,
            available_balance=self.available_balance,
            positions=self.positions,
            pending_orders=self.pending_orders,
            initial_margin=self.initial_margin,
            maintenance_margin=self.maintenance_margin,
            risk_data_valid=(
                self.source_complete
                and self.reconciliation_converged
                and not self.source_errors
            ),
            snapshot_id=self.snapshot_id,
            source_version=self.source_version,
            exchange_time_ms=self.exchange_time_ms,
            observed_at_ms=self.observed_at_ms,
            strict_completeness=True,
        )
        if self.source_errors:
            from dataclasses import replace

            account = replace(
                portfolio.account,
                risk_data_valid=False,
                risk_data_errors=tuple(
                    dict.fromkeys((*portfolio.account.risk_data_errors, *self.source_errors))
                ),
            )
            return RiskPortfolioSnapshot(account, portfolio.positions, portfolio.pending_orders)
        return portfolio


class AccountRiskFactsPort(Protocol):
    """Read the latest reconciled full-account facts without exchange writes."""

    def read_account_risk_facts(self, *, account_id: str) -> AccountRiskFacts: ...
