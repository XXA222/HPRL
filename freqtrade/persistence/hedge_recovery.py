"""Startup recovery coordinator for orders and position projections."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from freqtrade.persistence.hedge_models import FillEvent, OrderSnapshot, PositionSnapshot
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository, LedgerRecovery


@dataclass(frozen=True)
class AccountRecoveryResult:
    account_id: str
    order_count: int
    position_count: int


@dataclass(frozen=True)
class RecoveryReport:
    accounts: tuple[AccountRecoveryResult, ...]

    @property
    def order_count(self) -> int:
        return sum(item.order_count for item in self.accounts)

    @property
    def position_count(self) -> int:
        return sum(item.position_count for item in self.accounts)


class LedgerRecoveryCoordinator:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def discover_accounts(self) -> tuple[str, ...]:
        accounts: set[str] = set()
        with self.session_factory() as session:
            for model in (FillEvent, OrderSnapshot, PositionSnapshot):
                accounts.update(session.scalars(select(model.account_id).distinct()).all())
        return tuple(sorted(accounts))

    def recover_account(self, account_id: str, *, symbol: str | None = None) -> LedgerRecovery:
        with self.session_factory.begin() as session:
            return HedgeLedgerRepository(session).rebuild_current_position_snapshots(
                account_id=account_id,
                symbol=symbol,
            )

    def recover_all(self) -> RecoveryReport:
        results: list[AccountRecoveryResult] = []
        for account_id in self.discover_accounts():
            recovery = self.recover_account(account_id)
            results.append(
                AccountRecoveryResult(
                    account_id=account_id,
                    order_count=len(recovery.orders),
                    position_count=len(recovery.positions),
                )
            )
        return RecoveryReport(accounts=tuple(results))
