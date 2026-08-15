"""Transactional application service for the hedge persistence subsystem."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from sqlalchemy.orm import Session, sessionmaker

from freqtrade.persistence.hedge_outbox import (
    OutboxEnvelope,
    PublishBatchResult,
    TransactionalOutboxPublisher,
)
from freqtrade.persistence.hedge_reconciliation import (
    HedgeReconciler,
    PositionFact,
    ReconciliationPolicy,
    ReconciliationSummary,
)
from freqtrade.persistence.hedge_recovery import LedgerRecoveryCoordinator, RecoveryReport
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository, LedgerRecovery


T = TypeVar("T")


class HedgePersistenceService:
    """Own transaction boundaries for ledger, projection, and outbox changes."""

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def transaction(self, operation: Callable[[HedgeLedgerRepository], T]) -> T:
        with self.session_factory.begin() as session:
            session.expire_on_commit = False
            result = operation(HedgeLedgerRepository(session))
            session.flush()
            return result

    def create_order_intent(self, **values: Any):
        return self.transaction(lambda repository: repository.create_order_intent(**values))

    def transition_order_intent(self, **values: Any):
        return self.transaction(
            lambda repository: repository.transition_order_intent(**values)
        )

    def record_fill(self, **values: Any):
        return self.transaction(lambda repository: repository.apply_fill(**values))

    def record_order_snapshot(self, **values: Any):
        return self.transaction(lambda repository: repository.append_order_snapshot(**values))

    def record_position_snapshot(self, **values: Any):
        return self.transaction(
            lambda repository: repository.append_position_snapshot(**values)
        )

    def record_account_event(self, **values: Any):
        return self.transaction(lambda repository: repository.record_account_event(**values))

    def record_account_risk_snapshot(self, **values: Any):
        return self.transaction(
            lambda repository: repository.append_account_risk_snapshot(**values)
        )

    def update_strategy_side_state(self, **values: Any):
        return self.transaction(
            lambda repository: repository.upsert_strategy_side_state(**values)
        )

    def rebuild_current_positions(self, **values: Any) -> LedgerRecovery:
        return self.transaction(
            lambda repository: repository.rebuild_current_projections(**values)
        )

    def record_target_position(self, **values: Any):
        return self.transaction(lambda repository: repository.record_target_position(**values))

    def update_core_position_state(self, **values: Any):
        return self.transaction(
            lambda repository: repository.upsert_core_position_state(**values)
        )

    def record_tactical_lot(self, **values: Any):
        return self.transaction(lambda repository: repository.record_tactical_lot(**values))

    def record_audit_event(self, **values: Any):
        return self.transaction(lambda repository: repository.record_audit_event(**values))

    def reconcile_positions(
        self,
        *,
        account_id: str,
        exchange: str,
        facts: list[PositionFact],
        trigger: str = "SCHEDULED",
        policy: ReconciliationPolicy | None = None,
        observed_at: datetime | None = None,
    ) -> ReconciliationSummary:
        def operation(repository: HedgeLedgerRepository) -> ReconciliationSummary:
            return HedgeReconciler(repository).reconcile_positions(
                account_id=account_id,
                exchange=exchange,
                facts=facts,
                trigger=trigger,
                policy=policy,
                observed_at=observed_at,
            )

        return self.transaction(operation)

    def recover_account(
        self,
        account_id: str,
        *,
        symbol: str | None = None,
    ) -> LedgerRecovery:
        return LedgerRecoveryCoordinator(self.session_factory).recover_account(
            account_id,
            symbol=symbol,
        )

    def recover_all(self) -> RecoveryReport:
        return LedgerRecoveryCoordinator(self.session_factory).recover_all()

    def publish_outbox(
        self,
        publish: Callable[[OutboxEnvelope], None],
        *,
        limit: int = 100,
    ) -> PublishBatchResult:
        return TransactionalOutboxPublisher(self.session_factory).publish_batch(
            publish,
            limit=limit,
        )
