"""Direction-three risk adapters for direction-five execution ports."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from freqtrade.enums.hedge import PositionAction as RiskAction
from freqtrade.enums.hedge import PositionSide as RiskSide
from freqtrade.hedge.execution.service import OrderIntent, RiskApproval
from freqtrade.hedge.risk.commit import (
    InMemoryRiskApprovalCommitStore,
    RiskApprovalCommitPort,
)
from freqtrade.hedge.risk.engine import HedgeRiskEngine
from freqtrade.hedge.risk.portfolio import RiskPortfolioSnapshot
from freqtrade.hedge.risk.runtime import HedgeRiskRuntime, OrderRiskIntent


class PortfolioRiskApprovalAdapter:
    """Legacy direct-engine adapter retained for isolated unit tests."""

    def __init__(
        self,
        *,
        engine: HedgeRiskEngine,
        portfolio_provider: Callable[[], RiskPortfolioSnapshot],
    ) -> None:
        self._engine = engine
        self._portfolio_provider = portfolio_provider

    def approve(self, intent: OrderIntent) -> RiskApproval:
        reference = intent.limit_price
        if reference is None:
            raw = intent.metadata.get("reference_price")
            reference = None if raw is None else Decimal(str(raw))
        if reference is None or reference <= 0:
            return RiskApproval(False, Decimal("0"), ("REFERENCE_PRICE_REQUIRED",))
        portfolio = self._portfolio_provider()
        decision = self._engine.evaluate_portfolio_order(
            portfolio=portfolio,
            symbol=intent.symbol,
            position_side=RiskSide(intent.position_side.value),
            action=RiskAction(intent.action.value),
            requested_quantity=intent.quantity,
            reference_price=reference,
        )
        return RiskApproval(
            decision.allowed,
            decision.approved_quantity,
            decision.reason_codes,
        )


class RuntimeRiskApprovalAdapter:
    """Use the unified Readiness/lease/lock/risk runtime for every execution intent.

    Direction five currently exposes a small synchronous approval port.  This adapter
    conservatively transfers a successful direction-three reservation to a verified
    commit store before returning the approval.  The composition root supplies the
    same runtime/commit store for the lifetime of the Paper application.
    """

    def __init__(
        self,
        *,
        runtime: HedgeRiskRuntime,
        portfolio_provider: Callable[[], RiskPortfolioSnapshot],
        leverage: Decimal,
        maintenance_margin_rate: Decimal,
        commit_store: RiskApprovalCommitPort | None = None,
    ) -> None:
        if leverage < 1 or not leverage.is_finite():
            raise ValueError("leverage must be a finite Decimal greater than or equal to 1")
        if (
            not maintenance_margin_rate.is_finite()
            or maintenance_margin_rate <= 0
            or maintenance_margin_rate > 1
        ):
            raise ValueError("maintenance_margin_rate must be in (0, 1]")
        self._runtime = runtime
        self._portfolio_provider = portfolio_provider
        self._leverage = leverage
        self._maintenance_margin_rate = maintenance_margin_rate
        self._commit_store = commit_store or InMemoryRiskApprovalCommitStore()

    @staticmethod
    def _reference_price(intent: OrderIntent) -> Decimal | None:
        if intent.limit_price is not None:
            return intent.limit_price
        raw = intent.metadata.get("reference_price")
        return None if raw is None else Decimal(str(raw))

    @staticmethod
    def _expires_at_ms(intent: OrderIntent) -> int | None:
        value = intent.metadata.get("expires_at")
        if value is None:
            return None
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            return -1
        return int(value.timestamp() * 1000)

    def approve(self, intent: OrderIntent) -> RiskApproval:
        reference = self._reference_price(intent)
        if reference is None or not reference.is_finite() or reference <= 0:
            return RiskApproval(False, Decimal("0"), ("REFERENCE_PRICE_REQUIRED",))
        expires_at_ms = self._expires_at_ms(intent)
        if expires_at_ms == -1:
            return RiskApproval(False, Decimal("0"), ("INTENT_EXPIRY_INVALID",))
        portfolio = self._portfolio_provider()
        decision = self._runtime.approve_order(
            portfolio=portfolio,
            intent=OrderRiskIntent(
                symbol=intent.symbol,
                position_side=RiskSide(intent.position_side.value),
                action=RiskAction(intent.action.value),
                requested_quantity=intent.quantity,
                reference_price=reference,
                leverage=self._leverage,
                intent_id=str(intent.intent_id),
                idempotency_key=intent.idempotency_key,
                correlation_id=str(intent.action_group_id or intent.intent_id),
                expires_at_ms=expires_at_ms,
                maintenance_margin_rate=self._maintenance_margin_rate,
            ),
        )
        if not decision.allowed:
            return RiskApproval(False, Decimal("0"), decision.reason_codes)
        reservation = decision.reservation
        if reservation is None:
            return RiskApproval(False, Decimal("0"), ("RISK_RESERVATION_MISSING",))
        try:
            reservation.confirm(
                commit_port=self._commit_store,
                durable_reference=f"paper-execution:{intent.idempotency_key}",
            )
        except Exception:
            reservation.release()
            return RiskApproval(False, Decimal("0"), ("RISK_DURABLE_HANDOFF_FAILED",))
        return RiskApproval(True, decision.approved_quantity, decision.reason_codes)
