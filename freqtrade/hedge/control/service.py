"""Production control-plane orchestration with RBAC, confirmation and audit."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from typing import Callable, Mapping

from freqtrade.hedge.control.models import (
    ControlAction,
    ControlOperationResult,
    ControlOutcome,
    ControlPlanItem,
    ControlPlaneStatus,
    ControlRequest,
)
from freqtrade.hedge.control.store import (
    ControlOperationConflict,
    ControlOperationStore,
)
from freqtrade.hedge.execution.action_group import (
    ActionGroupExecutor,
    build_close_both_plan,
)
from freqtrade.hedge.execution.service import (
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.integration.production_main_loop import (
    HedgeExecutionMode,
    ProductionEquivalentHedgeMainLoop,
)
from freqtrade.hedge.control.auth import (
    ConfirmationService,
    HedgePrincipal,
    HedgeRole,
)

_ROLE_REQUIREMENTS = {
    ControlAction.STOP_NEW_ORDERS: HedgeRole.OPERATOR,
    ControlAction.RESUME_NEW_ORDERS: HedgeRole.RISK_MANAGER,
    ControlAction.KILL_SWITCH_ACTIVATE: HedgeRole.OPERATOR,
    ControlAction.KILL_SWITCH_RELEASE: HedgeRole.ADMIN,
    ControlAction.CANCEL_MANAGED_ORDERS: HedgeRole.RISK_MANAGER,
    ControlAction.CLOSE_LONG: HedgeRole.RISK_MANAGER,
    ControlAction.CLOSE_SHORT: HedgeRole.RISK_MANAGER,
    ControlAction.CLOSE_BOTH: HedgeRole.RISK_MANAGER,
}
_CONFIRMATION_REQUIRED = frozenset(
    {
        ControlAction.RESUME_NEW_ORDERS,
        ControlAction.KILL_SWITCH_RELEASE,
        ControlAction.CANCEL_MANAGED_ORDERS,
        ControlAction.CLOSE_LONG,
        ControlAction.CLOSE_SHORT,
        ControlAction.CLOSE_BOTH,
    }
)


class ControlPermissionError(PermissionError):
    pass


class ControlConfirmationError(PermissionError):
    pass


class HedgeControlService:
    def __init__(
        self,
        *,
        loop: ProductionEquivalentHedgeMainLoop,
        operation_store: ControlOperationStore,
        confirmation_service: ConfirmationService,
        account_view_provider: Callable[[], object],
        audit_recorder: Callable[..., object] | None = None,
        exchange_write_surface: str = "NONE",
        action_group_executor: ActionGroupExecutor | None = None,
    ) -> None:
        if not isinstance(loop, ProductionEquivalentHedgeMainLoop):
            raise TypeError("loop must be ProductionEquivalentHedgeMainLoop")
        if not callable(account_view_provider):
            raise TypeError("account_view_provider must be callable")
        self.loop = loop
        self.operation_store = operation_store
        self.confirmations = confirmation_service
        self.account_view_provider = account_view_provider
        self.audit_recorder = audit_recorder
        self.exchange_write_surface = str(exchange_write_surface).strip().upper() or "NONE"
        self.action_group_executor = action_group_executor or ActionGroupExecutor(loop.engine)
        self._lock = RLock()

    @property
    def account_id(self) -> str:
        return self.loop.account_id

    @property
    def confirmation_required_actions(self) -> tuple[ControlAction, ...]:
        return tuple(sorted(_CONFIRMATION_REQUIRED, key=lambda item: item.value))

    def status(self) -> ControlPlaneStatus:
        snapshot = self.loop.kill_switch.snapshot()
        if self.loop.mode is HedgeExecutionMode.HEDGE_SIMULATED:
            live_exchange_write = "SIMULATED"
        elif self._live_locked():
            live_exchange_write = "LOCKED"
        else:
            live_exchange_write = "ARMED"
        return ControlPlaneStatus(
            account_id=self.loop.account_id,
            mode=self.loop.mode.value,
            new_risk_enabled=self.loop.new_risk_enabled,
            kill_switch_mode=snapshot.mode.value,
            kill_switch_reason=snapshot.reason,
            live_exchange_write=live_exchange_write,
            allowed_symbols=self.loop.allowed_symbols,
            confirmation_required_actions=tuple(
                item.value for item in self.confirmation_required_actions
            ),
        )

    def restore_state(self) -> None:
        """Restore durable halt/new-risk state without replaying exchange actions."""
        actions = tuple(
            item.value
            for item in (
                ControlAction.STOP_NEW_ORDERS,
                ControlAction.RESUME_NEW_ORDERS,
                ControlAction.KILL_SWITCH_ACTIVATE,
                ControlAction.KILL_SWITCH_RELEASE,
            )
        )
        history = self.operation_store.latest_results(
            account_id=self.loop.account_id,
            actions=actions,
        )
        for result in history:
            if result.outcome not in {ControlOutcome.SUCCEEDED, ControlOutcome.REPLAYED}:
                continue
            if result.action is ControlAction.STOP_NEW_ORDERS:
                self.loop.stop_new_orders(reason=result.reason, actor=result.actor)
            elif result.action is ControlAction.RESUME_NEW_ORDERS:
                self.loop.resume_new_orders(actor=result.actor, confirmed=True)
            elif result.action is ControlAction.KILL_SWITCH_ACTIVATE:
                self.loop.kill_switch.activate(reason=result.reason, actor=result.actor)
            elif result.action is ControlAction.KILL_SWITCH_RELEASE:
                self.loop.kill_switch.deactivate(actor=result.actor, confirmed=True)

    def issue_confirmation(
        self,
        *,
        principal: HedgePrincipal,
        request: ControlRequest,
    ) -> str:
        self._authorize(principal, request.action)
        if request.action not in _CONFIRMATION_REQUIRED:
            raise ControlConfirmationError("action does not require confirmation")
        self._validate_scope(request)
        token = self.confirmations.issue(
            subject=principal.subject,
            action=request.action.value,
            account_id=request.account_id,
            symbol=request.symbol,
            payload_hash=request.request_hash,
            idempotency_key=request.idempotency_key,
        )
        self._audit(
            event_type="CONTROL_CONFIRMATION_ISSUED",
            principal=principal,
            request=request,
            outcome="ISSUED",
            payload={"request_hash": request.request_hash},
        )
        return token

    def execute(
        self,
        *,
        principal: HedgePrincipal,
        request: ControlRequest,
        confirmation_token: str | None = None,
    ) -> ControlOperationResult:
        self._authorize(principal, request.action)
        self._validate_scope(request)
        existing = self.operation_store.lookup(request=request)
        if existing is not None:
            if existing.existing_result is not None:
                return self._replay(existing.existing_result)
            if existing.in_progress:
                raise RuntimeError("control operation is already in progress")
        if request.action in _CONFIRMATION_REQUIRED:
            if confirmation_token is None or not self.confirmations.consume(
                token=confirmation_token,
                subject=principal.subject,
                action=request.action.value,
                account_id=request.account_id,
                symbol=request.symbol,
                payload_hash=request.request_hash,
                idempotency_key=request.idempotency_key,
            ):
                raise ControlConfirmationError("valid one-time confirmation is required")

        with self._lock:
            claim = self.operation_store.claim(
                request=request,
                actor=principal.subject,
                actor_role=principal.role.name,
            )
            if claim.existing_result is not None:
                return self._replay(claim.existing_result)
            if claim.in_progress:
                raise RuntimeError("control operation is already in progress")
            if claim.recovered_stale:
                self._audit(
                    event_type="CONTROL_OPERATION_STALE_LEASE_RECOVERED",
                    principal=principal,
                    request=request,
                    outcome="RECOVERING",
                    payload={
                        "operation_id": str(claim.operation_id),
                        "recovery_policy": "SAME_ACTOR_SAME_IDEMPOTENCY_KEY",
                    },
                )
            try:
                result = self._dispatch(
                    principal=principal,
                    request=request,
                    operation_id=claim.operation_id,
                    created_at=claim.created_at,
                )
            except Exception as exc:
                result = ControlOperationResult.new(
                    action=request.action,
                    outcome=ControlOutcome.FAILED,
                    code=type(exc).__name__.upper(),
                    actor=principal.subject,
                    actor_role=principal.role.name,
                    request=request,
                    operation_id=claim.operation_id,
                    created_at=claim.created_at,
                    errors=(str(exc)[:1000],),
                )
                self.operation_store.fail(result)
                self._audit(
                    event_type="CONTROL_OPERATION_FAILED",
                    principal=principal,
                    request=request,
                    outcome=result.code,
                    payload=result.to_dict(),
                )
                raise
            self.operation_store.complete(result)
            self._audit(
                event_type="CONTROL_OPERATION_COMPLETED",
                principal=principal,
                request=request,
                outcome=result.code,
                payload=result.to_dict(),
            )
            return result

    @staticmethod
    def _replay(previous: ControlOperationResult) -> ControlOperationResult:
        return ControlOperationResult(
            operation_id=previous.operation_id,
            action=previous.action,
            outcome=ControlOutcome.REPLAYED,
            code=previous.code,
            actor=previous.actor,
            actor_role=previous.actor_role,
            account_id=previous.account_id,
            idempotency_key=previous.idempotency_key,
            reason=previous.reason,
            symbol=previous.symbol,
            created_at=previous.created_at,
            completed_at=previous.completed_at,
            replayed=True,
            writes_attempted=0,
            planned=previous.planned,
            executed_references=previous.executed_references,
            errors=previous.errors,
            details={**dict(previous.details), "original_outcome": previous.outcome.value},
        )

    def _dispatch(
        self,
        *,
        principal: HedgePrincipal,
        request: ControlRequest,
        operation_id,
        created_at: datetime,
    ) -> ControlOperationResult:
        action = request.action
        if action is ControlAction.STOP_NEW_ORDERS:
            self.loop.stop_new_orders(reason=request.reason, actor=principal.subject)
            return self._success(principal, request, operation_id, created_at, "NEW_RISK_STOPPED")
        if action is ControlAction.RESUME_NEW_ORDERS:
            self.loop.resume_new_orders(actor=principal.subject, confirmed=True)
            return self._success(principal, request, operation_id, created_at, "NEW_RISK_RESUMED")
        if action is ControlAction.KILL_SWITCH_ACTIVATE:
            plan = self._managed_cancel_plan(None)
            if self._live_locked():
                snapshot = self.loop.kill_switch.activate(
                    reason=request.reason,
                    actor=principal.subject,
                )
                self.loop.stop_new_orders(
                    reason=request.reason,
                    actor=principal.subject,
                )
                return self._success(
                    principal,
                    request,
                    operation_id,
                    created_at,
                    "KILL_SWITCH_HALTED",
                    writes_attempted=0,
                    planned=plan,
                    details={
                        "kill_switch_mode": snapshot.mode.value,
                        "managed_cancellations_deferred": len(plan),
                        "live_write_locked": True,
                    },
                )
            report = self.loop.emergency_stop(reason=request.reason, actor=principal.subject)
            return self._success(
                principal,
                request,
                operation_id,
                created_at,
                "KILL_SWITCH_HALTED",
                writes_attempted=len(report.canceled),
                planned=plan,
                executed=tuple(item.order.client_order_id for item in report.canceled),
                errors=tuple(item.message for item in report.errors),
                details={"kill_switch_mode": report.kill_switch.mode.value},
            )
        if action is ControlAction.KILL_SWITCH_RELEASE:
            snapshot = self.loop.resume_after_emergency(actor=principal.subject, confirmed=True)
            return self._success(
                principal,
                request,
                operation_id,
                created_at,
                "KILL_SWITCH_RUNNING",
                details={"kill_switch_mode": snapshot.mode.value},
            )
        if action is ControlAction.CANCEL_MANAGED_ORDERS:
            plan = self._managed_cancel_plan(request.symbol)
            if self._live_locked():
                return self._blocked(
                    principal,
                    request,
                    operation_id,
                    created_at,
                    "LIVE_WRITE_LOCKED",
                    planned=plan,
                )
            report = self.loop.cancel_managed_orders(symbol=request.symbol)
            return self._success(
                principal,
                request,
                operation_id,
                created_at,
                "MANAGED_ORDERS_CANCELED",
                writes_attempted=len(report.canceled),
                planned=plan,
                executed=tuple(item.order.client_order_id for item in report.canceled),
                errors=tuple(item.message for item in report.errors),
            )
        if action in {
            ControlAction.CLOSE_LONG,
            ControlAction.CLOSE_SHORT,
            ControlAction.CLOSE_BOTH,
        }:
            return self._close_positions(
                principal=principal,
                request=request,
                operation_id=operation_id,
                created_at=created_at,
            )
        raise ValueError(f"unsupported control action: {action.value}")

    def _close_positions(
        self,
        *,
        principal: HedgePrincipal,
        request: ControlRequest,
        operation_id,
        created_at: datetime,
    ) -> ControlOperationResult:
        view = self.account_view_provider()
        if str(getattr(view, "account_id", "")) != request.account_id:
            raise RuntimeError("readonly account does not match control account")
        sides = {
            ControlAction.CLOSE_LONG: ("LONG",),
            ControlAction.CLOSE_SHORT: ("SHORT",),
            ControlAction.CLOSE_BOTH: ("LONG", "SHORT"),
        }[request.action]
        positions = {
            str(item.position_side).upper(): item
            for item in getattr(view, "positions", ())
            if str(item.symbol).upper() == request.symbol
            and str(item.position_side).upper() in sides
            and Decimal(item.quantity) > 0
        }
        plan: list[ControlPlanItem] = []
        quantities: dict[str, Decimal] = {"LONG": Decimal("0"), "SHORT": Decimal("0")}
        marks: dict[str, Decimal] = {}
        for side in sides:
            item = positions.get(side)
            if item is None:
                continue
            available = Decimal(item.quantity)
            quantity = available if request.quantity is None else min(available, request.quantity)
            if quantity <= 0:
                continue
            quantities[side] = quantity
            marks[side] = Decimal(item.mark_price)
            plan.append(
                ControlPlanItem(
                    operation="CLOSE_POSITION",
                    symbol=request.symbol,
                    position_side=side,
                    quantity=quantity,
                )
            )
        if not plan:
            return self._success(
                principal,
                request,
                operation_id,
                created_at,
                "ALREADY_FLAT",
                planned=(),
            )
        if self._live_locked():
            return self._blocked(
                principal,
                request,
                operation_id,
                created_at,
                "LIVE_WRITE_LOCKED",
                planned=tuple(plan),
                details={"risk_reducing": True},
            )

        if request.action is ControlAction.CLOSE_BOTH:
            group, intents = build_close_both_plan(
                account_id=request.account_id,
                symbol=str(request.symbol),
                long_quantity=quantities["LONG"],
                short_quantity=quantities["SHORT"],
                idempotency_key=f"control:{request.idempotency_key}",
                order_type=OrderType.MARKET,
                action_group_id=operation_id,
            )
            enriched = tuple(
                OrderIntent(
                    account_id=intent.account_id,
                    symbol=intent.symbol,
                    position_side=intent.position_side,
                    action=intent.action,
                    quantity=intent.quantity,
                    idempotency_key=intent.idempotency_key,
                    order_type=intent.order_type,
                    limit_price=intent.limit_price,
                    reduce_only=intent.reduce_only,
                    action_group_id=intent.action_group_id,
                    metadata={
                        **dict(intent.metadata),
                        "reference_price": str(marks[intent.position_side.value]),
                        "control_operation_id": str(operation_id),
                    },
                )
                for intent in intents
            )
            report = self.action_group_executor.execute_plan(group, enriched)
            return self._success(
                principal,
                request,
                operation_id,
                created_at,
                "CLOSE_BOTH_ACCEPTED" if report.fully_successful else report.outcome,
                writes_attempted=report.attempted,
                planned=tuple(plan),
                executed=tuple(item.order.client_order_id for item in report.results),
                errors=report.errors,
                details={"action_group_id": str(group.action_group_id)},
            )

        side_name = sides[0]
        side = PositionSide(side_name)
        intent = OrderIntent(
            account_id=request.account_id,
            symbol=str(request.symbol),
            position_side=side,
            action=IntentAction.CLOSE,
            quantity=quantities[side_name],
            idempotency_key=f"control:{request.idempotency_key}:{side_name}",
            order_type=OrderType.MARKET,
            reduce_only=True,
            metadata={
                "reference_price": str(marks[side_name]),
                "control_operation_id": str(operation_id),
                "control_action": request.action.value,
            },
        )
        execution = self.loop.engine.submit(intent)
        return self._success(
            principal,
            request,
            operation_id,
            created_at,
            f"{side_name}_CLOSE_ACCEPTED",
            writes_attempted=1,
            planned=tuple(plan),
            executed=(execution.order.client_order_id,),
        )

    def _managed_cancel_plan(self, symbol: str | None) -> tuple[ControlPlanItem, ...]:
        rows = self.loop.ownership.managed_open_orders(account_id=self.loop.account_id)
        result = []
        for order in rows:
            if symbol is not None and order.intent.symbol != symbol:
                continue
            result.append(
                ControlPlanItem(
                    operation="CANCEL_MANAGED_ORDER",
                    symbol=order.intent.symbol,
                    position_side=order.intent.position_side.value,
                    reference=order.client_order_id,
                )
            )
        return tuple(result)

    def _live_locked(self) -> bool:
        return (
            self.loop.mode is HedgeExecutionMode.HEDGE_PRODUCTION_LOCKED
            or self.exchange_write_surface == "NONE"
        )

    def _validate_scope(self, request: ControlRequest) -> None:
        if request.account_id != self.loop.account_id:
            raise ValueError("control request account_id does not match main loop")
        if request.symbol is not None and request.symbol not in self.loop.allowed_symbols:
            raise ValueError("control request symbol is not allowlisted")

    @staticmethod
    def _authorize(principal: HedgePrincipal, action: ControlAction) -> None:
        required = _ROLE_REQUIREMENTS[action]
        if principal.role < required:
            raise ControlPermissionError(
                f"{action.value} requires {required.name} role"
            )

    def _success(
        self,
        principal: HedgePrincipal,
        request: ControlRequest,
        operation_id,
        created_at: datetime,
        code: str,
        *,
        writes_attempted: int = 0,
        planned: tuple[ControlPlanItem, ...] = (),
        executed: tuple[str, ...] = (),
        errors: tuple[str, ...] = (),
        details: Mapping[str, object] | None = None,
    ) -> ControlOperationResult:
        outcome = ControlOutcome.SUCCEEDED if not errors else ControlOutcome.FAILED
        return ControlOperationResult.new(
            action=request.action,
            outcome=outcome,
            code=code,
            actor=principal.subject,
            actor_role=principal.role.name,
            request=request,
            operation_id=operation_id,
            created_at=created_at,
            writes_attempted=writes_attempted,
            planned=planned,
            executed_references=executed,
            errors=errors,
            details=details,
        )

    def _blocked(
        self,
        principal: HedgePrincipal,
        request: ControlRequest,
        operation_id,
        created_at: datetime,
        code: str,
        *,
        planned: tuple[ControlPlanItem, ...],
        details: Mapping[str, object] | None = None,
    ) -> ControlOperationResult:
        return ControlOperationResult.new(
            action=request.action,
            outcome=ControlOutcome.BLOCKED,
            code=code,
            actor=principal.subject,
            actor_role=principal.role.name,
            request=request,
            operation_id=operation_id,
            created_at=created_at,
            writes_attempted=0,
            planned=planned,
            details=details,
        )

    def _audit(
        self,
        *,
        event_type: str,
        principal: HedgePrincipal,
        request: ControlRequest,
        outcome: str,
        payload: Mapping[str, object],
    ) -> None:
        if self.audit_recorder is None:
            return
        try:
            self.audit_recorder(
                account_id=request.account_id,
                exchange="binance",
                event_type=event_type,
                entity_type="ControlOperation",
                entity_id=request.idempotency_key,
                severity="INFO" if "FAILED" not in event_type else "ERROR",
                reason_code=outcome[:64],
                correlation_id=request.idempotency_key,
                actor=principal.subject,
                payload=dict(payload),
            )
        except Exception:
            # Audit storage failure must not be hidden by the API layer. The operation
            # result is already durable in ControlOperationStore and can be reconciled.
            raise RuntimeError("CONTROL_AUDIT_STORE_UNAVAILABLE")


__all__ = [
    "ControlConfirmationError",
    "ControlOperationConflict",
    "ControlPermissionError",
    "HedgeControlService",
]
