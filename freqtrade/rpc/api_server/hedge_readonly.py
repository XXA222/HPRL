"""Read-only hedge REST router and QQ/WeChat read-command dispatcher."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Callable, Protocol, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from .hedge_auth import HedgePrincipal
from .hedge_schemas import (
    ActionGroupStatusResponse,
    DisabledWriteErrorResponse,
    DisabledWriteResponse,
    DualLegPositionsResponse,
    ExecutionOrderListResponse,
    ExecutionOrderSchema,
    HedgeEventListResponse,
    HedgeEventRecordSchema,
    HedgeReadonlyCommandResponse,
    HedgeReadonlyCommandSchema,
    PairSummaryResponse,
    OperationAuditSchema,
    ReadinessStatusSchema,
    ReconciliationStatusSchema,
    RiskStatusSchema,
    UserStreamStatusSchema,
)

from freqtrade.hedge.execution.ledger import InMemoryExecutionLedger
from freqtrade.hedge.execution.service import ExecutionOrder, ExecutionService
from freqtrade.hedge.execution.state_machine import OrderState

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")


class HedgeReadonlyQueryPort(Protocol):
    def positions(
        self,
        *,
        account_id: str,
        symbol: str,
        source: str | None = None,
    ) -> DualLegPositionsResponse: ...

    def risk(self, *, account_id: str, source: str | None = None) -> RiskStatusSchema: ...

    def reconciliation(
        self,
        *,
        account_id: str,
        source: str | None = None,
    ) -> ReconciliationStatusSchema: ...

    def readiness(
        self, *, account_id: str, source: str | None = None
    ) -> ReadinessStatusSchema: ...

    def user_stream(
        self, *, account_id: str, source: str | None = None
    ) -> UserStreamStatusSchema: ...

    def audit(
        self,
        *,
        account_id: str,
        limit: int,
    ) -> Sequence[OperationAuditSchema]: ...


class HedgeExecutionReadonlyQueryPort(Protocol):
    def orders(
        self,
        *,
        account_id: str,
        symbol: str | None,
        status: str | None,
        limit: int,
    ) -> ExecutionOrderListResponse: ...

    def order(self, *, client_order_id: str) -> ExecutionOrderSchema: ...

    def action_group(
        self,
        *,
        action_group_id: UUID,
    ) -> ActionGroupStatusResponse: ...

    def pair_summary(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> PairSummaryResponse: ...

    def events(
        self,
        *,
        account_id: str,
        limit: int,
    ) -> HedgeEventListResponse: ...


def _execution_order_schema(order: ExecutionOrder) -> ExecutionOrderSchema:
    return ExecutionOrderSchema(
        client_order_id=order.client_order_id,
        intent_id=str(order.intent.intent_id),
        action_group_id=(
            str(order.intent.action_group_id)
            if order.intent.action_group_id is not None
            else None
        ),
        account_id=order.intent.account_id,
        symbol=order.intent.symbol,
        position_side=order.intent.position_side.value,
        action=order.intent.action.value,
        order_type=order.intent.order_type.value,
        status=order.lifecycle.status.value,
        requested_quantity=order.intent.quantity,
        approved_quantity=order.approved_quantity,
        filled_quantity=order.lifecycle.filled_quantity,
        remaining_quantity=(
            order.approved_quantity - order.lifecycle.filled_quantity
        ),
        limit_price=order.intent.limit_price,
        average_price=order.lifecycle.average_price,
        reduce_only=order.intent.reduce_only,
        exchange_order_id=order.lifecycle.exchange_order_id,
        reason=order.lifecycle.reason,
        created_at=order.created_at,
        updated_at=order.lifecycle.updated_at,
    )


class ExecutionReadonlyQueryAdapter:
    """Concrete read model backed by the direction-five execution service."""

    def __init__(self, service: ExecutionService) -> None:
        if not isinstance(service, ExecutionService):
            raise TypeError("service must be an ExecutionService")
        self._service = service

    def orders(
        self,
        *,
        account_id: str,
        symbol: str | None,
        status: str | None,
        limit: int,
    ) -> ExecutionOrderListResponse:
        statuses = None
        if status is not None:
            try:
                statuses = (OrderState(status),)
            except ValueError as exc:
                raise ValueError("status is invalid") from exc
        orders = self._service.list_orders(
            account_id=account_id,
            symbol=symbol,
            statuses=statuses,
            limit=limit,
        )
        models = tuple(_execution_order_schema(order) for order in orders)
        return ExecutionOrderListResponse(
            orders=models,
            count=len(models),
            as_of=datetime.now(UTC),
        )

    def order(self, *, client_order_id: str) -> ExecutionOrderSchema:
        return _execution_order_schema(self._service.get_order(client_order_id))

    def action_group(
        self,
        *,
        action_group_id: UUID,
    ) -> ActionGroupStatusResponse:
        orders = self._service.action_group_orders(action_group_id)
        models = tuple(_execution_order_schema(order) for order in orders)
        if not orders:
            status = "EMPTY"
        else:
            states = {order.lifecycle.status for order in orders}
            successes = {
                OrderState.ACKNOWLEDGED,
                OrderState.PARTIAL,
                OrderState.FILLED,
            }
            failures = {OrderState.CANCELED, OrderState.REJECTED}
            if OrderState.UNKNOWN in states:
                status = "UNKNOWN"
            elif all(order.lifecycle.status is OrderState.FILLED for order in orders):
                status = "COMPLETED"
            elif states <= failures:
                status = "FAILED"
            elif states & failures and states & successes:
                status = "PARTIAL_FAILURE"
            else:
                status = "IN_PROGRESS"
        filled = sum(
            (order.lifecycle.filled_quantity for order in orders),
            Decimal("0"),
        )
        return ActionGroupStatusResponse(
            action_group_id=str(action_group_id),
            status=status,
            orders=models,
            filled_quantity=filled,
            as_of=datetime.now(UTC),
        )


class ExecutionLedgerReadonlyAdapter(ExecutionReadonlyQueryAdapter):
    """Extended order/pair/event read model backed by the transactional fake ledger."""

    def __init__(self, service: ExecutionService, ledger: InMemoryExecutionLedger) -> None:
        super().__init__(service)
        if not isinstance(ledger, InMemoryExecutionLedger):
            raise TypeError("ledger must be an InMemoryExecutionLedger")
        self._ledger = ledger

    def pair_summary(self, *, account_id: str, symbol: str) -> PairSummaryResponse:
        positions = self._ledger.positions(account_id=account_id, symbol=symbol)
        by_side = {item.position_key.position_side.value: item for item in positions}
        long = by_side.get("LONG")
        short = by_side.get("SHORT")
        long_qty = Decimal("0") if long is None else long.quantity
        short_qty = Decimal("0") if short is None else short.quantity
        pending_entry = Decimal("0")
        pending_reduce = Decimal("0")
        for order in self._service.list_orders(
            account_id=account_id, symbol=symbol, include_terminal=False
        ):
            remaining = order.approved_quantity - order.lifecycle.filled_quantity
            if order.intent.reduces_risk:
                pending_reduce += remaining
            else:
                pending_entry += remaining
        return PairSummaryResponse(
            account_id=account_id,
            symbol=symbol,
            long_quantity=long_qty,
            short_quantity=short_qty,
            net_quantity=long_qty - short_qty,
            gross_quantity=long_qty + short_qty,
            long_average_price=Decimal("0") if long is None else long.average_entry_price,
            short_average_price=Decimal("0") if short is None else short.average_entry_price,
            realized_pnl=sum((item.realized_pnl for item in positions), Decimal("0")),
            fees=sum((item.fees for item in positions), Decimal("0")),
            funding=sum((item.funding for item in positions), Decimal("0")),
            pending_entry_quantity=pending_entry,
            pending_reduce_quantity=pending_reduce,
            as_of=datetime.now(UTC),
        )

    def events(self, *, account_id: str, limit: int) -> HedgeEventListResponse:
        records = tuple(
            HedgeEventRecordSchema(
                event_type=item.event_type,
                occurred_at=item.occurred_at,
                payload=dict(item.payload),
            )
            for item in self._ledger.audit(limit=limit)
            if str(item.payload.get("account_id", account_id)) == account_id
        )
        return HedgeEventListResponse(
            events=records, count=len(records), as_of=datetime.now(UTC)
        )


def _required_text(
    value: object,
    *,
    field_name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f"{field_name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > max_length
        or _CONTROL.search(normalized)
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be 1..{max_length} valid characters",
        )
    return normalized


def _required_symbol(value: object) -> str:
    raw = _required_text(value, field_name="symbol", max_length=64).upper()
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise HTTPException(status_code=422, detail="symbol contains invalid characters")
    parts = raw.split(":")
    if len(parts) > 2:
        raise HTTPException(status_code=422, detail="symbol settlement suffix is invalid")
    normalized = re.sub(r"[/_-]", "", parts[0])
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise HTTPException(status_code=422, detail="symbol is invalid")
    if len(parts) == 2 and not normalized.endswith(parts[1]):
        raise HTTPException(status_code=422, detail="symbol settlement suffix mismatches")
    return normalized


def _source_kwargs(source: str | None) -> dict[str, str]:
    """Preserve compatibility with pre-source query adapters and test doubles."""

    return {} if source is None else {"source": source}


def create_hedge_readonly_router(
    query: HedgeReadonlyQueryPort,
    *,
    principal_dependency: Callable[..., HedgePrincipal],
    execution_query: HedgeExecutionReadonlyQueryPort | None = None,
) -> APIRouter:
    if not callable(principal_dependency):
        raise TypeError("principal_dependency must be callable")
    required = (
        "positions",
        "risk",
        "reconciliation",
        "readiness",
        "user_stream",
        "audit",
    )
    if any(not callable(getattr(query, name, None)) for name in required):
        raise TypeError("query does not implement HedgeReadonlyQueryPort")
    router = APIRouter(prefix="/hedge", tags=["hedge-readonly"])

    def authenticated(
        principal: HedgePrincipal = Depends(principal_dependency),
    ) -> HedgePrincipal:
        if not isinstance(principal, HedgePrincipal):
            raise HTTPException(status_code=401, detail="invalid hedge principal")
        return principal

    @router.get("/positions/{symbol}", response_model=DualLegPositionsResponse)
    def positions(
        symbol: str,
        account_id: str = "default",
        source: str | None = Query(default=None, pattern="^(EXCHANGE|PAPER|LIVE|SHADOW)$"),
        _: HedgePrincipal = Depends(authenticated),
    ) -> DualLegPositionsResponse:
        return query.positions(
            account_id=_required_text(account_id, field_name="account_id"),
            symbol=_required_symbol(symbol),
            **_source_kwargs(source),
        )

    @router.get("/risk", response_model=RiskStatusSchema)
    def risk(
        account_id: str = "default",
        source: str | None = Query(default=None, pattern="^(EXCHANGE|PAPER|LIVE|SHADOW)$"),
        _: HedgePrincipal = Depends(authenticated),
    ) -> RiskStatusSchema:
        return query.risk(
            account_id=_required_text(account_id, field_name="account_id"),
            **_source_kwargs(source),
        )

    @router.get("/reconciliation", response_model=ReconciliationStatusSchema)
    def reconciliation(
        account_id: str = "default",
        source: str | None = Query(default=None, pattern="^(EXCHANGE|PAPER|LIVE|SHADOW)$"),
        _: HedgePrincipal = Depends(authenticated),
    ) -> ReconciliationStatusSchema:
        return query.reconciliation(
            account_id=_required_text(account_id, field_name="account_id"),
            **_source_kwargs(source),
        )

    @router.get("/readiness", response_model=ReadinessStatusSchema)
    def readiness(
        account_id: str = "default",
        source: str | None = Query(default=None, pattern="^(EXCHANGE|PAPER|LIVE|SHADOW)$"),
        _: HedgePrincipal = Depends(authenticated),
    ) -> ReadinessStatusSchema:
        return query.readiness(
            account_id=_required_text(account_id, field_name="account_id"),
            **_source_kwargs(source),
        )

    @router.get("/user-stream", response_model=UserStreamStatusSchema)
    def user_stream(
        account_id: str = "default",
        source: str | None = Query(default=None, pattern="^(EXCHANGE|PAPER|LIVE|SHADOW)$"),
        _: HedgePrincipal = Depends(authenticated),
    ) -> UserStreamStatusSchema:
        return query.user_stream(
            account_id=_required_text(account_id, field_name="account_id"),
            **_source_kwargs(source),
        )

    @router.get("/audit", response_model=list[OperationAuditSchema])
    def audit(
        account_id: str = "default",
        limit: int = Query(default=100, ge=1, le=500),
        _: HedgePrincipal = Depends(authenticated),
    ) -> Sequence[OperationAuditSchema]:
        return query.audit(
            account_id=_required_text(account_id, field_name="account_id"),
            limit=limit,
        )

    if execution_query is not None:
        execution_required = ("orders", "order", "action_group")
        if any(
            not callable(getattr(execution_query, name, None))
            for name in execution_required
        ):
            raise TypeError(
                "execution_query does not implement HedgeExecutionReadonlyQueryPort"
            )

        @router.get("/orders", response_model=ExecutionOrderListResponse)
        def execution_orders(
            account_id: str = "default",
            symbol: str | None = None,
            status: str | None = None,
            limit: int = Query(default=100, ge=1, le=1000),
            _: HedgePrincipal = Depends(authenticated),
        ) -> ExecutionOrderListResponse:
            normalized_status = status.strip().upper() if status else None
            try:
                return execution_query.orders(
                    account_id=_required_text(
                        account_id,
                        field_name="account_id",
                    ),
                    symbol=None if symbol is None else _required_symbol(symbol),
                    status=normalized_status,
                    limit=limit,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        @router.get(
            "/orders/{client_order_id}",
            response_model=ExecutionOrderSchema,
        )
        def execution_order(
            client_order_id: str,
            _: HedgePrincipal = Depends(authenticated),
        ) -> ExecutionOrderSchema:
            try:
                return execution_query.order(
                    client_order_id=_required_text(
                        client_order_id,
                        field_name="client_order_id",
                        max_length=256,
                    )
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="order not found") from exc

        @router.get(
            "/action-groups/{action_group_id}",
            response_model=ActionGroupStatusResponse,
        )
        def action_group(
            action_group_id: UUID,
            _: HedgePrincipal = Depends(authenticated),
        ) -> ActionGroupStatusResponse:
            return execution_query.action_group(
                action_group_id=action_group_id
            )


        if callable(getattr(execution_query, "pair_summary", None)):
            @router.get("/pair-summary/{symbol}", response_model=PairSummaryResponse)
            def pair_summary(
                symbol: str,
                account_id: str = "default",
                _: HedgePrincipal = Depends(authenticated),
            ) -> PairSummaryResponse:
                return execution_query.pair_summary(
                    account_id=_required_text(account_id, field_name="account_id"),
                    symbol=_required_symbol(symbol),
                )

        if callable(getattr(execution_query, "events", None)):
            @router.get("/events", response_model=HedgeEventListResponse)
            def events(
                account_id: str = "default",
                limit: int = Query(default=100, ge=1, le=500),
                _: HedgePrincipal = Depends(authenticated),
            ) -> HedgeEventListResponse:
                return execution_query.events(
                    account_id=_required_text(account_id, field_name="account_id"),
                    limit=limit,
                )

    return router


class HedgeReadonlyCommandDispatcher:
    """Reusable, explicitly enumerated QQ/WeChat read-only command contract."""

    def __init__(
        self,
        query: HedgeReadonlyQueryPort,
        *,
        execution_query: HedgeExecutionReadonlyQueryPort | None = None,
    ) -> None:
        required = (
            "positions",
            "risk",
            "reconciliation",
            "readiness",
            "user_stream",
        )
        if any(not callable(getattr(query, name, None)) for name in required):
            raise TypeError("query does not implement the read command contract")
        self._query = query
        if execution_query is not None:
            execution_required = ("orders", "order", "action_group")
            if any(
                not callable(getattr(execution_query, name, None))
                for name in execution_required
            ):
                raise TypeError("execution_query does not implement the read contract")
        self._execution_query = execution_query

    def dispatch(
        self,
        command: HedgeReadonlyCommandSchema,
    ) -> HedgeReadonlyCommandResponse:
        if not isinstance(command, HedgeReadonlyCommandSchema):
            raise TypeError("command must be a HedgeReadonlyCommandSchema")
        handlers = {
            "hedge.positions": self._positions,
            "hedge.risk": self._risk,
            "hedge.reconciliation": self._reconciliation,
            "hedge.readiness": self._readiness,
            "hedge.user_stream": self._user_stream,
        }
        if self._execution_query is not None:
            handlers.update(
                {
                    "hedge.orders": self._orders,
                    "hedge.order": self._order,
                    "hedge.action_group": self._action_group,
                }
            )
            if callable(getattr(self._execution_query, "pair_summary", None)):
                handlers["hedge.pair_summary"] = self._pair_summary
            if callable(getattr(self._execution_query, "events", None)):
                handlers["hedge.events"] = self._events
        handler = handlers.get(command.command)
        if handler is None:
            return HedgeReadonlyCommandResponse(
                request_id=command.request_id,
                ok=False,
                command=command.command,
                message="execution read model is not configured",
                data={},
            )
        try:
            data = handler(command)
        except (KeyError, ValueError) as exc:
            return HedgeReadonlyCommandResponse(
                request_id=command.request_id,
                ok=False,
                command=command.command,
                message=str(exc) or type(exc).__name__,
                data={},
            )
        return HedgeReadonlyCommandResponse(
            request_id=command.request_id,
            ok=True,
            command=command.command,
            message="OK",
            data=data,
        )

    def _positions(self, command: HedgeReadonlyCommandSchema) -> dict:
        if command.symbol is None:  # already enforced by schema
            raise ValueError("hedge.positions requires symbol")
        model = self._query.positions(
            account_id=command.account_id,
            symbol=command.symbol,
        )
        return model.model_dump(mode="json")

    def _risk(self, command: HedgeReadonlyCommandSchema) -> dict:
        return self._query.risk(
            account_id=command.account_id
        ).model_dump(mode="json")

    def _reconciliation(self, command: HedgeReadonlyCommandSchema) -> dict:
        return self._query.reconciliation(
            account_id=command.account_id
        ).model_dump(mode="json")

    def _readiness(self, command: HedgeReadonlyCommandSchema) -> dict:
        return self._query.readiness(
            account_id=command.account_id
        ).model_dump(mode="json")

    def _user_stream(self, command: HedgeReadonlyCommandSchema) -> dict:
        return self._query.user_stream(
            account_id=command.account_id
        ).model_dump(mode="json")

    def _orders(self, command: HedgeReadonlyCommandSchema) -> dict:
        if self._execution_query is None:  # pragma: no cover - guarded by dispatch
            raise RuntimeError("execution query is unavailable")
        model = self._execution_query.orders(
            account_id=command.account_id,
            symbol=command.symbol,
            status=command.status.upper() if command.status else None,
            limit=command.limit,
        )
        return model.model_dump(mode="json")

    def _order(self, command: HedgeReadonlyCommandSchema) -> dict:
        if self._execution_query is None:  # pragma: no cover - guarded by dispatch
            raise RuntimeError("execution query is unavailable")
        if command.client_order_id is None:  # guarded by schema
            raise ValueError("hedge.order requires client_order_id")
        return self._execution_query.order(
            client_order_id=command.client_order_id
        ).model_dump(mode="json")

    def _pair_summary(self, command: HedgeReadonlyCommandSchema) -> dict:
        if self._execution_query is None or command.symbol is None:
            raise ValueError("hedge.pair_summary requires symbol")
        return self._execution_query.pair_summary(
            account_id=command.account_id, symbol=command.symbol
        ).model_dump(mode="json")

    def _events(self, command: HedgeReadonlyCommandSchema) -> dict:
        if self._execution_query is None:
            raise RuntimeError("execution query is unavailable")
        return self._execution_query.events(
            account_id=command.account_id, limit=command.limit
        ).model_dump(mode="json")

    def _action_group(self, command: HedgeReadonlyCommandSchema) -> dict:
        if self._execution_query is None:  # pragma: no cover - guarded by dispatch
            raise RuntimeError("execution query is unavailable")
        if command.action_group_id is None:  # guarded by schema
            raise ValueError("hedge.action_group requires action_group_id")
        try:
            group_id = UUID(command.action_group_id)
        except ValueError as exc:
            raise ValueError("action_group_id must be a UUID") from exc
        return self._execution_query.action_group(
            action_group_id=group_id
        ).model_dump(mode="json")


def create_disabled_hedge_write_router() -> APIRouter:
    """Optional compatibility router. Every write remains hard-disabled."""
    router = APIRouter(prefix="/hedge", tags=["hedge-write-disabled"])

    def disabled() -> JSONResponse:
        payload = DisabledWriteResponse().model_dump(mode="json")
        return JSONResponse(status_code=503, content={"detail": payload})

    router.add_api_route(
        "/orders",
        disabled,
        methods=["POST"],
        response_model=DisabledWriteErrorResponse,
        status_code=503,
    )
    router.add_api_route(
        "/kill-switch",
        disabled,
        methods=["POST"],
        response_model=DisabledWriteErrorResponse,
        status_code=503,
    )
    return router
