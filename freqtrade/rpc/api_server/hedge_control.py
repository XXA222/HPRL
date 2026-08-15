"""Authenticated R3.3 Hedge control-plane REST routes."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from freqtrade.hedge.control.models import ControlAction, ControlRequest
from freqtrade.hedge.control.service import (
    ControlConfirmationError,
    ControlPermissionError,
    HedgeControlService,
)
from freqtrade.hedge.control.store import ControlOperationConflict
from freqtrade.rpc.api_server.hedge_auth import HedgePrincipal


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ControlRequestSchema(_Schema):
    action: ControlAction
    account_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1, max_length=1024)
    symbol: str | None = Field(default=None, min_length=1, max_length=64)
    quantity: Decimal | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confirmation_token: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str | None) -> str | None:
        return None if value is None else value.upper()

    def domain(self) -> ControlRequest:
        return ControlRequest(
            action=self.action,
            account_id=self.account_id,
            idempotency_key=self.idempotency_key,
            reason=self.reason,
            symbol=self.symbol,
            quantity=self.quantity,
            metadata=self.metadata,
        )


class ConfirmationRequestSchema(_Schema):
    action: ControlAction
    account_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1, max_length=1024)
    symbol: str | None = Field(default=None, min_length=1, max_length=64)
    quantity: Decimal | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def domain(self) -> ControlRequest:
        return ControlRequest(
            action=self.action,
            account_id=self.account_id,
            idempotency_key=self.idempotency_key,
            reason=self.reason,
            symbol=self.symbol,
            quantity=self.quantity,
            metadata=self.metadata,
        )


class ConfirmationResponseSchema(_Schema):
    token: str
    action: ControlAction
    account_id: str
    idempotency_key: str
    request_hash: str
    one_time: bool = True


class ControlStatusResponseSchema(_Schema):
    account_id: str
    mode: str
    new_risk_enabled: bool
    kill_switch_mode: str
    kill_switch_reason: str | None
    live_exchange_write: str
    allowed_symbols: list[str]
    confirmation_required_actions: list[str]
    observed_at: str


class ControlOperationResponseSchema(_Schema):
    operation_id: str
    action: str
    outcome: str
    code: str
    actor: str
    actor_role: str
    account_id: str
    idempotency_key: str
    reason: str
    symbol: str | None
    created_at: str
    completed_at: str
    replayed: bool
    writes_attempted: int
    planned: list[dict[str, Any]]
    executed_references: list[str]
    errors: list[str]
    details: dict[str, Any]


def create_hedge_control_router(
    service: HedgeControlService,
    *,
    principal_dependency: Callable[..., HedgePrincipal],
) -> APIRouter:
    if not isinstance(service, HedgeControlService):
        raise TypeError("service must be HedgeControlService")
    router = APIRouter(prefix="/hedge/control", tags=["hedge-control"])

    @router.get("/status", response_model=ControlStatusResponseSchema)
    def status(
        _: HedgePrincipal = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        return service.status().to_dict()

    @router.post("/confirmations", response_model=ConfirmationResponseSchema)
    def confirmation(
        body: ConfirmationRequestSchema,
        principal: HedgePrincipal = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        request = body.domain()
        try:
            token = service.issue_confirmation(principal=principal, request=request)
        except ControlPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ControlConfirmationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "token": token,
            "action": request.action.value,
            "account_id": request.account_id,
            "idempotency_key": request.idempotency_key,
            "request_hash": request.request_hash,
            "one_time": True,
        }

    @router.post("/operations", response_model=ControlOperationResponseSchema)
    def operation(
        body: ControlRequestSchema,
        principal: HedgePrincipal = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        request = body.domain()
        try:
            return service.execute(
                principal=principal,
                request=request,
                confirmation_token=body.confirmation_token,
            ).to_dict()
        except ControlPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ControlConfirmationError as exc:
            raise HTTPException(status_code=412, detail=str(exc)) from exc
        except ControlOperationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
