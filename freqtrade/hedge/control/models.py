"""Typed control-plane commands and immutable operation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any, Mapping
from uuid import UUID, uuid4

from freqtrade.hedge.symbols import raw_symbol

_FIXED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})


class ControlAction(StrEnum):
    STOP_NEW_ORDERS = "STOP_NEW_ORDERS"
    RESUME_NEW_ORDERS = "RESUME_NEW_ORDERS"
    KILL_SWITCH_ACTIVATE = "KILL_SWITCH_ACTIVATE"
    KILL_SWITCH_RELEASE = "KILL_SWITCH_RELEASE"
    CANCEL_MANAGED_ORDERS = "CANCEL_MANAGED_ORDERS"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    CLOSE_BOTH = "CLOSE_BOTH"


class ControlOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    REPLAYED = "REPLAYED"


class ControlOperationState(StrEnum):
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _required_text(value: object, *, field_name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result or len(result) > limit:
        raise ValueError(f"{field_name} is invalid")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{field_name} contains control characters")
    return result


def _optional_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = raw_symbol(value)
    if normalized not in _FIXED_SYMBOLS:
        raise ValueError("control symbol must be BTCUSDT or ETHUSDT")
    return normalized


def _quantity(value: Decimal | str | int | float | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("quantity must be numeric")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError("quantity must be finite and positive")
    return parsed


@dataclass(frozen=True, slots=True)
class ControlRequest:
    action: ControlAction
    account_id: str
    idempotency_key: str
    reason: str
    symbol: str | None = None
    quantity: Decimal | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = self.action if isinstance(self.action, ControlAction) else ControlAction(self.action)
        account_id = _required_text(self.account_id, field_name="account_id", limit=128)
        idempotency_key = _required_text(
            self.idempotency_key,
            field_name="idempotency_key",
            limit=255,
        )
        reason = _required_text(self.reason, field_name="reason", limit=1024)
        symbol = _optional_symbol(self.symbol)
        quantity = _quantity(self.quantity)
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        if action in {
            ControlAction.CLOSE_LONG,
            ControlAction.CLOSE_SHORT,
            ControlAction.CLOSE_BOTH,
        } and symbol is None:
            raise ValueError("close operations require symbol")
        if action not in {
            ControlAction.CLOSE_LONG,
            ControlAction.CLOSE_SHORT,
            ControlAction.CLOSE_BOTH,
        } and quantity is not None:
            raise ValueError("quantity is valid only for close operations")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "idempotency_key", idempotency_key)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "metadata", metadata)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "account_id": self.account_id,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
            "symbol": self.symbol,
            "quantity": None if self.quantity is None else str(self.quantity),
            "metadata": dict(self.metadata),
        }

    @property
    def request_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ControlPlanItem:
    operation: str
    symbol: str | None = None
    position_side: str | None = None
    quantity: Decimal | None = None
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "symbol": self.symbol,
            "position_side": self.position_side,
            "quantity": None if self.quantity is None else str(self.quantity),
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class ControlOperationResult:
    operation_id: UUID
    action: ControlAction
    outcome: ControlOutcome
    code: str
    actor: str
    actor_role: str
    account_id: str
    idempotency_key: str
    reason: str
    created_at: datetime
    completed_at: datetime
    symbol: str | None = None
    replayed: bool = False
    writes_attempted: int = 0
    planned: tuple[ControlPlanItem, ...] = ()
    executed_references: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, UUID):
            raise TypeError("operation_id must be UUID")
        if self.created_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("operation timestamps must be timezone-aware")
        if self.completed_at < self.created_at:
            raise ValueError("completed_at must not precede created_at")
        if self.writes_attempted < 0:
            raise ValueError("writes_attempted must be nonnegative")

    @classmethod
    def new(
        cls,
        *,
        action: ControlAction,
        outcome: ControlOutcome,
        code: str,
        actor: str,
        actor_role: str,
        request: ControlRequest,
        operation_id: UUID | None = None,
        created_at: datetime | None = None,
        replayed: bool = False,
        writes_attempted: int = 0,
        planned: tuple[ControlPlanItem, ...] = (),
        executed_references: tuple[str, ...] = (),
        errors: tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
    ) -> "ControlOperationResult":
        started = created_at or datetime.now(UTC)
        return cls(
            operation_id=operation_id or uuid4(),
            action=action,
            outcome=outcome,
            code=_required_text(code, field_name="code", limit=128),
            actor=_required_text(actor, field_name="actor", limit=128),
            actor_role=_required_text(actor_role, field_name="actor_role", limit=64),
            account_id=request.account_id,
            idempotency_key=request.idempotency_key,
            reason=request.reason,
            symbol=request.symbol,
            created_at=started,
            completed_at=datetime.now(UTC),
            replayed=replayed,
            writes_attempted=writes_attempted,
            planned=planned,
            executed_references=executed_references,
            errors=errors,
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": str(self.operation_id),
            "action": self.action.value,
            "outcome": self.outcome.value,
            "code": self.code,
            "actor": self.actor,
            "actor_role": self.actor_role,
            "account_id": self.account_id,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
            "symbol": self.symbol,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "replayed": self.replayed,
            "writes_attempted": self.writes_attempted,
            "planned": [item.to_dict() for item in self.planned],
            "executed_references": list(self.executed_references),
            "errors": list(self.errors),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlOperationResult":
        return cls(
            operation_id=UUID(str(payload["operation_id"])),
            action=ControlAction(str(payload["action"])),
            outcome=ControlOutcome(str(payload["outcome"])),
            code=str(payload["code"]),
            actor=str(payload["actor"]),
            actor_role=str(payload["actor_role"]),
            account_id=str(payload["account_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            reason=str(payload["reason"]),
            symbol=None if payload.get("symbol") is None else str(payload["symbol"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            completed_at=datetime.fromisoformat(str(payload["completed_at"])),
            replayed=bool(payload.get("replayed", False)),
            writes_attempted=int(payload.get("writes_attempted", 0)),
            planned=tuple(
                ControlPlanItem(
                    operation=str(item["operation"]),
                    symbol=item.get("symbol"),
                    position_side=item.get("position_side"),
                    quantity=(
                        None
                        if item.get("quantity") is None
                        else Decimal(str(item["quantity"]))
                    ),
                    reference=item.get("reference"),
                )
                for item in payload.get("planned", ())
            ),
            executed_references=tuple(str(item) for item in payload.get("executed_references", ())),
            errors=tuple(str(item) for item in payload.get("errors", ())),
            details=dict(payload.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class ControlPlaneStatus:
    account_id: str
    mode: str
    new_risk_enabled: bool
    kill_switch_mode: str
    kill_switch_reason: str | None
    live_exchange_write: str
    allowed_symbols: tuple[str, ...]
    confirmation_required_actions: tuple[str, ...]
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "mode": self.mode,
            "new_risk_enabled": self.new_risk_enabled,
            "kill_switch_mode": self.kill_switch_mode,
            "kill_switch_reason": self.kill_switch_reason,
            "live_exchange_write": self.live_exchange_write,
            "allowed_symbols": list(self.allowed_symbols),
            "confirmation_required_actions": list(self.confirmation_required_actions),
            "observed_at": self.observed_at.isoformat(),
        }
