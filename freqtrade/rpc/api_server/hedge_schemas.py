"""Strict Pydantic contracts for the hedge read-only control plane."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")
AccountId = Annotated[str, Field(min_length=1, max_length=128)]
ReadonlyCommand = Literal[
    "hedge.positions",
    "hedge.risk",
    "hedge.reconciliation",
    "hedge.readiness",
    "hedge.user_stream",
    "hedge.orders",
    "hedge.order",
    "hedge.action_group",
    "hedge.pair_summary",
    "hedge.events",
]


def _validate_text(value: object, *, field_name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    result = value.strip()
    if not result or len(result) > limit or _CONTROL.search(result):
        raise ValueError(f"{field_name} is invalid")
    return result


def _validate_json_value(
    value: object,
    *,
    field_name: str,
    depth: int = 0,
    budget: list[int] | None = None,
) -> object:
    """Validate and normalize an untrusted JSON-compatible value."""
    if budget is None:
        budget = [10_000]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"{field_name} contains too many values")
    if depth > 12:
        raise ValueError(f"{field_name} nesting is too deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value) > 65_536 or _CONTROL.search(value):
            raise ValueError(f"{field_name} contains an invalid string")
        return value
    if isinstance(value, dict):
        if len(value) > 1_000:
            raise ValueError(f"{field_name} mapping is too large")
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _validate_text(raw_key, field_name=f"{field_name} key", limit=256)
            if key in normalized:
                raise ValueError(f"{field_name} keys collide after normalization")
            normalized[key] = _validate_json_value(
                raw_value,
                field_name=field_name,
                depth=depth + 1,
                budget=budget,
            )
        return normalized
    if isinstance(value, (list, tuple)):
        if len(value) > 1_000:
            raise ValueError(f"{field_name} sequence is too large")
        return [
            _validate_json_value(
                item,
                field_name=field_name,
                depth=depth + 1,
                budget=budget,
            )
            for item in value
        ]
    raise ValueError(f"{field_name} contains unsupported {type(value).__name__}")


def _validate_json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    result = _validate_json_value(value, field_name=field_name)
    if not isinstance(result, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{field_name} must be a mapping")
    return result


def _symbol(value: object) -> str:
    raw = _validate_text(value, field_name="symbol", limit=64).upper()
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    parts = raw.split(":")
    if len(parts) > 2:
        raise ValueError("symbol contains multiple settlement suffixes")
    normalized = re.sub(r"[/_-]", "", parts[0])
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    if len(parts) == 2 and not normalized.endswith(parts[1]):
        raise ValueError("settlement suffix must match quote asset")
    return normalized


class HedgeSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ProjectionMetadataSchema(HedgeSchema):
    source: Literal["EXCHANGE", "PAPER", "LIVE", "SHADOW"] = "EXCHANGE"
    source_version: str = Field(default="0", min_length=1, max_length=128)
    sequence: int = Field(default=0, ge=0)
    event_time: AwareDatetime | None = None
    observed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    stale: bool = True
    operation_mode: Literal["readonly", "paper", "shadow", "live"] = "readonly"


class LegPositionSchema(HedgeSchema):
    account_id: AccountId
    symbol: str = Field(min_length=1, max_length=64)
    position_side: Literal["LONG", "SHORT"]
    quantity: Decimal = Field(ge=0)
    entry_price: Decimal = Field(ge=0)
    mark_price: Decimal | None = Field(default=None, ge=0)
    unrealized_pnl: Decimal | None = None
    leverage: Decimal | None = Field(default=None, gt=0)

    @field_validator("account_id")
    @classmethod
    def _account(cls, value: str) -> str:
        return _validate_text(value, field_name="account_id", limit=128)

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: object) -> str:
        return _symbol(value)

    @model_validator(mode="after")
    def _price_invariants(self) -> "LegPositionSchema":
        if self.quantity == 0 and self.entry_price != 0:
            raise ValueError("flat position must have entry_price == 0")
        if self.quantity > 0 and self.entry_price <= 0:
            raise ValueError("open position requires positive entry_price")
        return self

    @field_serializer(
        "quantity",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
        "leverage",
    )
    def _serialize_decimal(self, value: Decimal | None) -> str | None:
        return None if value is None else str(value)


class DualLegPositionsResponse(ProjectionMetadataSchema):
    account_id: AccountId
    symbol: str = Field(min_length=1, max_length=64)
    legs: tuple[LegPositionSchema, ...]
    as_of: AwareDatetime

    @field_validator("account_id")
    @classmethod
    def _account(cls, value: str) -> str:
        return _validate_text(value, field_name="account_id", limit=128)

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: object) -> str:
        return _symbol(value)

    @model_validator(mode="after")
    def _validate_legs(self) -> "DualLegPositionsResponse":
        if len(self.legs) > 2:
            raise ValueError("at most LONG and SHORT legs are allowed")
        sides = [leg.position_side for leg in self.legs]
        if len(sides) != len(set(sides)):
            raise ValueError("position legs must not duplicate a side")
        for leg in self.legs:
            if leg.account_id != self.account_id or leg.symbol != self.symbol:
                raise ValueError("position leg scope must match response scope")
        return self


class RiskStatusSchema(ProjectionMetadataSchema):
    account_id: AccountId
    equity: Decimal
    gross_notional: Decimal = Field(ge=0)
    gross_exposure_ratio: Decimal = Field(ge=0)
    margin_utilization: Decimal = Field(ge=0)
    liquidation_buffer_ratio: Decimal
    halted: bool = False
    reasons: tuple[str, ...] = ()

    @field_validator("account_id")
    @classmethod
    def _account(cls, value: str) -> str:
        return _validate_text(value, field_name="account_id", limit=128)

    @field_validator("reasons")
    @classmethod
    def _reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 32:
            raise ValueError("too many risk reasons")
        return tuple(
            _validate_text(item, field_name="reason", limit=256)
            for item in value
        )

    @field_serializer(
        "equity",
        "gross_notional",
        "gross_exposure_ratio",
        "margin_utilization",
        "liquidation_buffer_ratio",
    )
    def _serialize_decimal(self, value: Decimal) -> str:
        return str(value)


class ReconciliationStatusSchema(ProjectionMetadataSchema):
    status: Literal["HEALTHY", "DRIFT", "RUNNING", "UNKNOWN", "NOT_APPLICABLE"]
    last_run_at: AwareDatetime | None = None
    drift_count: int = Field(default=0, ge=0)
    unresolved_count: int = Field(default=0, ge=0)
    details: tuple[str, ...] = ()

    @field_validator("details")
    @classmethod
    def _details(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 100:
            raise ValueError("too many reconciliation details")
        return tuple(
            _validate_text(item, field_name="detail", limit=1024)
            for item in value
        )


class UserStreamStatusSchema(ProjectionMetadataSchema):
    state: Literal[
        "CONNECTED",
        "STALE",
        "DISCONNECTED",
        "RECONNECTING",
        "UNKNOWN",
        "NOT_APPLICABLE",
    ]
    last_event_at: AwareDatetime | None = None
    age_ms: int | None = Field(default=None, ge=0)
    reconnect_count: int = Field(default=0, ge=0)


class ReadinessStatusSchema(ProjectionMetadataSchema):
    ready: bool
    read_only: bool = True
    live_trading_enabled: bool = False
    kill_switch: Literal["RUNNING", "HALTED"]
    unknown_leg_locks: tuple[str, ...] = ()
    checks: dict[str, bool] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @field_validator("unknown_leg_locks")
    @classmethod
    def _locks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 1_000:
            raise ValueError("too many unknown leg locks")
        normalized = tuple(
            _validate_text(item, field_name="unknown leg lock", limit=256)
            for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("unknown leg locks must be unique")
        return normalized

    @field_validator("checks", mode="before")
    @classmethod
    def _checks(cls, value: object) -> dict[str, bool]:
        if not isinstance(value, dict):
            raise ValueError("checks must be a mapping")
        if len(value) > 256:
            raise ValueError("too many readiness checks")
        normalized: dict[str, bool] = {}
        for raw_key, raw_value in value.items():
            key = _validate_text(raw_key, field_name="check name", limit=128)
            if key in normalized:
                raise ValueError("readiness check names collide after normalization")
            if not isinstance(raw_value, bool):
                raise ValueError("readiness check values must be booleans")
            normalized[key] = raw_value
        return normalized

    @field_validator("reasons")
    @classmethod
    def _reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 100:
            raise ValueError("too many readiness reasons")
        return tuple(
            _validate_text(item, field_name="readiness reason", limit=512)
            for item in value
        )

    @model_validator(mode="after")
    def _safe_defaults(self) -> "ReadinessStatusSchema":
        if not self.read_only or self.live_trading_enabled:
            raise ValueError("direction-five readiness must remain read-only")
        return self


class OperationAuditSchema(HedgeSchema):
    audit_id: str = Field(min_length=1, max_length=128)
    actor: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=128)
    outcome: str = Field(min_length=1, max_length=128)
    occurred_at: AwareDatetime
    correlation_id: str | None = Field(default=None, max_length=128)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("audit_id", "actor", "action", "outcome")
    @classmethod
    def _required_fields(cls, value: str) -> str:
        return _validate_text(value, field_name="audit field", limit=128)

    @field_validator("correlation_id")
    @classmethod
    def _correlation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text(value, field_name="correlation_id", limit=128)

    @field_validator("details", mode="before")
    @classmethod
    def _details(cls, value: object) -> dict[str, object]:
        return _validate_json_mapping(value, field_name="audit details")


class ExecutionOrderSchema(HedgeSchema):
    client_order_id: str = Field(min_length=1, max_length=256)
    intent_id: str = Field(min_length=1, max_length=64)
    action_group_id: str | None = Field(default=None, max_length=64)
    account_id: AccountId
    symbol: str = Field(min_length=1, max_length=64)
    position_side: Literal["LONG", "SHORT"]
    action: Literal["OPEN", "INCREASE", "REDUCE", "CLOSE"]
    order_type: Literal["MARKET", "LIMIT"]
    status: Literal[
        "PREPARED",
        "SUBMITTING",
        "ACKNOWLEDGED",
        "PARTIAL",
        "FILLED",
        "CANCELED",
        "REJECTED",
        "UNKNOWN",
    ]
    requested_quantity: Decimal = Field(gt=0)
    approved_quantity: Decimal = Field(ge=0)
    filled_quantity: Decimal = Field(ge=0)
    remaining_quantity: Decimal = Field(ge=0)
    limit_price: Decimal | None = Field(default=None, gt=0)
    average_price: Decimal | None = Field(default=None, gt=0)
    reduce_only: bool
    exchange_order_id: str | None = Field(default=None, max_length=256)
    reason: str | None = Field(default=None, max_length=1024)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @field_validator("client_order_id", "intent_id", "account_id")
    @classmethod
    def _order_text(cls, value: str) -> str:
        return _validate_text(value, field_name="execution order field", limit=256)

    @field_validator("action_group_id", "exchange_order_id", "reason")
    @classmethod
    def _optional_order_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text(value, field_name="execution order field", limit=1024)

    @field_validator("symbol", mode="before")
    @classmethod
    def _order_symbol(cls, value: object) -> str:
        return _symbol(value)

    @model_validator(mode="after")
    def _quantities(self) -> "ExecutionOrderSchema":
        if self.approved_quantity > self.requested_quantity:
            raise ValueError("approved quantity exceeds requested quantity")
        if self.filled_quantity > self.approved_quantity:
            raise ValueError("filled quantity exceeds approved quantity")
        if self.remaining_quantity != self.approved_quantity - self.filled_quantity:
            raise ValueError("remaining quantity is inconsistent")
        return self

    @field_serializer(
        "requested_quantity",
        "approved_quantity",
        "filled_quantity",
        "remaining_quantity",
        "limit_price",
        "average_price",
    )
    def _serialize_order_decimal(self, value: Decimal | None) -> str | None:
        return None if value is None else str(value)


class ExecutionOrderListResponse(HedgeSchema):
    orders: tuple[ExecutionOrderSchema, ...]
    count: int = Field(ge=0)
    as_of: AwareDatetime

    @model_validator(mode="after")
    def _count_matches(self) -> "ExecutionOrderListResponse":
        if self.count != len(self.orders):
            raise ValueError("count does not match orders")
        return self


class ActionGroupStatusResponse(HedgeSchema):
    action_group_id: str = Field(min_length=1, max_length=64)
    status: Literal[
        "EMPTY",
        "IN_PROGRESS",
        "UNKNOWN",
        "COMPLETED",
        "PARTIAL_FAILURE",
        "FAILED",
    ]
    orders: tuple[ExecutionOrderSchema, ...]
    filled_quantity: Decimal = Field(ge=0)
    as_of: AwareDatetime

    @field_validator("action_group_id")
    @classmethod
    def _group_id(cls, value: str) -> str:
        return _validate_text(value, field_name="action_group_id", limit=64)

    @field_serializer("filled_quantity")
    def _serialize_group_decimal(self, value: Decimal) -> str:
        return str(value)


class PairSummaryResponse(HedgeSchema):
    account_id: AccountId
    symbol: str = Field(min_length=1, max_length=64)
    long_quantity: Decimal = Field(ge=0)
    short_quantity: Decimal = Field(ge=0)
    net_quantity: Decimal
    gross_quantity: Decimal = Field(ge=0)
    long_average_price: Decimal = Field(ge=0)
    short_average_price: Decimal = Field(ge=0)
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Field(default=Decimal("0"), ge=0)
    funding: Decimal = Decimal("0")
    pending_entry_quantity: Decimal = Field(default=Decimal("0"), ge=0)
    pending_reduce_quantity: Decimal = Field(default=Decimal("0"), ge=0)
    as_of: AwareDatetime

    @field_validator("symbol", mode="before")
    @classmethod
    def _summary_symbol(cls, value: object) -> str:
        return _symbol(value)

    @field_serializer(
        "long_quantity", "short_quantity", "net_quantity", "gross_quantity",
        "long_average_price", "short_average_price", "realized_pnl", "fees",
        "funding", "pending_entry_quantity", "pending_reduce_quantity",
    )
    def _summary_decimal(self, value: Decimal) -> str:
        return str(value)


class HedgeEventRecordSchema(HedgeSchema):
    event_type: str = Field(min_length=1, max_length=128)
    occurred_at: AwareDatetime
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def _event_type_text(cls, value: str) -> str:
        return _validate_text(value, field_name="event_type", limit=128)

    @field_validator("payload", mode="before")
    @classmethod
    def _event_payload(cls, value: object) -> dict[str, object]:
        return _validate_json_mapping(value, field_name="event payload")


class HedgeEventListResponse(HedgeSchema):
    events: tuple[HedgeEventRecordSchema, ...]
    count: int = Field(ge=0)
    as_of: AwareDatetime

    @model_validator(mode="after")
    def _event_count(self) -> "HedgeEventListResponse":
        if self.count != len(self.events):
            raise ValueError("count does not match events")
        return self


class HedgeWsEventSchema(HedgeSchema):
    schema_version: int = Field(default=1, ge=1, le=2_147_483_647)
    payload_version: int = Field(default=1, ge=1, le=2_147_483_647)
    event_id: str = Field(min_length=1, max_length=128)
    event_type: Literal[
        "INTENT",
        "ORDER",
        "FILL",
        "DRIFT",
        "HALT",
        "RECONNECT",
        "LOCK",
        "RECONCILIATION",
        "READINESS",
        "USER_STREAM",
        "AUDIT",
    ]
    occurred_at: AwareDatetime
    exchange_time: AwareDatetime | None = None
    observed_time: AwareDatetime | None = None
    account_id: AccountId
    symbol: str | None = Field(default=None, max_length=64)
    correlation_id: str | None = Field(default=None, max_length=128)
    payload: dict[str, JsonValue]

    @field_validator("event_id", "account_id")
    @classmethod
    def _required_fields(cls, value: str) -> str:
        return _validate_text(value, field_name="event field", limit=128)

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_optional_symbol(cls, value: object) -> str | None:
        return None if value is None else _symbol(value)

    @field_validator("correlation_id")
    @classmethod
    def _correlation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text(value, field_name="correlation_id", limit=128)

    @field_validator("payload", mode="before")
    @classmethod
    def _payload(cls, value: object) -> dict[str, object]:
        return _validate_json_mapping(value, field_name="event payload")


class DisabledWriteResponse(HedgeSchema):
    code: Literal["HEDGE_WRITE_DISABLED"] = "HEDGE_WRITE_DISABLED"
    message: str = "Hedge write operations are not enabled"


class DisabledWriteErrorResponse(HedgeSchema):
    detail: DisabledWriteResponse = Field(default_factory=DisabledWriteResponse)


class HedgeReadonlyCommandSchema(HedgeSchema):
    source: Literal["QQ", "WECHAT"]
    command: ReadonlyCommand
    request_id: str = Field(min_length=1, max_length=128)
    account_id: AccountId = "default"
    symbol: str | None = Field(default=None, max_length=64)
    client_order_id: str | None = Field(default=None, max_length=256)
    action_group_id: str | None = Field(default=None, max_length=64)
    status: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=50, ge=1, le=200)

    @field_validator("request_id", "account_id")
    @classmethod
    def _required_fields(cls, value: str) -> str:
        return _validate_text(value, field_name="command field", limit=128)

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: object) -> str | None:
        return None if value is None else _symbol(value)

    @field_validator("client_order_id", "action_group_id", "status")
    @classmethod
    def _optional_command_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text(value, field_name="command field", limit=256)

    @model_validator(mode="after")
    def _command_requirements(self) -> "HedgeReadonlyCommandSchema":
        if self.command in {"hedge.positions", "hedge.pair_summary"} and not self.symbol:
            raise ValueError(f"{self.command} requires symbol")
        if self.command == "hedge.order" and not self.client_order_id:
            raise ValueError("hedge.order requires client_order_id")
        if self.command == "hedge.action_group" and not self.action_group_id:
            raise ValueError("hedge.action_group requires action_group_id")
        return self


class HedgeReadonlyCommandResponse(HedgeSchema):
    request_id: str = Field(min_length=1, max_length=128)
    ok: bool
    command: ReadonlyCommand
    message: str = Field(min_length=1, max_length=1024)
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("request_id", "command", "message")
    @classmethod
    def _required_fields(cls, value: str) -> str:
        return _validate_text(value, field_name="response field", limit=1024)

    @field_validator("data", mode="before")
    @classmethod
    def _data(cls, value: object) -> dict[str, object]:
        return _validate_json_mapping(value, field_name="response data")
