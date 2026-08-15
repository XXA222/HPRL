from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Iterable, Protocol, runtime_checkable

from freqtrade.hedge.planning.context import (
    IntentAction,
    OrderIntent,
    PositionBucket,
    PositionSide,
    ZERO,
    utc_aware,
)


class EventKind(StrEnum):
    SIGNAL = "SIGNAL"
    BAR = "BAR"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    FILL = "FILL"
    FUNDING = "FUNDING"
    ACCOUNT = "ACCOUNT"
    LIQUIDATION = "LIQUIDATION"


class LiquidityRole(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class AccountEventType(StrEnum):
    FEE = "FEE"
    FUNDING = "FUNDING"
    BALANCE_ADJUSTMENT = "BALANCE_ADJUSTMENT"
    LIQUIDATION = "LIQUIDATION"


@dataclass(frozen=True, slots=True)
class MarketRules:
    tick_size: Decimal = Decimal("0.01")
    qty_step: Decimal = Decimal("0.0001")
    min_qty: Decimal = ZERO
    min_notional: Decimal = ZERO

    def __post_init__(self) -> None:
        values = (self.tick_size, self.qty_step, self.min_qty, self.min_notional)
        if any(not value.is_finite() for value in values):
            raise ValueError("market rules must be finite")
        if self.tick_size <= ZERO or self.qty_step <= ZERO:
            raise ValueError("market tick and quantity steps must be positive")
        if self.min_qty < ZERO or self.min_notional < ZERO:
            raise ValueError("market minimums cannot be negative")


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Update the deterministic planning signal used at this bar close.

    Orders produced from this signal become eligible for matching on the next
    bar.  ``target_net`` therefore mirrors the live Paper signal contract
    without allowing the analyzed candle to fill its own newly-created orders.
    """

    timestamp: datetime
    symbol: str
    long_signal: Decimal
    short_signal: Decimal
    target_net: Decimal | None = None
    model_version: str = "strategy"
    reason: str = ""
    target_net_ratio: Decimal | None = None
    confidence: Decimal = Decimal("1")
    risk_scale: Decimal = Decimal("1")
    long_exposure_scale: Decimal = Decimal("1")
    short_exposure_scale: Decimal = Decimal("1")
    allow_new_risk: bool = True
    regime: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.symbol.strip():
            raise ValueError("signal symbol cannot be empty")
        if not self.long_signal.is_finite() or not self.short_signal.is_finite():
            raise ValueError("strategy signals must be finite")
        if self.target_net is not None and not self.target_net.is_finite():
            raise ValueError("signal target_net must be finite")
        if self.target_net_ratio is not None and (not self.target_net_ratio.is_finite() or self.target_net_ratio < -1 or self.target_net_ratio > 1):
            raise ValueError("signal target_net_ratio must be within [-1, 1]")
        for name in ("confidence", "risk_scale", "long_exposure_scale", "short_exposure_scale"):
            value = getattr(self, name)
            if not value.is_finite() or value < 0 or value > 1:
                raise ValueError(f"signal {name} must be within [0, 1]")
        if not isinstance(self.allow_new_risk, bool):
            raise TypeError("signal allow_new_risk must be bool")
        if len(self.regime) > 64:
            raise ValueError("signal regime cannot exceed 64 characters")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("signal model_version must be a non-empty string")
        if len(self.model_version) > 128:
            raise ValueError("signal model_version cannot exceed 128 characters")
        if not isinstance(self.reason, str) or len(self.reason) > 256:
            raise ValueError("signal reason must be a string of at most 256 characters")


@dataclass(frozen=True, slots=True)
class BarEvent:
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.symbol.strip():
            raise ValueError("bar symbol cannot be empty")
        values = (self.open, self.high, self.low, self.close)
        if any(not value.is_finite() for value in values):
            raise ValueError("bar values must be finite")
        if self.volume is not None and not self.volume.is_finite():
            raise ValueError("bar volume must be finite when supplied")
        if min(self.open, self.high, self.low, self.close) <= ZERO:
            raise ValueError("OHLC prices must be positive")
        if self.volume is not None and self.volume < ZERO:
            raise ValueError("bar volume cannot be negative")
        if self.high < self.low:
            raise ValueError("bar high cannot be below low")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("invalid OHLC bounds")


@dataclass(frozen=True, slots=True)
class FundingEvent:
    timestamp: datetime
    symbol: str
    rate: Decimal
    mark_price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.symbol.strip():
            raise ValueError("funding symbol cannot be empty")
        if not self.rate.is_finite() or not self.mark_price.is_finite():
            raise ValueError("funding values must be finite")
        if self.mark_price <= ZERO:
            raise ValueError("funding mark price must be positive")


@dataclass(frozen=True, slots=True)
class FillEvent:
    event_id: str
    timestamp: datetime
    order_id: str
    intent_id: str
    symbol: str
    position_side: PositionSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    reduce_only: bool
    bucket: PositionBucket
    action: IntentAction | str
    liquidity_role: LiquidityRole | str = LiquidityRole.TAKER
    layer: int = 0
    tactical_lot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if isinstance(self.action, str):
            object.__setattr__(self, "action", IntentAction(self.action))
        if isinstance(self.liquidity_role, str):
            object.__setattr__(self, "liquidity_role", LiquidityRole(self.liquidity_role))
        if not self.event_id or not self.order_id or not self.intent_id or not self.symbol.strip():
            raise ValueError("fill identifiers cannot be empty")
        if any(not value.is_finite() for value in (self.quantity, self.price, self.fee)):
            raise ValueError("fill values must be finite")
        if self.quantity <= ZERO or self.price <= ZERO:
            raise ValueError("fill quantity and price must be positive")
        if self.fee < ZERO:
            raise ValueError("fill fee cannot be negative")
        if self.layer < 0:
            raise ValueError("fill layer cannot be negative")
        reducing = self.action in {IntentAction.REDUCE, IntentAction.CLOSE, IntentAction.UNSTUCK}
        if reducing != self.reduce_only:
            raise ValueError("fill action and reduce_only must agree")




@dataclass(frozen=True, slots=True)
class AccountEvent:
    event_id: str
    timestamp: datetime
    symbol: str
    event_type: AccountEventType | str
    amount: Decimal
    position_side: PositionSide | None = None
    source_event_id: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if isinstance(self.event_type, str):
            object.__setattr__(self, "event_type", AccountEventType(self.event_type))
        if not self.event_id or not self.symbol.strip():
            raise ValueError("account event identifiers cannot be empty")
        if not self.amount.is_finite():
            raise ValueError("account event amount must be finite")


@dataclass(frozen=True, slots=True)
class LiquidationEvent:
    event_id: str
    timestamp: datetime
    symbol: str
    price: Decimal
    long_quantity: Decimal
    short_quantity: Decimal
    realized_pnl: Decimal
    fee: Decimal
    equity_before: Decimal
    maintenance_margin: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.event_id or not self.symbol.strip():
            raise ValueError("liquidation event identifiers cannot be empty")
        numeric = (
            self.price,
            self.long_quantity,
            self.short_quantity,
            self.realized_pnl,
            self.fee,
            self.equity_before,
            self.maintenance_margin,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("liquidation event values must be finite")
        if self.price <= ZERO:
            raise ValueError("liquidation price must be positive")
        if self.long_quantity < ZERO or self.short_quantity < ZERO or self.fee < ZERO:
            raise ValueError("liquidation quantities and fee cannot be negative")

@dataclass(frozen=True, slots=True)
class OrderAcceptedEvent:
    timestamp: datetime
    order_id: str
    intent: OrderIntent

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.order_id:
            raise ValueError("accepted order id cannot be empty")


@dataclass(frozen=True, slots=True)
class OrderCancelledEvent:
    timestamp: datetime
    order_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.order_id:
            raise ValueError("cancelled order id cannot be empty")


SimulationInputEvent = SignalEvent | BarEvent | FundingEvent
StandardEvent = (
    SignalEvent
    | BarEvent
    | FundingEvent
    | FillEvent
    | AccountEvent
    | LiquidationEvent
    | OrderAcceptedEvent
    | OrderCancelledEvent
)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    timestamp: datetime
    balance: Decimal
    equity: Decimal
    long_quantity: Decimal
    long_average_price: Decimal
    short_quantity: Decimal
    short_average_price: Decimal
    gross_notional: Decimal
    net_notional: Decimal
    fees: Decimal
    funding: Decimal
    realized_pnl: Decimal
    available_balance: Decimal = ZERO
    active_order_margin: Decimal = ZERO
    maintenance_margin: Decimal = ZERO
    margin_ratio: Decimal = ZERO
    long_realized_pnl: Decimal = ZERO
    short_realized_pnl: Decimal = ZERO
    liquidation_buffer: Decimal = ZERO
    liquidation_buffer_ratio: Decimal = ZERO
    liquidated: bool = False


@dataclass(frozen=True, slots=True)
class SimulationResult:
    events: tuple[StandardEvent, ...]
    snapshots: tuple[SimulationSnapshot, ...]
    report: dict[str, Decimal | int | str | bool]


@runtime_checkable
class SimulationPort(Protocol):
    def replay(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        ...
