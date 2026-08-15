from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
from enum import StrEnum
from hashlib import sha256
from json import dumps
from typing import Protocol, runtime_checkable

ZERO = Decimal("0")
ONE = Decimal("1")


def D(value: Decimal | int | float | str) -> Decimal:
    """Create a finite Decimal without inheriting binary float noise."""
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("decimal value must be finite")
    return result


def q_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= ZERO:
        raise ValueError("quantization step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def q_up(value: Decimal, step: Decimal) -> Decimal:
    if step <= ZERO:
        raise ValueError("quantization step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def decimal_key(value: Decimal) -> str:
    """Canonical decimal representation for stable intent identifiers."""
    normalized = value.normalize()
    if normalized == ZERO:
        return "0"
    return format(normalized, "f")


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def direction(self) -> Decimal:
        return ONE if self is PositionSide.LONG else -ONE


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class IntentAction(StrEnum):
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    UNSTUCK = "UNSTUCK"


class PositionBucket(StrEnum):
    CORE = "CORE"
    TACTICAL = "TACTICAL"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"


class TrailingPhase(StrEnum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    mark: Decimal
    tick_size: Decimal = Decimal("0.01")
    qty_step: Decimal = Decimal("0.0001")
    min_qty: Decimal = ZERO
    min_notional: Decimal = ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_aware(self.timestamp))
        if not self.symbol.strip():
            raise ValueError("market symbol cannot be empty")
        numeric = (
            self.bid,
            self.ask,
            self.mark,
            self.tick_size,
            self.qty_step,
            self.min_qty,
            self.min_notional,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("market values must be finite")
        if self.bid <= ZERO or self.ask <= ZERO or self.mark <= ZERO:
            raise ValueError("market prices must be positive")
        if self.bid > self.ask:
            raise ValueError("bid cannot exceed ask")
        if self.tick_size <= ZERO or self.qty_step <= ZERO:
            raise ValueError("market tick and quantity steps must be positive")
        if self.min_qty < ZERO or self.min_notional < ZERO:
            raise ValueError("market minimums cannot be negative")


@dataclass(frozen=True, slots=True)
class TacticalLot:
    lot_id: str
    quantity: Decimal
    average_price: Decimal
    opened_at: datetime
    layer: int = 0
    realized_pnl: Decimal = ZERO
    fees: Decimal = ZERO
    funding: Decimal = ZERO
    closed_quantity: Decimal = ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "opened_at", utc_aware(self.opened_at))
        if not self.lot_id:
            raise ValueError("tactical lot id cannot be empty")
        numeric = (
            self.quantity,
            self.average_price,
            self.realized_pnl,
            self.fees,
            self.funding,
            self.closed_quantity,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("tactical lot values must be finite")
        if self.quantity < ZERO or self.closed_quantity < ZERO:
            raise ValueError("tactical lot quantities cannot be negative")
        if self.quantity > ZERO and self.average_price <= ZERO:
            raise ValueError("open tactical lot average price must be positive")
        if self.quantity == ZERO and self.average_price < ZERO:
            raise ValueError("closed tactical lot average price cannot be negative")
        if self.fees < ZERO:
            raise ValueError("tactical lot fees cannot be negative")
        if self.layer < 0:
            raise ValueError("tactical lot layer cannot be negative")


@dataclass(frozen=True, slots=True)
class LegPosition:
    side: PositionSide
    quantity: Decimal = ZERO
    average_price: Decimal = ZERO
    core_quantity: Decimal = ZERO
    core_average_price: Decimal = ZERO
    tactical_quantity: Decimal = ZERO
    tactical_average_price: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    tactical_realized_pnl: Decimal = ZERO
    tactical_lots: tuple[TacticalLot, ...] = ()

    def __post_init__(self) -> None:
        numeric = (
            self.quantity,
            self.average_price,
            self.core_quantity,
            self.core_average_price,
            self.tactical_quantity,
            self.tactical_average_price,
            self.realized_pnl,
            self.tactical_realized_pnl,
        )
        if any(not value.is_finite() for value in numeric):
            raise ValueError("position values must be finite")
        if min(self.quantity, self.core_quantity, self.tactical_quantity) < ZERO:
            raise ValueError("position quantities cannot be negative")
        if self.core_quantity + self.tactical_quantity != self.quantity:
            raise ValueError("core + tactical quantity must equal total quantity")
        if self.quantity > ZERO and self.average_price <= ZERO:
            raise ValueError("open position average price must be positive")
        if self.core_quantity > ZERO and self.core_average_price <= ZERO:
            raise ValueError("open core average price must be positive")
        if self.tactical_quantity > ZERO and self.tactical_average_price <= ZERO:
            raise ValueError("open tactical average price must be positive")
        if self.quantity == ZERO and self.average_price != ZERO:
            raise ValueError("flat position average price must be zero")
        if self.core_quantity == ZERO and self.core_average_price != ZERO:
            raise ValueError("flat core average price must be zero")
        if self.tactical_quantity == ZERO and self.tactical_average_price != ZERO:
            raise ValueError("flat tactical average price must be zero")
        lot_ids = [lot.lot_id for lot in self.tactical_lots]
        if len(lot_ids) != len(set(lot_ids)):
            raise ValueError("tactical lot ids must be unique")
        open_lot_qty = sum((lot.quantity for lot in self.tactical_lots), ZERO)
        if self.tactical_lots and open_lot_qty != self.tactical_quantity:
            raise ValueError("open tactical lot quantity must equal tactical quantity")

    def unrealized_pnl(self, mark: Decimal) -> Decimal:
        if not mark.is_finite() or mark <= ZERO:
            raise ValueError("mark price must be finite and positive")
        if self.quantity <= ZERO:
            return ZERO
        return (mark - self.average_price) * self.quantity * self.side.direction


@dataclass(frozen=True, slots=True)
class ActiveOrder:
    order_id: str
    symbol: str
    position_side: PositionSide
    order_side: OrderSide
    quantity: Decimal
    price: Decimal
    reduce_only: bool
    bucket: PositionBucket
    action: IntentAction
    created_at: datetime
    client_order_id: str = ""
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    layer: int = 0
    tactical_lot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", utc_aware(self.created_at))
        if not self.order_id or not self.symbol.strip():
            raise ValueError("active order identifiers cannot be empty")
        if not self.quantity.is_finite() or not self.price.is_finite():
            raise ValueError("active order values must be finite")
        if self.quantity <= ZERO or self.price <= ZERO:
            raise ValueError("active order quantity and price must be positive")
        if self.layer < 0:
            raise ValueError("active order layer cannot be negative")
        reducing = self.action in {
            IntentAction.REDUCE,
            IntentAction.CLOSE,
            IntentAction.UNSTUCK,
        }
        if reducing != self.reduce_only:
            raise ValueError("active order action and reduce_only must agree")
        expected_side = (
            OrderSide.SELL
            if self.position_side is PositionSide.LONG and self.reduce_only
            else OrderSide.BUY
            if self.position_side is PositionSide.LONG
            else OrderSide.BUY
            if self.reduce_only
            else OrderSide.SELL
        )
        if self.order_side is not expected_side:
            raise ValueError(
                "active order side is inconsistent with position side and reduce_only"
            )

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class WalletSnapshot:
    balance: Decimal
    equity: Decimal
    available_balance: Decimal
    long: LegPosition
    short: LegPosition
    active_orders: tuple[ActiveOrder, ...] = ()
    leverage: Decimal = ONE

    def __post_init__(self) -> None:
        numeric = (self.balance, self.equity, self.available_balance, self.leverage)
        if any(not value.is_finite() for value in numeric):
            raise ValueError("wallet values must be finite")
        if self.leverage <= ZERO:
            raise ValueError("wallet leverage must be positive")
        if self.long.side is not PositionSide.LONG or self.short.side is not PositionSide.SHORT:
            raise ValueError("wallet legs must use their matching position sides")
        ids = [order.order_id for order in self.active_orders]
        if len(ids) != len(set(ids)):
            raise ValueError("wallet active order ids must be unique")

    def leg(self, side: PositionSide) -> LegPosition:
        return self.long if side is PositionSide.LONG else self.short

    def gross_notional(self, mark: Decimal) -> Decimal:
        return (self.long.quantity + self.short.quantity) * mark

    def net_notional(self, mark: Decimal) -> Decimal:
        return (self.long.quantity - self.short.quantity) * mark


@dataclass(frozen=True, slots=True)
class StrategyLegState:
    side: PositionSide
    grid_layers_filled: int = 0
    last_entry_at: datetime | None = None
    last_reduce_at: datetime | None = None
    trailing_extreme: Decimal | None = None
    trailing_armed: bool = False
    sequence: int = 0
    trailing_phase: TrailingPhase = TrailingPhase.IDLE
    trailing_trigger_price: Decimal | None = None
    trailing_started_at: datetime | None = None
    trailing_confirmed_at: datetime | None = None
    trailing_cooldown_until: datetime | None = None
    last_unstuck_at: datetime | None = None
    unstuck_budget_day: str = ""
    unstuck_budget_week: str = ""
    unstuck_daily_loss: Decimal = ZERO
    unstuck_weekly_loss: Decimal = ZERO

    def __post_init__(self) -> None:
        if isinstance(self.trailing_phase, str):
            object.__setattr__(self, "trailing_phase", TrailingPhase(self.trailing_phase))
        if self.grid_layers_filled < 0 or self.sequence < 0:
            raise ValueError("strategy counters cannot be negative")
        if self.last_entry_at is not None:
            object.__setattr__(self, "last_entry_at", utc_aware(self.last_entry_at))
        if self.last_reduce_at is not None:
            object.__setattr__(self, "last_reduce_at", utc_aware(self.last_reduce_at))
        for field_name in (
            "trailing_started_at",
            "trailing_confirmed_at",
            "trailing_cooldown_until",
            "last_unstuck_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, utc_aware(value))
        if self.trailing_extreme is not None:
            if not self.trailing_extreme.is_finite() or self.trailing_extreme <= ZERO:
                raise ValueError("trailing extreme must be finite and positive")
        if self.trailing_trigger_price is not None:
            if not self.trailing_trigger_price.is_finite() or self.trailing_trigger_price <= ZERO:
                raise ValueError("trailing trigger price must be finite and positive")
        if not self.unstuck_daily_loss.is_finite() or not self.unstuck_weekly_loss.is_finite():
            raise ValueError("unstuck loss budgets must be finite")
        if self.unstuck_daily_loss < ZERO or self.unstuck_weekly_loss < ZERO:
            raise ValueError("unstuck loss budgets cannot be negative")
        if self.trailing_armed and self.trailing_phase is TrailingPhase.IDLE:
            object.__setattr__(self, "trailing_phase", TrailingPhase.CONFIRMED)
        if self.trailing_phase is TrailingPhase.CONFIRMED and not self.trailing_armed:
            object.__setattr__(self, "trailing_armed", True)
        if self.trailing_phase is not TrailingPhase.CONFIRMED and self.trailing_armed:
            raise ValueError("only confirmed trailing phase may be armed")

    def next_sequence(self) -> "StrategyLegState":
        return replace(self, sequence=self.sequence + 1)


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    long_enabled: bool = True
    short_enabled: bool = True
    core_wallet_exposure_long: Decimal = Decimal("0.20")
    core_wallet_exposure_short: Decimal = Decimal("0.20")
    tactical_wallet_exposure_long: Decimal = Decimal("0.10")
    tactical_wallet_exposure_short: Decimal = Decimal("0.10")
    max_wallet_exposure_long: Decimal = Decimal("0.40")
    max_wallet_exposure_short: Decimal = Decimal("0.40")
    max_gross_wallet_exposure: Decimal = Decimal("0.65")
    initial_entry_fraction: Decimal = Decimal("0.25")
    max_grid_layers: int = 4
    grid_spacing: Decimal = Decimal("0.01")
    grid_spacing_growth: Decimal = Decimal("1.20")
    grid_qty_growth: Decimal = Decimal("1.15")
    trailing_rebound: Decimal = Decimal("0.003")
    take_profit_spacing: Decimal = Decimal("0.008")
    take_profit_layers: int = 3
    tactical_reduce_fraction: Decimal = Decimal("0.34")
    core_min_fraction: Decimal = Decimal("0.75")
    cooldown_seconds: int = 60
    replace_price_tolerance_ticks: int = 2
    replace_qty_tolerance_steps: int = 5
    replace_min_age_seconds: int = 15
    unstuck_trigger_gross_exposure: Decimal = Decimal("0.60")
    unstuck_reduce_fraction: Decimal = Decimal("0.20")
    maintenance_margin_rate: Decimal = Decimal("0.005")
    target_net_wallet_exposure: Decimal = ZERO
    net_repair_threshold: Decimal = Decimal("0.02")
    trailing_trigger_distance: Decimal = Decimal("0.01")
    trailing_timeout_seconds: int = 900
    max_pending_entries: int = 4
    max_single_order_notional: Decimal = ZERO
    unstuck_max_holding_seconds: int = 86400
    unstuck_daily_loss_budget: Decimal = Decimal("0.01")
    unstuck_weekly_loss_budget: Decimal = Decimal("0.03")
    unstuck_min_cooldown_seconds: int = 3600
    unstuck_min_risk_improvement: Decimal = Decimal("0.001")
    unstuck_limit_only: bool = False
    liquidation_fee_rate: Decimal = Decimal("0.005")
    liquidation_buffer_warning_ratio: Decimal = Decimal("0.05")

    def __post_init__(self) -> None:
        decimal_fields = (
            self.core_wallet_exposure_long,
            self.core_wallet_exposure_short,
            self.tactical_wallet_exposure_long,
            self.tactical_wallet_exposure_short,
            self.max_wallet_exposure_long,
            self.max_wallet_exposure_short,
            self.max_gross_wallet_exposure,
            self.initial_entry_fraction,
            self.grid_spacing,
            self.grid_spacing_growth,
            self.grid_qty_growth,
            self.trailing_rebound,
            self.take_profit_spacing,
            self.tactical_reduce_fraction,
            self.core_min_fraction,
            self.unstuck_trigger_gross_exposure,
            self.unstuck_reduce_fraction,
            self.maintenance_margin_rate,
            self.target_net_wallet_exposure,
            self.net_repair_threshold,
            self.trailing_trigger_distance,
            self.max_single_order_notional,
            self.unstuck_daily_loss_budget,
            self.unstuck_weekly_loss_budget,
            self.unstuck_min_risk_improvement,
            self.liquidation_fee_rate,
            self.liquidation_buffer_warning_ratio,
        )
        if any(not value.is_finite() for value in decimal_fields):
            raise ValueError("planner decimal values must be finite")
        ratio_fields = (
            self.core_wallet_exposure_long,
            self.core_wallet_exposure_short,
            self.tactical_wallet_exposure_long,
            self.tactical_wallet_exposure_short,
            self.max_wallet_exposure_long,
            self.max_wallet_exposure_short,
            self.max_gross_wallet_exposure,
            self.initial_entry_fraction,
            self.trailing_rebound,
            self.tactical_reduce_fraction,
            self.core_min_fraction,
            self.unstuck_trigger_gross_exposure,
            self.unstuck_reduce_fraction,
            self.maintenance_margin_rate,
            self.net_repair_threshold,
            self.trailing_trigger_distance,
            self.unstuck_daily_loss_budget,
            self.unstuck_weekly_loss_budget,
            self.unstuck_min_risk_improvement,
            self.liquidation_fee_rate,
            self.liquidation_buffer_warning_ratio,
        )
        if any(value < ZERO for value in ratio_fields):
            raise ValueError("planner ratios cannot be negative")
        bounded_fractions = (
            self.initial_entry_fraction,
            self.trailing_rebound,
            self.tactical_reduce_fraction,
            self.core_min_fraction,
            self.unstuck_reduce_fraction,
            self.maintenance_margin_rate,
            self.net_repair_threshold,
            self.trailing_trigger_distance,
            self.unstuck_daily_loss_budget,
            self.unstuck_weekly_loss_budget,
            self.unstuck_min_risk_improvement,
            self.liquidation_fee_rate,
            self.liquidation_buffer_warning_ratio,
        )
        if any(value > ONE for value in bounded_fractions):
            raise ValueError("planner fractions cannot exceed one")
        if self.trailing_rebound >= ONE:
            raise ValueError("trailing rebound must be less than one")
        if self.max_grid_layers < 0 or self.take_profit_layers < 0:
            raise ValueError("layer counts cannot be negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown cannot be negative")
        if self.trailing_timeout_seconds < 0:
            raise ValueError("trailing timeout cannot be negative")
        if self.max_pending_entries < 0:
            raise ValueError("max pending entries cannot be negative")
        if self.unstuck_max_holding_seconds < 0 or self.unstuck_min_cooldown_seconds < 0:
            raise ValueError("unstuck timing values cannot be negative")
        if not isinstance(self.unstuck_limit_only, bool):
            raise TypeError("unstuck_limit_only must be a boolean")
        if self.max_single_order_notional < ZERO:
            raise ValueError("maximum single-order notional cannot be negative")
        if self.replace_min_age_seconds < 0:
            raise ValueError("replacement minimum age cannot be negative")
        if self.replace_price_tolerance_ticks < 0 or self.replace_qty_tolerance_steps < 0:
            raise ValueError("replacement tolerances cannot be negative")
        if self.grid_spacing <= ZERO or self.take_profit_spacing <= ZERO:
            raise ValueError("grid spacing must be positive")
        if self.grid_qty_growth <= ZERO or self.grid_spacing_growth <= ZERO:
            raise ValueError("grid growth factors must be positive")
        if self.max_wallet_exposure_long > self.max_gross_wallet_exposure:
            raise ValueError("long exposure cap exceeds gross cap")
        if self.max_wallet_exposure_short > self.max_gross_wallet_exposure:
            raise ValueError("short exposure cap exceeds gross cap")

    def enabled(self, side: PositionSide) -> bool:
        return self.long_enabled if side is PositionSide.LONG else self.short_enabled

    def core_exposure(self, side: PositionSide) -> Decimal:
        if side is PositionSide.LONG:
            return self.core_wallet_exposure_long
        return self.core_wallet_exposure_short

    def tactical_exposure(self, side: PositionSide) -> Decimal:
        if side is PositionSide.LONG:
            return self.tactical_wallet_exposure_long
        return self.tactical_wallet_exposure_short

    def side_cap(self, side: PositionSide) -> Decimal:
        if side is PositionSide.LONG:
            return self.max_wallet_exposure_long
        return self.max_wallet_exposure_short


@dataclass(frozen=True, slots=True)
class PlanningContext:
    market: MarketSnapshot
    wallet: WalletSnapshot
    config: PlannerConfig
    long_state: StrategyLegState = field(
        default_factory=lambda: StrategyLegState(PositionSide.LONG)
    )
    short_state: StrategyLegState = field(
        default_factory=lambda: StrategyLegState(PositionSide.SHORT)
    )
    long_signal: Decimal = ZERO
    short_signal: Decimal = ZERO
    target_net_quantity: Decimal | None = None

    def __post_init__(self) -> None:
        if (
            self.long_state.side is not PositionSide.LONG
            or self.short_state.side is not PositionSide.SHORT
        ):
            raise ValueError("strategy leg states must match their configured sides")
        if not self.long_signal.is_finite() or not self.short_signal.is_finite():
            raise ValueError("strategy signals must be finite")
        if self.target_net_quantity is not None and not self.target_net_quantity.is_finite():
            raise ValueError("target net quantity must be finite when supplied")

    def state(self, side: PositionSide) -> StrategyLegState:
        return self.long_state if side is PositionSide.LONG else self.short_state

    def signal(self, side: PositionSide) -> Decimal:
        return self.long_signal if side is PositionSide.LONG else self.short_signal


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    symbol: str
    position_side: PositionSide
    order_side: OrderSide
    action: IntentAction
    bucket: PositionBucket
    quantity: Decimal
    price: Decimal
    reduce_only: bool
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    layer: int = 0
    reason: str = ""
    tactical_lot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.intent_id or not self.symbol.strip():
            raise ValueError("intent identifiers cannot be empty")
        if not self.quantity.is_finite() or not self.price.is_finite():
            raise ValueError("intent values must be finite")
        if self.quantity <= ZERO or self.price <= ZERO:
            raise ValueError("intent quantity and price must be positive")
        if self.layer < 0:
            raise ValueError("intent layer cannot be negative")
        reducing = self.action in {IntentAction.REDUCE, IntentAction.CLOSE, IntentAction.UNSTUCK}
        if reducing != self.reduce_only:
            raise ValueError("reduce actions and reduce_only flag must agree")
        expected_side = (
            OrderSide.SELL
            if self.position_side is PositionSide.LONG and self.reduce_only
            else OrderSide.BUY
            if self.position_side is PositionSide.LONG
            else OrderSide.BUY
            if self.reduce_only
            else OrderSide.SELL
        )
        if self.order_side is not expected_side:
            raise ValueError("order side is inconsistent with position side and reduce_only")

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price

    @staticmethod
    def deterministic(
        *,
        symbol: str,
        position_side: PositionSide,
        order_side: OrderSide,
        action: IntentAction,
        bucket: PositionBucket,
        quantity: Decimal,
        price: Decimal,
        reduce_only: bool,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.GTC,
        layer: int = 0,
        reason: str = "",
        epoch: str = "",
        tactical_lot_id: str | None = None,
    ) -> "OrderIntent":
        payload = {
            "symbol": symbol,
            "position_side": position_side.value,
            "order_side": order_side.value,
            "action": action.value,
            "bucket": bucket.value,
            "quantity": decimal_key(quantity),
            "price": decimal_key(price),
            "reduce_only": reduce_only,
            "order_type": order_type.value,
            "time_in_force": time_in_force.value,
            "layer": layer,
            "reason": reason,
            "epoch": epoch,
            "tactical_lot_id": tactical_lot_id or "",
        }
        serialized = dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = sha256(serialized.encode()).hexdigest()[:24]
        return OrderIntent(
            intent_id=f"hp-{digest}",
            symbol=symbol,
            position_side=position_side,
            order_side=order_side,
            action=action,
            bucket=bucket,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only,
            order_type=order_type,
            time_in_force=time_in_force,
            layer=layer,
            reason=reason,
            tactical_lot_id=tactical_lot_id,
        )


@dataclass(frozen=True, slots=True)
class PlanningResult:
    ideal_orders: tuple[OrderIntent, ...]
    submit_orders: tuple[OrderIntent, ...]
    cancel_order_ids: tuple[str, ...]
    kept_order_ids: tuple[str, ...]
    long_state: StrategyLegState
    short_state: StrategyLegState
    diagnostics: tuple[str, ...] = ()
    modify_order_ids: tuple[str, ...] = ()
    delete_order_ids: tuple[str, ...] = ()
    risk_cancel_order_ids: tuple[str, ...] = ()
    target_net_quantity: Decimal = ZERO
    net_gap_quantity: Decimal = ZERO
    long_target_quantity: Decimal = ZERO
    short_target_quantity: Decimal = ZERO


@runtime_checkable
class StrategyPlanningPort(Protocol):
    def plan(self, context: PlanningContext) -> PlanningResult:
        """Return deterministic desired orders and next immutable strategy state."""
        ...
