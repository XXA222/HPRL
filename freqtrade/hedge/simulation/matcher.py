from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256

from freqtrade.hedge.planning.context import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
    ZERO,
    q_down,
    q_up,
)
from .cross_wallet import CrossWallet
from .exchange import BarEvent, FillEvent, LiquidationEvent, LiquidityRole


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Deterministic fill, fee and liquidity model shared by all simulation modes."""

    fee_rate: Decimal | None = None
    maker_fee_rate: Decimal = Decimal("0.0004")
    taker_fee_rate: Decimal = Decimal("0.0004")
    volume_participation: Decimal = Decimal("0.10")
    market_slippage_bps: Decimal = ZERO
    price_tick: Decimal = Decimal("0.00000001")
    qty_step: Decimal = Decimal("0.00000001")
    min_fill_qty: Decimal = ZERO
    min_fill_notional: Decimal = ZERO
    max_entry_layers_per_bar: int = 1
    max_reduce_layers_per_bar: int = 1
    max_fill_ratio_per_order: Decimal = Decimal("1")
    max_fills_per_bar: int = 0

    def __post_init__(self) -> None:
        if self.fee_rate is not None:
            if not self.fee_rate.is_finite() or self.fee_rate < ZERO:
                raise ValueError("matcher fee rate must be finite and non-negative")
            object.__setattr__(self, "maker_fee_rate", self.fee_rate)
            object.__setattr__(self, "taker_fee_rate", self.fee_rate)
        values = (
            self.maker_fee_rate,
            self.taker_fee_rate,
            self.volume_participation,
            self.market_slippage_bps,
            self.price_tick,
            self.qty_step,
            self.min_fill_qty,
            self.min_fill_notional,
            self.max_fill_ratio_per_order,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("matcher configuration must be finite")
        if self.maker_fee_rate < ZERO or self.taker_fee_rate < ZERO:
            raise ValueError("matcher fee rates cannot be negative")
        if not ZERO <= self.volume_participation <= Decimal("1"):
            raise ValueError("volume participation must be between zero and one")
        if not ZERO <= self.market_slippage_bps < Decimal("10000"):
            raise ValueError("market slippage must be between zero and 10000 bps")
        if self.price_tick <= ZERO or self.qty_step <= ZERO:
            raise ValueError("matcher price and quantity steps must be positive")
        if self.min_fill_qty < ZERO or self.min_fill_notional < ZERO:
            raise ValueError("matcher minimum fills cannot be negative")
        if self.max_entry_layers_per_bar < 0 or self.max_reduce_layers_per_bar < 0:
            raise ValueError("per-bar layer limits cannot be negative")
        if not ZERO < self.max_fill_ratio_per_order <= Decimal("1"):
            raise ValueError("max fill ratio per order must be in (0, 1]")
        if self.max_fills_per_bar < 0:
            raise ValueError("max fills per bar cannot be negative")

    def fee_for(self, role: LiquidityRole) -> Decimal:
        return (
            self.maker_fee_rate
            if role is LiquidityRole.MAKER
            else self.taker_fee_rate
        )


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    fills: tuple[FillEvent, ...]
    expired_order_ids: tuple[str, ...]
    path: tuple[Decimal, ...]
    ending_equity: Decimal
    gross_peak: Decimal
    equity_peak: Decimal
    max_drawdown: Decimal
    liquidation_event: LiquidationEvent | None = None


@dataclass(frozen=True, slots=True)
class _Trigger:
    price: Decimal
    liquidity_role: LiquidityRole


def _event_id(
    order_id: str,
    timestamp: datetime,
    quantity: Decimal,
    price: Decimal,
    ordinal: int,
) -> str:
    raw = f"{order_id}|{timestamp.isoformat()}|{quantity}|{price}|{ordinal}".encode()
    return "fill-" + sha256(raw).hexdigest()[:24]


def _market_price(
    intent: OrderIntent,
    reference: Decimal,
    config: MatchConfig,
) -> Decimal:
    adjustment = config.market_slippage_bps / Decimal("10000")
    raw = reference * (
        Decimal("1") + adjustment
        if intent.order_side is OrderSide.BUY
        else Decimal("1") - adjustment
    )
    return (
        q_up(raw, config.price_tick)
        if intent.order_side is OrderSide.BUY
        else q_down(raw, config.price_tick)
    )


def _trigger_price(
    intent: OrderIntent,
    segment_start: Decimal,
    segment_end: Decimal,
    segment_index: int,
    *,
    accepted_at: datetime | None,
    bar_timestamp: datetime,
    config: MatchConfig,
) -> _Trigger | None:
    if intent.order_type is OrderType.MARKET:
        if segment_index != 0:
            return None
        return _Trigger(
            _market_price(intent, segment_start, config),
            LiquidityRole.TAKER,
        )
    if intent.time_in_force is TimeInForce.IOC:
        if segment_index != 0:
            return None
        if intent.order_side is OrderSide.BUY and segment_start <= intent.price:
            return _Trigger(segment_start, LiquidityRole.TAKER)
        if intent.order_side is OrderSide.SELL and segment_start >= intent.price:
            return _Trigger(segment_start, LiquidityRole.TAKER)
        return None

    resting = accepted_at is None or accepted_at < bar_timestamp
    open_role = LiquidityRole.MAKER if resting else LiquidityRole.TAKER
    if intent.order_side is OrderSide.BUY:
        if segment_start <= intent.price:
            return _Trigger(segment_start, open_role)
        if segment_end < segment_start and segment_end <= intent.price <= segment_start:
            return _Trigger(intent.price, LiquidityRole.MAKER)
    else:
        if segment_start >= intent.price:
            return _Trigger(segment_start, open_role)
        if segment_end > segment_start and segment_start <= intent.price <= segment_end:
            return _Trigger(intent.price, LiquidityRole.MAKER)
    return None


class ConservativeMatcher:
    """Evaluate admissible OHLC paths and commit the lower-equity/riskier outcome."""

    def __init__(self, config: MatchConfig | None = None) -> None:
        self.config = config or MatchConfig()

    @staticmethod
    def _clone(wallet: CrossWallet) -> CrossWallet:
        import copy

        return copy.deepcopy(wallet)

    def _fillable_quantity(
        self,
        *,
        remaining: Decimal,
        volume_budget: Decimal,
        price: Decimal,
        reduce_capacity: Decimal | None,
    ) -> Decimal:
        qty = min(
            remaining,
            remaining * self.config.max_fill_ratio_per_order,
            volume_budget,
        )
        if reduce_capacity is not None:
            qty = min(qty, reduce_capacity)
        qty = q_down(qty, self.config.qty_step)
        if qty <= ZERO:
            return ZERO
        if qty < self.config.min_fill_qty:
            return ZERO
        if qty * price < self.config.min_fill_notional:
            return ZERO
        return qty

    def _path_outcome(
        self,
        bar: BarEvent,
        wallet: CrossWallet,
        path: tuple[Decimal, ...],
    ) -> MatchOutcome:
        shadow = self._clone(wallet)
        volume_budget = (
            Decimal("Infinity")
            if bar.volume is None
            else bar.volume * self.config.volume_participation
        )
        fills: list[FillEvent] = []
        expired: set[str] = set()
        seen: set[str] = set()
        ordinal = 0
        liquidation_event: LiquidationEvent | None = None
        filled_layers: dict[tuple[object, bool], set[int]] = {}

        def layer_allowed(intent: OrderIntent) -> bool:
            key = (intent.position_side, intent.reduce_only)
            layers = filled_layers.setdefault(key, set())
            if intent.layer in layers:
                return True
            limit = (
                self.config.max_reduce_layers_per_bar
                if intent.reduce_only
                else self.config.max_entry_layers_per_bar
            )
            return limit == 0 or len(layers) < limit

        def record_layer(intent: OrderIntent) -> None:
            filled_layers.setdefault(
                (intent.position_side, intent.reduce_only),
                set(),
            ).add(intent.layer)

        def liquidate_if_needed(price: Decimal) -> bool:
            nonlocal liquidation_event, ordinal
            if not shadow.should_liquidate(price):
                return False
            liquidation_event = shadow.create_liquidation_event(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                price=price,
                ordinal=ordinal,
            )
            shadow.apply_liquidation(liquidation_event)
            shadow.observe_risk(price)
            ordinal += 1
            return True

        shadow.observe_risk(path[0])
        if liquidate_if_needed(path[0]):
            return MatchOutcome(
                fills=(),
                expired_order_ids=tuple(sorted(wallet.active_orders)),
                path=path,
                ending_equity=shadow.equity(bar.close),
                gross_peak=shadow.gross_peak,
                equity_peak=shadow.equity_peak,
                max_drawdown=shadow.max_drawdown,
                liquidation_event=liquidation_event,
            )

        for segment_index, (segment_start, segment_end) in enumerate(zip(path, path[1:])):
            shadow.observe_risk(segment_start)
            candidates: list[tuple[Decimal, bool, str, _Trigger]] = []
            for order_id, (intent, remaining) in sorted(shadow.active_orders.items()):
                if order_id in seen or remaining <= ZERO or not layer_allowed(intent):
                    continue
                trigger = _trigger_price(
                    intent,
                    segment_start,
                    segment_end,
                    segment_index,
                    accepted_at=shadow.order_accepted_at(order_id),
                    bar_timestamp=bar.timestamp,
                    config=self.config,
                )
                if trigger is None:
                    if segment_index == 0 and (
                        intent.order_type is OrderType.MARKET
                        or intent.time_in_force is TimeInForce.IOC
                    ):
                        shadow.cancel_order(order_id)
                        expired.add(order_id)
                        seen.add(order_id)
                    continue
                candidates.append(
                    (
                        abs(trigger.price - segment_start),
                        not intent.reduce_only,
                        order_id,
                        trigger,
                    )
                )

            for _, _, order_id, trigger in sorted(candidates, key=lambda item: item[:3]):
                if (
                    self.config.max_fills_per_bar > 0
                    and len(fills) >= self.config.max_fills_per_bar
                ):
                    break
                item = shadow.active_orders.get(order_id)
                if item is None or order_id in seen or volume_budget <= ZERO:
                    continue
                intent, remaining = item
                if not layer_allowed(intent):
                    continue
                reduce_capacity = (
                    shadow.leg(intent.position_side).quantity
                    if intent.reduce_only
                    else None
                )
                qty = self._fillable_quantity(
                    remaining=remaining,
                    volume_budget=volume_budget,
                    price=trigger.price,
                    reduce_capacity=reduce_capacity,
                )
                if qty > ZERO:
                    fee = qty * trigger.price * self.config.fee_for(
                        trigger.liquidity_role
                    )
                    fill = FillEvent(
                        event_id=_event_id(
                            order_id,
                            bar.timestamp,
                            qty,
                            trigger.price,
                            ordinal,
                        ),
                        timestamp=bar.timestamp,
                        order_id=order_id,
                        intent_id=intent.intent_id,
                        symbol=intent.symbol,
                        position_side=intent.position_side,
                        quantity=qty,
                        price=trigger.price,
                        fee=fee,
                        reduce_only=intent.reduce_only,
                        bucket=intent.bucket,
                        action=intent.action,
                        liquidity_role=trigger.liquidity_role,
                        layer=intent.layer,
                        tactical_lot_id=intent.tactical_lot_id,
                    )
                    shadow.apply_fill(fill)
                    record_layer(intent)
                    shadow.observe_risk(trigger.price)
                    fills.append(fill)
                    ordinal += 1
                    volume_budget -= qty
                    if liquidate_if_needed(trigger.price):
                        break
                if (
                    intent.order_type is OrderType.MARKET
                    or intent.time_in_force is TimeInForce.IOC
                ):
                    if shadow.remaining(order_id) > ZERO:
                        shadow.cancel_order(order_id)
                        expired.add(order_id)
                seen.add(order_id)

            if liquidation_event is not None:
                break
            if segment_index == 0:
                for order_id, (intent, remaining) in tuple(shadow.active_orders.items()):
                    if remaining <= ZERO:
                        continue
                    if (
                        intent.order_type is OrderType.MARKET
                        or intent.time_in_force is TimeInForce.IOC
                    ):
                        shadow.cancel_order(order_id)
                        expired.add(order_id)
                        seen.add(order_id)
            shadow.observe_risk(segment_end)
            if liquidate_if_needed(segment_end):
                break

        return MatchOutcome(
            fills=tuple(fills),
            expired_order_ids=tuple(sorted(expired)),
            path=path,
            ending_equity=shadow.equity(bar.close),
            gross_peak=shadow.gross_peak,
            equity_peak=shadow.equity_peak,
            max_drawdown=shadow.max_drawdown,
            liquidation_event=liquidation_event,
        )

    def match_outcome(self, bar: BarEvent, wallet: CrossWallet) -> MatchOutcome:
        paths = (
            (bar.open, bar.high, bar.low, bar.close),
            (bar.open, bar.low, bar.high, bar.close),
        )
        candidates = [self._path_outcome(bar, wallet, path) for path in paths]
        candidates.sort(
            key=lambda item: (
                item.ending_equity,
                -item.max_drawdown,
                len(item.fills),
                item.path,
            )
        )
        return candidates[0]

    def match(self, bar: BarEvent, wallet: CrossWallet) -> tuple[FillEvent, ...]:
        return self.match_outcome(bar, wallet).fills
