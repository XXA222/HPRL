from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from hashlib import sha256

from freqtrade.hedge.planning.context import (
    ActiveOrder,
    IntentAction,
    LegPosition,
    OrderIntent,
    PositionBucket,
    PositionSide,
    TacticalLot,
    WalletSnapshot,
    ZERO,
    utc_aware,
)
from .exchange import FillEvent, LiquidationEvent, LiquidityRole, SimulationSnapshot


@dataclass(slots=True)
class MutableTacticalLot:
    lot_id: str
    quantity: Decimal
    average_price: Decimal
    opened_at: datetime
    layer: int = 0
    realized_pnl: Decimal = ZERO
    fees: Decimal = ZERO
    funding: Decimal = ZERO
    closed_quantity: Decimal = ZERO

    def increase(self, qty: Decimal, price: Decimal, fee: Decimal) -> None:
        total = self.quantity + qty
        self.average_price = (
            (self.average_price * self.quantity) + price * qty
        ) / total
        self.quantity = total
        self.fees += fee

    def reduce(self, qty: Decimal, price: Decimal, fee: Decimal, direction: Decimal) -> Decimal:
        if qty > self.quantity:
            raise ValueError("tactical lot reduction exceeds lot quantity")
        pnl = (price - self.average_price) * qty * direction
        self.quantity -= qty
        self.closed_quantity += qty
        self.realized_pnl += pnl
        self.fees += fee
        return pnl

    def immutable(self) -> TacticalLot:
        return TacticalLot(
            lot_id=self.lot_id,
            quantity=self.quantity,
            average_price=self.average_price,
            opened_at=self.opened_at,
            layer=self.layer,
            realized_pnl=self.realized_pnl,
            fees=self.fees,
            funding=self.funding,
            closed_quantity=self.closed_quantity,
        )


@dataclass(slots=True)
class MutableLeg:
    side: PositionSide
    quantity: Decimal = ZERO
    average_price: Decimal = ZERO
    core_quantity: Decimal = ZERO
    core_average_price: Decimal = ZERO
    tactical_quantity: Decimal = ZERO
    tactical_average_price: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    tactical_realized_pnl: Decimal = ZERO
    add_count: int = 0
    reduce_count: int = 0
    tactical_lots: dict[str, MutableTacticalLot] = field(default_factory=dict)

    def _refresh_tactical_aggregate(self) -> None:
        open_lots = [lot for lot in self.tactical_lots.values() if lot.quantity > ZERO]
        self.tactical_quantity = sum((lot.quantity for lot in open_lots), ZERO)
        if self.tactical_quantity > ZERO:
            self.tactical_average_price = sum(
                (lot.average_price * lot.quantity for lot in open_lots),
                ZERO,
            ) / self.tactical_quantity
        else:
            self.tactical_average_price = ZERO

    def _legacy_lot_id(self, layer: int) -> str:
        raw = f"{self.side.value}|legacy|{layer}|{len(self.tactical_lots)}".encode()
        return "lot-" + sha256(raw).hexdigest()[:24]

    def _increase_bucket(
        self,
        bucket: PositionBucket,
        qty: Decimal,
        price: Decimal,
        *,
        tactical_lot_id: str | None,
        opened_at: datetime,
        layer: int,
        fee: Decimal,
    ) -> None:
        if bucket is PositionBucket.CORE:
            total = self.core_quantity + qty
            self.core_average_price = (
                (self.core_average_price * self.core_quantity) + price * qty
            ) / total
            self.core_quantity = total
            return

        lot_id = tactical_lot_id or self._legacy_lot_id(layer)
        lot = self.tactical_lots.get(lot_id)
        if lot is None:
            self.tactical_lots[lot_id] = MutableTacticalLot(
                lot_id=lot_id,
                quantity=qty,
                average_price=price,
                opened_at=opened_at,
                layer=layer,
                fees=fee,
            )
        else:
            lot.increase(qty, price, fee)
        self._refresh_tactical_aggregate()

    def increase(
        self,
        qty: Decimal,
        price: Decimal,
        bucket: PositionBucket,
        *,
        tactical_lot_id: str | None,
        opened_at: datetime,
        layer: int,
        fee: Decimal,
    ) -> None:
        if qty <= ZERO:
            raise ValueError("increase quantity must be positive")
        total = self.quantity + qty
        self.average_price = ((self.average_price * self.quantity) + price * qty) / total
        self.quantity = total
        self._increase_bucket(
            bucket,
            qty,
            price,
            tactical_lot_id=tactical_lot_id,
            opened_at=opened_at,
            layer=layer,
            fee=fee,
        )
        self.add_count += 1

    def _open_lots(self, preferred: str | None = None) -> list[MutableTacticalLot]:
        lots = [lot for lot in self.tactical_lots.values() if lot.quantity > ZERO]
        lots.sort(key=lambda item: (item.opened_at, item.layer, item.lot_id))
        if preferred is None:
            return lots
        return sorted(lots, key=lambda item: item.lot_id != preferred)

    def _reduce_tactical(
        self,
        qty: Decimal,
        price: Decimal,
        fee: Decimal,
        tactical_lot_id: str | None,
    ) -> tuple[Decimal, Decimal]:
        remaining = min(qty, self.tactical_quantity)
        reduced = ZERO
        pnl = ZERO
        for lot in self._open_lots(tactical_lot_id):
            if remaining <= ZERO:
                break
            lot_qty = min(remaining, lot.quantity)
            lot_fee = fee * lot_qty / qty
            pnl += lot.reduce(lot_qty, price, lot_fee, self.side.direction)
            reduced += lot_qty
            remaining -= lot_qty
        self._refresh_tactical_aggregate()
        return pnl, reduced

    def reduce(
        self,
        qty: Decimal,
        price: Decimal,
        bucket: PositionBucket,
        *,
        fee: Decimal,
        tactical_lot_id: str | None,
    ) -> tuple[Decimal, Decimal, Decimal]:
        if qty <= ZERO:
            raise ValueError("reduce quantity must be positive")
        if qty > self.quantity:
            raise ValueError("reduce quantity exceeds open position")
        pnl = (price - self.average_price) * qty * self.side.direction
        tactical_pnl = ZERO
        tactical_reduced_qty = ZERO

        if bucket is PositionBucket.TACTICAL and self.tactical_quantity > ZERO:
            tactical_pnl, tactical_reduced_qty = self._reduce_tactical(
                qty,
                price,
                fee,
                tactical_lot_id,
            )
            core_qty = qty - tactical_reduced_qty
            if core_qty > ZERO:
                self.core_quantity -= core_qty
                if self.core_quantity == ZERO:
                    self.core_average_price = ZERO
        else:
            core_qty = min(qty, self.core_quantity)
            self.core_quantity -= core_qty
            if self.core_quantity == ZERO:
                self.core_average_price = ZERO
            tactical_qty = qty - core_qty
            if tactical_qty > ZERO:
                tactical_pnl, tactical_reduced_qty = self._reduce_tactical(
                    tactical_qty,
                    price,
                    fee * tactical_qty / qty,
                    tactical_lot_id,
                )

        self.quantity -= qty
        self.realized_pnl += pnl
        self.tactical_realized_pnl += tactical_pnl
        self.reduce_count += 1
        if self.quantity == ZERO:
            self.average_price = ZERO
            self.core_quantity = ZERO
            self.core_average_price = ZERO
            self.tactical_quantity = ZERO
            self.tactical_average_price = ZERO
        return pnl, tactical_pnl, tactical_reduced_qty

    def allocate_funding(self, amount: Decimal) -> None:
        if self.quantity <= ZERO or self.tactical_quantity <= ZERO:
            return
        tactical_amount = amount * self.tactical_quantity / self.quantity
        for lot in self._open_lots():
            lot.funding += tactical_amount * lot.quantity / self.tactical_quantity

    def oldest_tactical_opened_at(self) -> datetime | None:
        lots = self._open_lots()
        return lots[0].opened_at if lots else None

    def immutable(self) -> LegPosition:
        return LegPosition(
            side=self.side,
            quantity=self.quantity,
            average_price=self.average_price,
            core_quantity=self.core_quantity,
            core_average_price=self.core_average_price,
            tactical_quantity=self.tactical_quantity,
            tactical_average_price=self.tactical_average_price,
            realized_pnl=self.realized_pnl,
            tactical_realized_pnl=self.tactical_realized_pnl,
            tactical_lots=tuple(
                lot.immutable()
                for lot in sorted(
                    self.tactical_lots.values(),
                    key=lambda item: (item.opened_at, item.layer, item.lot_id),
                )
            ),
        )


@dataclass(slots=True)
class CrossWallet:
    initial_balance: Decimal
    leverage: Decimal = Decimal("1")
    fee_rate: Decimal = Decimal("0.0004")
    balance: Decimal = field(init=False)
    long: MutableLeg = field(default_factory=lambda: MutableLeg(PositionSide.LONG))
    short: MutableLeg = field(default_factory=lambda: MutableLeg(PositionSide.SHORT))
    active_orders: dict[str, tuple[OrderIntent, Decimal]] = field(default_factory=dict)
    order_accepted_times: dict[str, datetime] = field(default_factory=dict)
    processed_fill_ids: set[str] = field(default_factory=set)
    processed_liquidation_ids: set[str] = field(default_factory=set)
    realized_by_fill: dict[str, Decimal] = field(default_factory=dict)
    fees_paid: Decimal = ZERO
    maker_fees_paid: Decimal = ZERO
    taker_fees_paid: Decimal = ZERO
    maker_fill_count: int = 0
    taker_fill_count: int = 0
    long_fees_paid: Decimal = ZERO
    short_fees_paid: Decimal = ZERO
    tactical_fees_paid: Decimal = ZERO
    funding_paid: Decimal = ZERO
    long_funding: Decimal = ZERO
    short_funding: Decimal = ZERO
    tactical_funding: Decimal = ZERO
    gross_peak: Decimal = ZERO
    max_drawdown: Decimal = ZERO
    equity_peak: Decimal = field(init=False)
    hedge_duration_seconds: Decimal = ZERO
    last_timestamp: datetime | None = None
    dual_leg_active: bool = False
    core_cost_basis_initial: dict[PositionSide, Decimal] = field(default_factory=dict)
    maintenance_margin_rate: Decimal = Decimal("0.005")
    liquidation_fee_rate: Decimal = Decimal("0.005")
    liquidation_buffer_warning_ratio: Decimal = Decimal("0.05")
    liquidated: bool = False
    liquidation_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.initial_balance,
            self.leverage,
            self.fee_rate,
            self.maintenance_margin_rate,
            self.liquidation_fee_rate,
            self.liquidation_buffer_warning_ratio,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("wallet configuration must be finite")
        if self.initial_balance < ZERO:
            raise ValueError("initial balance cannot be negative")
        if self.leverage <= ZERO:
            raise ValueError("leverage must be positive")
        if self.fee_rate < ZERO:
            raise ValueError("fee rate cannot be negative")
        if not ZERO <= self.maintenance_margin_rate <= Decimal("1"):
            raise ValueError("maintenance margin rate must be between zero and one")
        if not ZERO <= self.liquidation_fee_rate <= Decimal("1"):
            raise ValueError("liquidation fee rate must be between zero and one")
        if not ZERO <= self.liquidation_buffer_warning_ratio <= Decimal("1"):
            raise ValueError("liquidation warning ratio must be between zero and one")
        if self.long.side is not PositionSide.LONG or self.short.side is not PositionSide.SHORT:
            raise ValueError("mutable wallet legs have invalid sides")
        self.balance = self.initial_balance
        self.equity_peak = self.initial_balance

    def leg(self, side: PositionSide) -> MutableLeg:
        return self.long if side is PositionSide.LONG else self.short

    def accept_order(
        self,
        order_id: str,
        intent: OrderIntent,
        accepted_at: datetime | None = None,
    ) -> None:
        if self.liquidated:
            raise ValueError("liquidated wallet cannot accept new orders")
        if not order_id:
            raise ValueError("order id cannot be empty")
        existing = self.active_orders.get(order_id)
        if existing is not None:
            if existing == (intent, intent.quantity):
                return
            raise ValueError(f"order id already exists with different content: {order_id}")
        self.active_orders[order_id] = (intent, intent.quantity)
        if accepted_at is not None:
            self.order_accepted_times[order_id] = utc_aware(accepted_at)

    def order_accepted_at(self, order_id: str) -> datetime | None:
        return self.order_accepted_times.get(order_id)

    def cancel_order(self, order_id: str) -> None:
        self.active_orders.pop(order_id, None)
        self.order_accepted_times.pop(order_id, None)

    def remaining(self, order_id: str) -> Decimal:
        item = self.active_orders.get(order_id)
        return item[1] if item else ZERO

    def apply_fill(self, fill: FillEvent) -> bool:
        """Apply one validated fill. Duplicate event ids are idempotent no-ops."""
        if fill.event_id in self.processed_fill_ids:
            return False
        if self.liquidated:
            raise ValueError("cannot apply normal fill after liquidation")
        item = self.active_orders.get(fill.order_id)
        if item is None:
            raise ValueError(f"fill references unknown active order: {fill.order_id}")
        intent, remaining = item
        if (
            fill.intent_id != intent.intent_id
            or fill.symbol != intent.symbol
            or fill.position_side is not intent.position_side
            or fill.reduce_only != intent.reduce_only
            or fill.bucket is not intent.bucket
            or fill.action is not intent.action
            or fill.layer != intent.layer
            or fill.tactical_lot_id != intent.tactical_lot_id
        ):
            raise ValueError("fill does not match its active order intent")
        if fill.quantity > remaining:
            raise ValueError("fill quantity exceeds active order remaining quantity")

        leg = self.leg(fill.position_side)
        if fill.reduce_only and fill.quantity > leg.quantity:
            raise ValueError("reduce-only fill exceeds open position quantity")

        remaining_after = remaining - fill.quantity
        if remaining_after == ZERO:
            self.active_orders.pop(fill.order_id, None)
            self.order_accepted_times.pop(fill.order_id, None)
        else:
            self.active_orders[fill.order_id] = (intent, remaining_after)

        tactical_fee = ZERO
        realized = ZERO
        if fill.reduce_only:
            realized, _, tactical_reduced_qty = leg.reduce(
                fill.quantity,
                fill.price,
                fill.bucket,
                fee=fill.fee,
                tactical_lot_id=fill.tactical_lot_id,
            )
            self.balance += realized
            tactical_fee = fill.fee * tactical_reduced_qty / fill.quantity
        elif fill.action in {IntentAction.OPEN, IntentAction.INCREASE}:
            if (
                fill.bucket is PositionBucket.CORE
                and fill.position_side not in self.core_cost_basis_initial
            ):
                self.core_cost_basis_initial[fill.position_side] = fill.price
            leg.increase(
                fill.quantity,
                fill.price,
                fill.bucket,
                tactical_lot_id=fill.tactical_lot_id,
                opened_at=fill.timestamp,
                layer=fill.layer,
                fee=fill.fee if fill.bucket is PositionBucket.TACTICAL else ZERO,
            )
        else:
            raise ValueError("non-reduce fill has an unsupported action")

        self.balance -= fill.fee
        self.fees_paid += fill.fee
        if fill.liquidity_role is LiquidityRole.MAKER:
            self.maker_fees_paid += fill.fee
            self.maker_fill_count += 1
        else:
            self.taker_fees_paid += fill.fee
            self.taker_fill_count += 1
        if fill.position_side is PositionSide.LONG:
            self.long_fees_paid += fill.fee
        else:
            self.short_fees_paid += fill.fee
        if fill.reduce_only:
            self.tactical_fees_paid += tactical_fee
        elif fill.bucket is PositionBucket.TACTICAL:
            self.tactical_fees_paid += fill.fee
        self.realized_by_fill[fill.event_id] = realized
        self.processed_fill_ids.add(fill.event_id)
        return True

    def apply_funding(self, long_amount: Decimal, short_amount: Decimal) -> Decimal:
        if not long_amount.is_finite() or not short_amount.is_finite():
            raise ValueError("funding amounts must be finite")
        total = long_amount + short_amount
        self.balance += total
        self.funding_paid += total
        self.long_funding += long_amount
        self.short_funding += short_amount
        self.long.allocate_funding(long_amount)
        self.short.allocate_funding(short_amount)
        if self.long.quantity > ZERO:
            self.tactical_funding += long_amount * (
                self.long.tactical_quantity / self.long.quantity
            )
        if self.short.quantity > ZERO:
            self.tactical_funding += short_amount * (
                self.short.tactical_quantity / self.short.quantity
            )
        return total

    def unrealized(self, mark: Decimal) -> Decimal:
        return (
            self.long.immutable().unrealized_pnl(mark)
            + self.short.immutable().unrealized_pnl(mark)
        )

    def equity(self, mark: Decimal) -> Decimal:
        return self.balance + self.unrealized(mark)

    def gross_notional(self, mark: Decimal) -> Decimal:
        return (self.long.quantity + self.short.quantity) * mark

    def net_notional(self, mark: Decimal) -> Decimal:
        return (self.long.quantity - self.short.quantity) * mark

    def active_order_margin(self) -> Decimal:
        return sum(
            (
                remaining * intent.price / self.leverage
                for intent, remaining in self.active_orders.values()
                if not intent.reduce_only
            ),
            ZERO,
        )

    def maintenance_margin(self, mark: Decimal) -> Decimal:
        return self.gross_notional(mark) * self.maintenance_margin_rate

    def liquidation_fee_reserve(self, mark: Decimal) -> Decimal:
        return self.gross_notional(mark) * self.liquidation_fee_rate

    def liquidation_buffer(self, mark: Decimal) -> Decimal:
        return (
            self.equity(mark)
            - self.maintenance_margin(mark)
            - self.liquidation_fee_reserve(mark)
        )

    def liquidation_buffer_ratio(self, mark: Decimal) -> Decimal:
        if self.liquidated:
            return ZERO
        gross = self.gross_notional(mark)
        return self.liquidation_buffer(mark) / gross if gross > ZERO else Decimal("1")

    def liquidation_warning(self, mark: Decimal) -> bool:
        return (
            self.gross_notional(mark) > ZERO
            and self.liquidation_buffer_ratio(mark)
            <= self.liquidation_buffer_warning_ratio
        )

    def should_liquidate(self, mark: Decimal) -> bool:
        return (
            not self.liquidated
            and self.gross_notional(mark) > ZERO
            and self.liquidation_buffer(mark) <= ZERO
        )

    def create_liquidation_event(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        price: Decimal,
        ordinal: int,
    ) -> LiquidationEvent:
        long_pnl = (price - self.long.average_price) * self.long.quantity
        short_pnl = (self.short.average_price - price) * self.short.quantity
        gross = self.gross_notional(price)
        raw = (
            f"{symbol}|{timestamp.isoformat()}|{price}|{ordinal}|"
            f"{self.long.quantity}|{self.short.quantity}"
        ).encode()
        return LiquidationEvent(
            event_id="liq-" + sha256(raw).hexdigest()[:24],
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            long_quantity=self.long.quantity,
            short_quantity=self.short.quantity,
            realized_pnl=long_pnl + short_pnl,
            fee=gross * self.liquidation_fee_rate,
            equity_before=self.equity(price),
            maintenance_margin=self.maintenance_margin(price),
        )

    def apply_liquidation(self, event: LiquidationEvent) -> bool:
        if event.event_id in self.processed_liquidation_ids:
            return False
        if event.long_quantity != self.long.quantity or event.short_quantity != self.short.quantity:
            raise ValueError("liquidation event quantities do not match wallet")
        total_qty = event.long_quantity + event.short_quantity
        long_fee = (
            event.fee * event.long_quantity / total_qty if total_qty > ZERO else ZERO
        )
        short_fee = event.fee - long_fee
        tactical_liquidation_fee = sum(
            (
                fee * leg.tactical_quantity / qty
                if qty > ZERO and leg.tactical_quantity > ZERO
                else ZERO
                for leg, qty, fee in (
                    (self.long, event.long_quantity, long_fee),
                    (self.short, event.short_quantity, short_fee),
                )
            ),
            ZERO,
        )
        realized = ZERO
        for leg, qty, fee in (
            (self.long, event.long_quantity, long_fee),
            (self.short, event.short_quantity, short_fee),
        ):
            if qty <= ZERO:
                continue
            bucket = (
                PositionBucket.TACTICAL
                if leg.tactical_quantity > ZERO
                else PositionBucket.CORE
            )
            pnl, _, _ = leg.reduce(
                qty,
                event.price,
                bucket,
                fee=fee,
                tactical_lot_id=None,
            )
            realized += pnl
        if realized != event.realized_pnl:
            raise ValueError("liquidation realized PnL does not match wallet")
        self.balance += realized - event.fee
        self.fees_paid += event.fee
        self.taker_fees_paid += event.fee
        self.taker_fill_count += int(event.long_quantity > ZERO) + int(event.short_quantity > ZERO)
        self.long_fees_paid += long_fee
        self.short_fees_paid += short_fee
        self.tactical_fees_paid += tactical_liquidation_fee
        self.active_orders.clear()
        self.order_accepted_times.clear()
        self.liquidated = True
        self.liquidation_count += 1
        self.processed_liquidation_ids.add(event.event_id)
        return True

    def margin_ratio(self, mark: Decimal) -> Decimal:
        maintenance = self.maintenance_margin(mark)
        return self.equity(mark) / maintenance if maintenance > ZERO else ZERO

    def available_balance(self, mark: Decimal) -> Decimal:
        position_margin = self.gross_notional(mark) / self.leverage
        return self.equity(mark) - position_margin - self.active_order_margin()

    def planner_snapshot(self, mark: Decimal, timestamp: datetime) -> WalletSnapshot:
        created_at = utc_aware(timestamp)
        active = tuple(
            ActiveOrder(
                order_id=order_id,
                symbol=intent.symbol,
                position_side=intent.position_side,
                order_side=intent.order_side,
                quantity=remaining,
                price=intent.price,
                reduce_only=intent.reduce_only,
                bucket=intent.bucket,
                action=intent.action,
                created_at=self.order_accepted_at(order_id) or created_at,
                client_order_id=intent.intent_id,
                order_type=intent.order_type,
                time_in_force=intent.time_in_force,
                layer=intent.layer,
                tactical_lot_id=intent.tactical_lot_id,
            )
            for order_id, (intent, remaining) in sorted(self.active_orders.items())
        )
        return WalletSnapshot(
            balance=self.balance,
            equity=self.equity(mark),
            available_balance=self.available_balance(mark),
            long=self.long.immutable(),
            short=self.short.immutable(),
            active_orders=active,
            leverage=self.leverage,
        )

    def observe_risk(self, mark: Decimal) -> None:
        equity = self.equity(mark)
        gross = self.gross_notional(mark)
        self.gross_peak = max(self.gross_peak, gross)
        self.equity_peak = max(self.equity_peak, equity)
        if self.equity_peak > ZERO:
            self.max_drawdown = max(
                self.max_drawdown,
                (self.equity_peak - equity) / self.equity_peak,
            )

    def merge_risk_metrics(
        self,
        *,
        gross_peak: Decimal,
        equity_peak: Decimal,
        max_drawdown: Decimal,
    ) -> None:
        self.gross_peak = max(self.gross_peak, gross_peak)
        self.equity_peak = max(self.equity_peak, equity_peak)
        self.max_drawdown = max(self.max_drawdown, max_drawdown)

    def observe_snapshot_state(self, timestamp: datetime, mark: Decimal) -> datetime:
        """Update time/risk facts normally observed when materializing a snapshot."""
        timestamp = utc_aware(timestamp)
        if self.last_timestamp is not None and self.dual_leg_active:
            elapsed = Decimal(str((timestamp - self.last_timestamp).total_seconds()))
            if elapsed > ZERO:
                self.hedge_duration_seconds += elapsed
        self.dual_leg_active = self.long.quantity > ZERO and self.short.quantity > ZERO
        self.last_timestamp = timestamp
        self.observe_risk(mark)
        return timestamp

    def snapshot(
        self,
        timestamp: datetime,
        mark: Decimal,
        *,
        update_metrics: bool = True,
    ) -> SimulationSnapshot:
        timestamp = (
            self.observe_snapshot_state(timestamp, mark)
            if update_metrics
            else utc_aware(timestamp)
        )
        return SimulationSnapshot(
            timestamp=timestamp,
            balance=self.balance,
            equity=self.equity(mark),
            long_quantity=self.long.quantity,
            long_average_price=self.long.average_price,
            short_quantity=self.short.quantity,
            short_average_price=self.short.average_price,
            gross_notional=self.gross_notional(mark),
            net_notional=self.net_notional(mark),
            fees=self.fees_paid,
            funding=self.funding_paid,
            realized_pnl=self.long.realized_pnl + self.short.realized_pnl,
            available_balance=self.available_balance(mark),
            active_order_margin=self.active_order_margin(),
            maintenance_margin=self.maintenance_margin(mark),
            margin_ratio=self.margin_ratio(mark),
            long_realized_pnl=self.long.realized_pnl,
            short_realized_pnl=self.short.realized_pnl,
            liquidation_buffer=self.liquidation_buffer(mark),
            liquidation_buffer_ratio=self.liquidation_buffer_ratio(mark),
            liquidated=self.liquidated,
        )
