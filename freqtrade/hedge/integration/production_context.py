"""Build planner input from the authoritative Binance read-only account view."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Protocol

from freqtrade.hedge.exchange.base import OrderFact, PositionFact, ReadonlyAccountView
from freqtrade.hedge.planning.context import (
    ActiveOrder,
    IntentAction,
    LegPosition,
    MarketSnapshot,
    OrderSide,
    OrderType,
    PlannerConfig,
    PlanningContext,
    PositionBucket,
    PositionSide,
    TimeInForce,
    WalletSnapshot,
)
from freqtrade.hedge.symbols import raw_symbol
from freqtrade.hedge.strategies.contract import (
    StrategyDirective,
    planner_config_for_directive,
    target_net_quantity_for_directive,
)

ZERO = Decimal("0")


class SignalLike(Protocol):
    symbol: str
    long_score: Decimal
    short_score: Decimal
    target_net: Decimal | None
    target_net_ratio: Decimal | None
    confidence: Decimal
    risk_scale: Decimal
    long_exposure_scale: Decimal
    short_exposure_scale: Decimal
    allow_new_risk: bool
    regime: str
    strategy_reason: str
    model_version: str


@dataclass(frozen=True, slots=True)
class PlanningContextEvidence:
    symbol: str
    account_revision: int
    long_quantity: Decimal
    short_quantity: Decimal
    active_order_count: int
    ignored_symbol_count: int
    position_classification: str = "EXCHANGE_POSITION_AS_PROTECTED_CORE"


@dataclass(frozen=True, slots=True)
class BuiltPlanningContext:
    context: PlanningContext
    evidence: PlanningContextEvidence


class ReadonlyPlanningContextBuilder:
    """Conservative bridge from exchange facts into the pure planner model.

    Existing exchange positions are classified as protected core because the
    exchange projection cannot prove which historical fills were tactical.
    This avoids accidental tactical reductions of pre-existing exposure.
    """

    def __init__(
        self,
        *,
        planner_config: PlannerConfig,
        allowed_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
        default_leverage: Decimal = Decimal("1"),
    ) -> None:
        self.planner_config = planner_config
        self.allowed_symbols = tuple(dict.fromkeys(raw_symbol(item) for item in allowed_symbols))
        self.default_leverage = Decimal(str(default_leverage))
        if self.default_leverage <= 0:
            raise ValueError("default_leverage must be positive")

    def build(
        self,
        *,
        account_view: ReadonlyAccountView,
        market: MarketSnapshot,
        signal: SignalLike,
    ) -> BuiltPlanningContext:
        symbol = raw_symbol(market.symbol)
        if symbol not in self.allowed_symbols:
            raise ValueError(f"symbol is outside the fixed perpetual allowlist: {symbol}")
        if raw_symbol(signal.symbol) != symbol:
            raise ValueError("signal symbol does not match market symbol")
        account = account_view.account_snapshot
        if account is None:
            raise RuntimeError("readonly account snapshot is unavailable")

        matching_positions = tuple(
            item for item in account_view.positions if raw_symbol(item.symbol) == symbol
        )
        long = self._leg(matching_positions, PositionSide.LONG)
        short = self._leg(matching_positions, PositionSide.SHORT)
        leverage = max(
            (Decimal(max(item.leverage, 1)) for item in matching_positions),
            default=self.default_leverage,
        )
        active_orders = tuple(
            self._active_order(item, market=market, long=long, short=short)
            for item in account_view.active_orders
            if raw_symbol(item.symbol) == symbol and item.active
        )
        active_orders = tuple(item for item in active_orders if item is not None)
        ignored = sum(
            1
            for item in (*account_view.positions, *account_view.active_orders)
            if raw_symbol(item.symbol) != symbol
        )
        balance = max(account.total_wallet_balance, ZERO)
        equity = account.total_margin_balance
        if equity <= ZERO:
            equity = balance + account.total_unrealized_pnl
        wallet = WalletSnapshot(
            balance=balance,
            equity=equity,
            available_balance=max(account.total_available_balance, ZERO),
            long=long,
            short=short,
            active_orders=active_orders,
            leverage=leverage,
        )
        target_net_ratio = getattr(signal, "target_net_ratio", None)
        target_net = getattr(signal, "target_net", None)
        directive = StrategyDirective(
            long_score=signal.long_score,
            short_score=signal.short_score,
            target_net_quantity=None if target_net_ratio is not None else target_net,
            target_net_ratio=target_net_ratio,
            confidence=getattr(signal, "confidence", Decimal("1")),
            risk_scale=getattr(signal, "risk_scale", Decimal("1")),
            long_exposure_scale=getattr(signal, "long_exposure_scale", Decimal("1")),
            short_exposure_scale=getattr(signal, "short_exposure_scale", Decimal("1")),
            allow_new_risk=bool(getattr(signal, "allow_new_risk", True)),
            regime=str(getattr(signal, "regime", "UNKNOWN")),
            reason=str(
                getattr(signal, "strategy_reason", getattr(signal, "reason", "LEGACY_SIGNAL"))
            ),
            model_version=str(getattr(signal, "model_version", "legacy")),
        )
        effective_config = planner_config_for_directive(self.planner_config, directive)
        effective_target = target_net_quantity_for_directive(
            directive=directive, base=self.planner_config, equity=wallet.equity, mark_price=market.mark
        )
        context = PlanningContext(
            market=market, wallet=wallet, config=effective_config,
            long_signal=directive.long_score, short_signal=directive.short_score,
            target_net_quantity=effective_target,
        )
        return BuiltPlanningContext(
            context=context,
            evidence=PlanningContextEvidence(
                symbol=symbol,
                account_revision=account_view.revision,
                long_quantity=long.quantity,
                short_quantity=short.quantity,
                active_order_count=len(active_orders),
                ignored_symbol_count=ignored,
            ),
        )

    @staticmethod
    def _leg(
        positions: tuple[PositionFact, ...],
        side: PositionSide,
    ) -> LegPosition:
        matching = [item for item in positions if str(item.position_side).upper() == side.value]
        if not matching:
            return LegPosition(side)
        quantity = sum((abs(item.quantity) for item in matching), ZERO)
        if quantity <= ZERO:
            return LegPosition(side)
        weighted = sum((abs(item.quantity) * item.entry_price for item in matching), ZERO)
        average = weighted / quantity if weighted > ZERO else ZERO
        realized = ZERO
        return LegPosition(
            side=side,
            quantity=quantity,
            average_price=average,
            core_quantity=quantity,
            core_average_price=average,
            tactical_quantity=ZERO,
            tactical_average_price=ZERO,
            realized_pnl=realized,
        )

    @staticmethod
    def _active_order(
        order: OrderFact,
        *,
        market: MarketSnapshot,
        long: LegPosition,
        short: LegPosition,
    ) -> ActiveOrder | None:
        remaining = max(order.original_quantity - order.cumulative_filled_quantity, ZERO)
        if remaining <= ZERO:
            return None
        try:
            position_side = PositionSide(str(order.position_side).upper())
            order_side = OrderSide(str(order.side).upper())
        except ValueError:
            return None
        raw: Mapping[str, object] = order.raw if isinstance(order.raw, Mapping) else {}
        raw_price = raw.get("price") or raw.get("p") or raw.get("stopPrice")
        price = order.average_price
        if price <= ZERO and raw_price is not None:
            try:
                price = Decimal(str(raw_price))
            except Exception:
                price = ZERO
        if price <= ZERO:
            price = market.mark
        reduce_only = bool(order.reduce_only)
        leg = long if position_side is PositionSide.LONG else short
        action = (
            IntentAction.REDUCE
            if reduce_only
            else IntentAction.OPEN
            if leg.quantity <= ZERO
            else IntentAction.INCREASE
        )
        order_type = (
            OrderType.MARKET
            if str(order.order_type).upper() == "MARKET"
            else OrderType.LIMIT
        )
        return ActiveOrder(
            order_id=str(order.exchange_order_id),
            client_order_id=str(order.client_order_id),
            symbol=market.symbol,
            position_side=position_side,
            order_side=order_side,
            quantity=remaining,
            price=price,
            reduce_only=reduce_only,
            bucket=PositionBucket.CORE,
            action=action,
            created_at=order.observed_at,
            order_type=order_type,
            time_in_force=TimeInForce.GTC,
        )
