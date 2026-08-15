"""Build unit-safe account risk snapshots from positions and pending orders."""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from decimal import Decimal
from typing import Iterable

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.errors import HedgeConfigurationError
from freqtrade.hedge.numeric import require_nonnegative, require_positive
from freqtrade.hedge.risk.liquidation import (
    calculate_account_maintenance_buffer,
    calculate_leg_liquidation_buffer,
    minimum_liquidation_buffer,
)
from freqtrade.hedge.risk.models import AccountRiskSnapshot, PendingOrderRisk, RiskRequest
from freqtrade.hedge.symbols import canonicalize_symbol


def _account_id(value: str) -> str:
    if not isinstance(value, str):
        raise HedgeConfigurationError("account_id must be a string.")
    normalized = value.strip()
    if not normalized:
        raise HedgeConfigurationError("account_id must not be empty.")
    return normalized


def _side(value: PositionSide | str) -> PositionSide:
    side = value if isinstance(value, PositionSide) else PositionSide(str(value).upper())
    if side is PositionSide.BOTH:
        raise HedgeConfigurationError("Hedge position side must be LONG or SHORT.")
    return side


@dataclass(frozen=True, slots=True)
class PositionRiskLeg:
    """Confirmed position facts for one hedge leg.

    Quantity is base-asset/contracts quantity. Notional and margin values are
    quote-currency amounts. No exposure ratio is accepted here; ratios are
    derived from account equity by :func:`build_risk_portfolio`.
    """

    account_id: str
    symbol: str
    position_side: PositionSide
    quantity: Decimal
    mark_price: Decimal
    leverage: Decimal = Decimal("1")
    reported_initial_margin: Decimal | None = None
    maintenance_margin: Decimal = Decimal("0")
    liquidation_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _account_id(self.account_id))
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(self, "position_side", _side(self.position_side))
        object.__setattr__(
            self,
            "quantity",
            require_nonnegative(self.quantity, field="quantity"),
        )
        object.__setattr__(
            self,
            "mark_price",
            require_positive(self.mark_price, field="mark_price"),
        )
        leverage = require_positive(self.leverage, field="leverage")
        if leverage < 1:
            raise HedgeConfigurationError("leverage must be greater than or equal to 1.")
        object.__setattr__(self, "leverage", leverage)
        object.__setattr__(
            self,
            "maintenance_margin",
            require_nonnegative(self.maintenance_margin, field="maintenance_margin"),
        )
        if self.reported_initial_margin is not None:
            object.__setattr__(
                self,
                "reported_initial_margin",
                require_nonnegative(
                    self.reported_initial_margin,
                    field="reported_initial_margin",
                ),
            )
        if self.liquidation_price is not None:
            object.__setattr__(
                self,
                "liquidation_price",
                require_nonnegative(self.liquidation_price, field="liquidation_price"),
            )

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.mark_price

    @property
    def signed_notional(self) -> Decimal:
        sign = Decimal("1") if self.position_side is PositionSide.LONG else Decimal("-1")
        return sign * self.notional

    @property
    def initial_margin(self) -> Decimal:
        if self.reported_initial_margin is not None:
            return self.reported_initial_margin
        return self.notional / self.leverage


@dataclass(frozen=True, slots=True)
class RiskPortfolioSnapshot:
    """Account snapshot plus the facts needed to construct side-aware requests."""

    account: AccountRiskSnapshot
    positions: tuple[PositionRiskLeg, ...]
    pending_orders: tuple[PendingOrderRisk, ...]

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        pending_orders = tuple(self.pending_orders)
        if any(item.account_id != self.account.account_id for item in positions):
            raise HedgeConfigurationError("Every position must belong to the snapshot account.")
        if any(item.account_id != self.account.account_id for item in pending_orders):
            raise HedgeConfigurationError(
                "Every pending order must belong to the snapshot account."
            )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "pending_orders", pending_orders)

    def confirmed_quantity(
        self,
        symbol: str,
        position_side: PositionSide | str,
    ) -> Decimal:
        canonical = canonicalize_symbol(symbol)
        side = _side(position_side)
        return sum(
            (
                item.quantity
                for item in self.positions
                if item.symbol == canonical and item.position_side is side
            ),
            Decimal("0"),
        )

    def leg_notional(
        self,
        symbol: str,
        position_side: PositionSide | str,
    ) -> Decimal:
        canonical = canonicalize_symbol(symbol)
        side = _side(position_side)
        return sum(
            (
                item.notional
                for item in self.positions
                if item.symbol == canonical and item.position_side is side
            ),
            Decimal("0"),
        )

    def symbol_gross_notional(self, symbol: str) -> Decimal:
        canonical = canonicalize_symbol(symbol)
        return sum(
            (item.notional for item in self.positions if item.symbol == canonical),
            Decimal("0"),
        )

    def pending_reduce_quantity(
        self,
        symbol: str,
        position_side: PositionSide | str,
    ) -> Decimal:
        canonical = canonicalize_symbol(symbol)
        side = _side(position_side)
        return sum(
            (
                item.remaining_quantity
                for item in self.pending_orders
                if item.symbol == canonical
                and item.position_side is side
                and item.action.reduces_risk
            ),
            Decimal("0"),
        )

    def pending_increase_notional(
        self,
        symbol: str,
        position_side: PositionSide | str | None = None,
    ) -> Decimal:
        canonical = canonicalize_symbol(symbol)
        side = None if position_side is None else _side(position_side)
        return sum(
            (
                item.remaining_notional
                for item in self.pending_orders
                if item.symbol == canonical
                and item.increases_risk
                and (side is None or item.position_side is side)
            ),
            Decimal("0"),
        )

    def build_request(
        self,
        *,
        symbol: str,
        position_side: PositionSide | str,
        action: PositionAction | str,
        requested_quantity: Decimal,
        reference_price: Decimal,
        leverage: Decimal = Decimal("1"),
        exchange: str | None = None,
        intent_id: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        expires_at_ms: int | None = None,
        target_snapshot_version: int | None = None,
        maintenance_margin_rate: Decimal = Decimal("0.01"),
    ) -> RiskRequest:
        canonical = canonicalize_symbol(symbol)
        side = _side(position_side)
        normalized_action = (
            action if isinstance(action, PositionAction) else PositionAction(str(action).upper())
        )
        confirmed = None
        pending_reduce = Decimal("0")
        if normalized_action.reduces_risk:
            confirmed = self.confirmed_quantity(canonical, side)
            pending_reduce = self.pending_reduce_quantity(canonical, side)
        return RiskRequest(
            account_id=self.account.account_id,
            symbol=canonical,
            position_side=side,
            action=normalized_action,
            requested_quantity=requested_quantity,
            reference_price=reference_price,
            current_leg_notional=self.leg_notional(canonical, side),
            current_symbol_gross_notional=self.symbol_gross_notional(canonical),
            pending_leg_increase_notional=self.pending_increase_notional(canonical, side),
            pending_symbol_increase_notional=self.pending_increase_notional(canonical),
            confirmed_quantity=confirmed,
            pending_reduce_quantity=pending_reduce,
            leverage=leverage,
            exchange=self.account.exchange if exchange is None else exchange,
            intent_id=uuid.uuid4().hex if intent_id is None else intent_id,
            idempotency_key=(
                uuid.uuid4().hex if idempotency_key is None else idempotency_key
            ),
            correlation_id=(
                uuid.uuid4().hex if correlation_id is None else correlation_id
            ),
            expires_at_ms=expires_at_ms,
            target_snapshot_version=target_snapshot_version,
            maintenance_margin_rate=maintenance_margin_rate,
        )

    def as_dict(self) -> dict[str, object]:
        account = self.account
        return {
            "account_id": account.account_id,
            "equity": str(account.equity),
            "wallet_balance": str(account.wallet_balance),
            "available_balance": str(account.available_balance),
            "initial_margin": str(account.initial_margin),
            "maintenance_margin": str(account.maintenance_margin),
            "gross_long_notional": str(account.gross_long_notional),
            "gross_short_notional": str(account.gross_short_notional),
            "gross_total_notional": str(account.gross_total_notional),
            "net_notional": str(account.net_notional),
            "gross_exposure_ratio": str(account.gross_exposure_ratio),
            "net_exposure_ratio": str(account.net_exposure_ratio),
            "margin_utilization": str(account.margin_utilization),
            "liquidation_buffer_ratio": str(account.liquidation_buffer_ratio),
            "pending_order_notional": str(account.pending_order_notional),
            "position_count": len(self.positions),
            "pending_order_count": len(self.pending_orders),
            "risk_data_valid": account.risk_data_valid,
            "observed_at_ms": account.observed_at_ms,
        }


def build_risk_portfolio(
    *,
    account_id: str,
    equity: Decimal,
    wallet_balance: Decimal,
    available_balance: Decimal,
    positions: Iterable[PositionRiskLeg] = (),
    pending_orders: Iterable[PendingOrderRisk] = (),
    initial_margin: Decimal | None = None,
    maintenance_margin: Decimal | None = None,
    risk_data_valid: bool = True,
    observed_at_ms: int | None = None,
    exchange: str = "binance",
    snapshot_id: str | None = None,
    source_version: int = 0,
    exchange_time_ms: int | None = None,
    strict_completeness: bool = True,
) -> RiskPortfolioSnapshot:
    """Aggregate account, position and pending-order facts into one snapshot."""

    normalized_account = _account_id(account_id)
    position_items = tuple(positions)
    pending_items = tuple(pending_orders)
    if any(item.account_id != normalized_account for item in position_items):
        raise HedgeConfigurationError("Position account_id does not match account_id.")
    if any(item.account_id != normalized_account for item in pending_items):
        raise HedgeConfigurationError("Pending order account_id does not match account_id.")

    gross_long = sum(
        (item.notional for item in position_items if item.position_side is PositionSide.LONG),
        Decimal("0"),
    )
    gross_short = sum(
        (item.notional for item in position_items if item.position_side is PositionSide.SHORT),
        Decimal("0"),
    )
    initial = (
        sum((item.initial_margin for item in position_items), Decimal("0"))
        if initial_margin is None
        else require_nonnegative(initial_margin, field="initial_margin")
    )
    maintenance = (
        sum((item.maintenance_margin for item in position_items), Decimal("0"))
        if maintenance_margin is None
        else require_nonnegative(maintenance_margin, field="maintenance_margin")
    )

    risk_increasing_orders = tuple(item for item in pending_items if item.increases_risk)
    pending_notional = sum(
        (item.remaining_notional for item in risk_increasing_orders),
        Decimal("0"),
    )
    pending_initial_margin = sum(
        (item.remaining_initial_margin for item in risk_increasing_orders),
        Decimal("0"),
    )
    pending_maintenance_margin = sum(
        (item.remaining_maintenance_margin for item in risk_increasing_orders),
        Decimal("0"),
    )
    pending_net_delta = sum(
        (item.signed_net_notional_delta for item in risk_increasing_orders),
        Decimal("0"),
    )
    pending_long = sum(
        (
            item.remaining_notional
            for item in risk_increasing_orders
            if item.position_side is PositionSide.LONG
        ),
        Decimal("0"),
    )
    pending_short = sum(
        (
            item.remaining_notional
            for item in risk_increasing_orders
            if item.position_side is PositionSide.SHORT
        ),
        Decimal("0"),
    )

    equity_value = require_positive(equity, field="equity")
    completeness_errors: list[str] = []
    open_positions = tuple(item for item in position_items if item.quantity > 0)
    if any(
        item.liquidation_price is None or item.liquidation_price <= 0
        for item in open_positions
    ):
        completeness_errors.append("LIQUIDATION_DATA_INCOMPLETE")
    if any(item.maintenance_margin <= 0 for item in open_positions):
        completeness_errors.append("MAINTENANCE_MARGIN_INCOMPLETE")

    leg_buffers = []
    if "LIQUIDATION_DATA_INCOMPLETE" not in completeness_errors:
        leg_buffers = [
            calculate_leg_liquidation_buffer(
                position_side=item.position_side,
                mark_price=item.mark_price,
                liquidation_price=item.liquidation_price,
            )
            for item in open_positions
            if item.liquidation_price is not None
        ]
    account_buffer = calculate_account_maintenance_buffer(
        equity=equity_value,
        maintenance_margin=maintenance,
    )
    liquidation_buffer = (
        minimum_liquidation_buffer(
            leg_buffers,
            account_maintenance_buffer=account_buffer,
        )
        if not completeness_errors
        else Decimal("0")
    )
    effective_valid = risk_data_valid and (
        not strict_completeness or not completeness_errors
    )

    account = AccountRiskSnapshot(
        account_id=normalized_account,
        equity=equity_value,
        wallet_balance=wallet_balance,
        available_balance=available_balance,
        initial_margin=initial,
        maintenance_margin=maintenance,
        gross_long_notional=gross_long,
        gross_short_notional=gross_short,
        net_notional=gross_long - gross_short,
        pending_order_notional=pending_notional,
        pending_order_initial_margin=pending_initial_margin,
        pending_order_maintenance_margin=pending_maintenance_margin,
        pending_net_notional_delta=pending_net_delta,
        pending_long_notional=pending_long,
        pending_short_notional=pending_short,
        liquidation_buffer_ratio=liquidation_buffer,
        risk_data_valid=effective_valid,
        risk_data_errors=tuple(completeness_errors),
        liquidation_data_complete=(
            "LIQUIDATION_DATA_INCOMPLETE" not in completeness_errors
        ),
        maintenance_margin_complete=(
            "MAINTENANCE_MARGIN_INCOMPLETE" not in completeness_errors
        ),
        exchange=exchange,
        snapshot_id=uuid.uuid4().hex if snapshot_id is None else snapshot_id,
        source_version=source_version,
        exchange_time_ms=exchange_time_ms,
        observed_at_ms=observed_at_ms,
    )
    return RiskPortfolioSnapshot(account, position_items, pending_items)
