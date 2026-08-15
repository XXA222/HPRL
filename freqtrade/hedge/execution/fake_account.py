"""Position-aware fake account for deterministic Hedge Mode execution tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from threading import RLock

from .fake_exchange import FakeExchangeExecutionPort
from .service import IntentAction, PositionSide


@dataclass(frozen=True, slots=True)
class FakeLegPosition:
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")


class FakeHedgeAccount:
    def __init__(self, *, fee_rate: Decimal = Decimal("0")) -> None:
        if not fee_rate.is_finite() or fee_rate < 0:
            raise ValueError("fee_rate must be finite and non-negative")
        self._fee_rate = fee_rate
        self._legs: dict[tuple[str, str, PositionSide], FakeLegPosition] = {}
        self._applied_trades: set[str] = set()
        self._lock = RLock()

    def snapshot(self) -> tuple[dict[str, str], ...]:
        """Return deterministic confirmed leg state for crash-safe Paper recovery."""

        with self._lock:
            rows = []
            for (account_id, symbol, position_side), leg in sorted(
                self._legs.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][2].value),
            ):
                rows.append(
                    {
                        "account_id": account_id,
                        "symbol": symbol,
                        "position_side": position_side.value,
                        "quantity": str(leg.quantity),
                        "average_price": str(leg.average_price),
                        "realized_pnl": str(leg.realized_pnl),
                        "fees": str(leg.fees),
                    }
                )
            return tuple(rows)

    def restore(self, rows: object) -> None:
        """Replace confirmed state from a validated local Paper snapshot."""

        if isinstance(rows, (str, bytes)):
            raise ValueError("paper account rows must be an iterable of mappings")
        try:
            items = tuple(rows)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("paper account rows must be iterable") from exc
        restored: dict[tuple[str, str, PositionSide], FakeLegPosition] = {}
        for raw in items:
            if not isinstance(raw, dict):
                raise ValueError("paper account row must be a mapping")
            account_id = str(raw.get("account_id", "")).strip()
            symbol = str(raw.get("symbol", "")).strip()
            side = PositionSide(str(raw.get("position_side", "")).upper())
            if not account_id or not symbol:
                raise ValueError("paper account identity must not be empty")
            values = {
                name: Decimal(str(raw.get(name, "0")))
                for name in ("quantity", "average_price", "realized_pnl", "fees")
            }
            if any(not value.is_finite() for value in values.values()):
                raise ValueError("paper account values must be finite")
            quantity = values["quantity"]
            average = values["average_price"]
            fees = values["fees"]
            if quantity < 0 or average < 0 or fees < 0:
                raise ValueError("paper account quantities, prices and fees cannot be negative")
            if quantity == 0 and average != 0:
                raise ValueError("flat recovered leg must use zero average price")
            if quantity > 0 and average <= 0:
                raise ValueError("open recovered leg requires positive average price")
            key = (account_id, symbol, side)
            if key in restored:
                raise ValueError("duplicate paper account leg")
            restored[key] = FakeLegPosition(
                quantity=quantity,
                average_price=average,
                realized_pnl=values["realized_pnl"],
                fees=fees,
            )
        with self._lock:
            self._legs = restored
            # Pending exchange trade identities are intentionally not recovered.
            self._applied_trades.clear()

    def seed(
        self,
        *,
        account_id: str,
        symbol: str,
        position_side: PositionSide,
        quantity: Decimal,
        average_price: Decimal,
    ) -> None:
        if quantity < 0 or average_price < 0:
            raise ValueError("seed position values must not be negative")
        if quantity == 0 and average_price != 0:
            raise ValueError("flat seed must use zero average price")
        if quantity > 0 and average_price <= 0:
            raise ValueError("open seed requires positive average price")
        with self._lock:
            self._legs[(account_id, symbol, position_side)] = FakeLegPosition(
                quantity=quantity,
                average_price=average_price,
            )

    def leg(self, *, account_id: str, symbol: str, position_side: PositionSide) -> FakeLegPosition:
        with self._lock:
            return self._legs.get((account_id, symbol, position_side), FakeLegPosition())

    def apply_fill(
        self,
        *,
        trade_id: str,
        account_id: str,
        symbol: str,
        position_side: PositionSide,
        action: IntentAction,
        quantity: Decimal,
        price: Decimal,
        fee_amount: Decimal | None = None,
    ) -> FakeLegPosition:
        if quantity <= 0 or price <= 0:
            raise ValueError("fill quantity and price must be positive")
        key = (account_id, symbol, position_side)
        with self._lock:
            if trade_id in self._applied_trades:
                return self._legs.get(key, FakeLegPosition())
            current = self._legs.get(key, FakeLegPosition())
            if fee_amount is None:
                fee = quantity * price * self._fee_rate
            else:
                fee = Decimal(fee_amount)
                if not fee.is_finite() or fee < 0:
                    raise ValueError("fee_amount must be finite and non-negative")
            if action in {IntentAction.OPEN, IntentAction.INCREASE}:
                next_quantity = current.quantity + quantity
                average = (
                    current.quantity * current.average_price + quantity * price
                ) / next_quantity
                realized = current.realized_pnl
            else:
                if quantity > current.quantity:
                    raise ValueError("fake reduce fill would reverse the leg")
                next_quantity = current.quantity - quantity
                if position_side is PositionSide.LONG:
                    realized = current.realized_pnl + (price - current.average_price) * quantity
                else:
                    realized = current.realized_pnl + (current.average_price - price) * quantity
                average = Decimal("0") if next_quantity == 0 else current.average_price
            updated = FakeLegPosition(
                quantity=next_quantity,
                average_price=average,
                realized_pnl=realized,
                fees=current.fees + fee,
            )
            self._legs[key] = updated
            self._applied_trades.add(trade_id)
            return updated


class PositionAwareFakeExchange(FakeExchangeExecutionPort):
    def __init__(self, account: FakeHedgeAccount | None = None) -> None:
        super().__init__()
        self.account = account or FakeHedgeAccount()

    def fill_order(
        self,
        client_order_id: str,
        *,
        quantity: Decimal | str,
        price: Decimal | str,
        exchange_trade_id: str | None = None,
        fee: Decimal | str | None = None,
        fee_currency: str | None = "USDT",
    ):
        approved = self._require_approved(client_order_id)
        fill_qty = Decimal(quantity)
        fill_price = Decimal(price)
        intent = approved.intent
        if intent.action.reduces_risk:
            leg = self.account.leg(
                account_id=intent.account_id,
                symbol=intent.symbol,
                position_side=intent.position_side,
            )
            if fill_qty > leg.quantity:
                raise ValueError("fake reduce fill exceeds confirmed leg quantity")
        snapshot = super().fill_order(
            client_order_id,
            quantity=fill_qty,
            price=fill_price,
            exchange_trade_id=exchange_trade_id,
            fee=fee,
            fee_currency=fee_currency,
        )
        trade_id = snapshot.exchange_trade_id
        if trade_id is None:
            raise RuntimeError("fake fill did not produce a trade id")
        self.account.apply_fill(
            trade_id=trade_id,
            account_id=intent.account_id,
            symbol=intent.symbol,
            position_side=intent.position_side,
            action=intent.action,
            quantity=fill_qty,
            price=fill_price,
            fee_amount=None if fee is None else Decimal(fee),
        )
        return snapshot
