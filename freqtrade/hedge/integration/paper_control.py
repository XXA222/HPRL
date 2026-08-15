"""Control-plane account projection for the durable Paper runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
from freqtrade.hedge.symbols import raw_symbol


@dataclass(frozen=True, slots=True)
class PaperControlPosition:
    symbol: str
    position_side: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class PaperControlAccountView:
    account_id: str
    positions: tuple[PaperControlPosition, ...]
    observed_at: datetime
    source: str = "PAPER"


class PaperAccountViewProvider:
    """Expose the Paper wallet using the read-only control-plane view contract."""

    def __init__(self, application: "IntegratedPaperHedgeApplication") -> None:
        required = ("account_id", "symbol", "wallet", "last_market")
        if any(not hasattr(application, name) for name in required):
            raise TypeError("application does not provide the Paper account-view contract")
        self.application: Any = application

    def __call__(self) -> PaperControlAccountView:
        market = self.application.last_market
        wallet = self.application.wallet(market)
        mark = (
            market.mark
            if market is not None
            else max(
                wallet.long.average_price,
                wallet.short.average_price,
                Decimal("1"),
            )
        )
        symbol = raw_symbol(self.application.symbol)
        positions: list[PaperControlPosition] = []
        for side_name, leg in (("LONG", wallet.long), ("SHORT", wallet.short)):
            if leg.quantity <= 0:
                continue
            positions.append(
                PaperControlPosition(
                    symbol=symbol,
                    position_side=side_name,
                    quantity=leg.quantity,
                    entry_price=leg.average_price,
                    mark_price=mark,
                    unrealized_pnl=leg.unrealized_pnl(mark),
                )
            )
        return PaperControlAccountView(
            account_id=self.application.account_id,
            positions=tuple(positions),
            observed_at=(market.timestamp if market is not None else datetime.now(UTC)),
        )
