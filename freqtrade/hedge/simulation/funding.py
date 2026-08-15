from __future__ import annotations

from decimal import Decimal

from .cross_wallet import CrossWallet
from .exchange import FundingEvent


class FundingEngine:
    def apply(self, wallet: CrossWallet, event: FundingEvent) -> Decimal:
        long_payment = -(wallet.long.quantity * event.mark_price * event.rate)
        short_payment = wallet.short.quantity * event.mark_price * event.rate
        return wallet.apply_funding(long_payment, short_payment)
