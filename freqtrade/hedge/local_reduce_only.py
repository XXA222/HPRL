"""Local reduce-only quantity calculation for Binance Hedge Mode."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.numeric import require_nonnegative


@dataclass(frozen=True, slots=True)
class SafeReduceResult:
    requested_quantity: Decimal
    confirmed_quantity: Decimal
    pending_reduce_quantity: Decimal
    available_quantity: Decimal
    allowed_quantity: Decimal
    reason_code: str

    @property
    def available_to_reduce(self) -> Decimal:
        return self.available_quantity

    @property
    def clipped(self) -> bool:
        return self.allowed_quantity != self.requested_quantity


# Original P2-H2 public name retained for source compatibility.
ReduceOnlyDecision = SafeReduceResult


def calculate_safe_reduce(
    *,
    requested_quantity: Decimal | str | int | float,
    confirmed_quantity: Decimal | str | int | float,
    pending_reduce_quantity: Decimal | str | int | float = Decimal("0"),
) -> SafeReduceResult:
    requested = require_nonnegative(requested_quantity, field="requested_quantity")
    confirmed = require_nonnegative(confirmed_quantity, field="confirmed_quantity")
    pending = require_nonnegative(pending_reduce_quantity, field="pending_reduce_quantity")
    available = max(confirmed - pending, Decimal("0"))
    allowed = min(requested, available)
    if allowed == requested:
        reason = "FULLY_ALLOWED"
    elif allowed == 0:
        reason = "NO_CONFIRMED_QUANTITY_AVAILABLE"
    else:
        reason = "CLIPPED_TO_CONFIRMED_AVAILABLE"
    return SafeReduceResult(
        requested_quantity=requested,
        confirmed_quantity=confirmed,
        pending_reduce_quantity=pending,
        available_quantity=available,
        allowed_quantity=allowed,
        reason_code=reason,
    )
