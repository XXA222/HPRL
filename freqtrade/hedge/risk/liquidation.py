"""Conservative liquidation-distance calculations for Cross hedge accounts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.numeric import require_nonnegative, require_positive, require_unit_interval


@dataclass(frozen=True, slots=True)
class LegLiquidationBuffer:
    position_side: PositionSide
    mark_price: Decimal
    liquidation_price: Decimal
    buffer_ratio: Decimal


def calculate_leg_liquidation_buffer(
    *,
    position_side: PositionSide | str,
    mark_price: Decimal,
    liquidation_price: Decimal,
) -> LegLiquidationBuffer:
    """Return the adverse-price distance to liquidation.

    A zero liquidation price is not interpreted as a perfect buffer. Venue APIs
    often use zero when the value is unavailable; callers must mark that fact as
    incomplete instead of silently treating it as safe.
    """

    side = (
        position_side
        if isinstance(position_side, PositionSide)
        else PositionSide(str(position_side).upper())
    )
    if side is PositionSide.BOTH:
        raise ValueError("Liquidation buffer requires LONG or SHORT side.")
    mark = require_positive(mark_price, field="mark_price")
    liquidation = require_positive(liquidation_price, field="liquidation_price")
    if side is PositionSide.LONG:
        ratio = max((mark - liquidation) / mark, Decimal("0"))
    else:
        ratio = max((liquidation - mark) / mark, Decimal("0"))
    ratio = min(ratio, Decimal("1"))
    return LegLiquidationBuffer(
        side,
        mark,
        liquidation,
        require_unit_interval(ratio, field="buffer_ratio"),
    )


def calculate_account_maintenance_buffer(
    *,
    equity: Decimal,
    maintenance_margin: Decimal,
) -> Decimal:
    equity_value = require_positive(equity, field="equity")
    maintenance = require_nonnegative(maintenance_margin, field="maintenance_margin")
    return min(max((equity_value - maintenance) / equity_value, Decimal("0")), Decimal("1"))


def calculate_projected_maintenance_buffer(
    *,
    equity: Decimal,
    maintenance_margin: Decimal,
    pending_maintenance_margin: Decimal,
    additional_notional: Decimal,
    maintenance_margin_rate: Decimal,
) -> Decimal:
    """Conservative post-fill account buffer used by the risk engine."""

    equity_value = require_positive(equity, field="equity")
    current = require_nonnegative(maintenance_margin, field="maintenance_margin")
    pending = require_nonnegative(
        pending_maintenance_margin,
        field="pending_maintenance_margin",
    )
    notional = require_nonnegative(additional_notional, field="additional_notional")
    rate = require_unit_interval(maintenance_margin_rate, field="maintenance_margin_rate")
    if rate <= 0:
        raise ValueError("maintenance_margin_rate must be positive.")
    projected = current + pending + notional * rate
    return min(max((equity_value - projected) / equity_value, Decimal("0")), Decimal("1"))


def minimum_liquidation_buffer(
    leg_buffers: Iterable[LegLiquidationBuffer],
    *,
    account_maintenance_buffer: Decimal,
) -> Decimal:
    account_buffer = require_unit_interval(
        account_maintenance_buffer,
        field="account_maintenance_buffer",
    )
    values = [account_buffer]
    values.extend(item.buffer_ratio for item in leg_buffers)
    return min(values)
