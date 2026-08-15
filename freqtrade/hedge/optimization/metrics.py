"""Metric normalization and Hedge-specific derived performance measures."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from freqtrade.hedge.optimization.types import exact_decimal


ZERO = Decimal(0)
EPSILON = Decimal("1e-18")


def _get(metrics: Mapping[str, Decimal], name: str, default: Decimal = ZERO) -> Decimal:
    return metrics.get(name, default)


def normalize_report(report: Mapping[str, object]) -> dict[str, Decimal]:
    """Convert every numeric report value to finite Decimal and add derived metrics.

    Boolean and textual report fields are intentionally excluded.  The original
    report mapping is never modified.
    """

    normalized: dict[str, Decimal] = {}
    for key, value in report.items():
        if (
            isinstance(value, bool)
            or value is None
            or (isinstance(value, str) and not value.strip())
        ):
            continue
        try:
            normalized[str(key)] = exact_decimal(value, field_name=f"report.{key}")
        except ValueError:
            # Simulation reports contain a few textual status fields.  They are
            # not optimizer metrics and must not silently become zero.
            if isinstance(value, str):
                continue
            raise

    initial = _get(normalized, "initial_balance")
    final_equity = _get(normalized, "final_equity", initial + _get(normalized, "total_pnl"))
    total_pnl = _get(normalized, "total_pnl", final_equity - initial)
    normalized["net_profit"] = total_pnl
    normalized["net_return"] = (
        total_pnl / initial if abs(initial) > EPSILON else ZERO
    )
    normalized.setdefault("total_return_ratio", normalized["net_return"])

    fees = abs(_get(normalized, "fees"))
    funding = abs(_get(normalized, "funding"))
    normalized["fee_drag_ratio"] = fees / initial if initial > EPSILON else ZERO
    normalized["funding_drag_ratio"] = funding / initial if initial > EPSILON else ZERO
    normalized["cost_drag_ratio"] = (
        fees + funding
    ) / initial if initial > EPSILON else ZERO

    long_pnl = _get(normalized, "long_pnl")
    short_pnl = _get(normalized, "short_pnl")
    normalized["leg_pnl_imbalance"] = abs(long_pnl - short_pnl) / max(
        abs(long_pnl) + abs(short_pnl), EPSILON
    )
    long_qty = _get(normalized, "final_long_quantity")
    short_qty = _get(normalized, "final_short_quantity")
    normalized["final_quantity_imbalance"] = abs(long_qty - short_qty) / max(
        long_qty + short_qty, EPSILON
    )

    gross_peak = _get(normalized, "gross_peak")
    if "gross_peak_ratio" not in normalized:
        normalized["gross_peak_ratio"] = gross_peak / initial if initial > EPSILON else ZERO
    normalized["return_on_peak_gross"] = total_pnl / max(gross_peak, EPSILON)

    add_count = _get(normalized, "add_count")
    reduce_count = _get(normalized, "reduce_count")
    normalized["order_action_count"] = add_count + reduce_count
    normalized["profit_per_action"] = total_pnl / max(add_count + reduce_count, Decimal(1))

    max_drawdown = abs(_get(normalized, "max_drawdown"))
    normalized["max_drawdown"] = max_drawdown
    normalized["return_drawdown_ratio"] = normalized["net_return"] / max(
        max_drawdown, Decimal("0.00000001")
    )
    liquidated = report.get("liquidated", False)
    if not isinstance(liquidated, bool):
        raise TypeError("report.liquidated must be a boolean when supplied")
    normalized["liquidated"] = Decimal(1) if liquidated else ZERO
    return normalized


def require_metrics(metrics: Mapping[str, Decimal], names: tuple[str, ...]) -> None:
    missing = sorted(name for name in names if name not in metrics)
    if missing:
        raise ValueError(f"required optimizer metrics are missing: {', '.join(missing)}")
    nonfinite = sorted(name for name in names if not metrics[name].is_finite())
    if nonfinite:
        raise ValueError(f"optimizer metrics are non-finite: {', '.join(nonfinite)}")
