"""Inject native per-leg exit decisions into a Hedge planning result.

The overlay is intentionally downstream of the pure planner and upstream of order
admission.  This preserves one authoritative execution path while allowing Freqtrade
ROI/stoploss/trailing semantics to be translated into side- and bucket-aware reduce
orders.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

from freqtrade.hedge.planning.context import (
    IntentAction,
    OrderIntent,
    OrderSide,
    PositionBucket,
    PositionSide,
)

from .exits import HedgeExitPolicy, HedgeExitPolicyEngine
from .models import HedgeBucket, HedgeSide, LegSnapshot, ONE, ZERO, utc_datetime


def _decimal(value: object, default: str) -> Decimal:
    if value is None:
        value = default
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("exit policy values must be finite")
    return result


def _roi_mapping(raw: object) -> dict[int, Decimal]:
    if not isinstance(raw, Mapping):
        return {0: Decimal("0.03")}
    result: dict[int, Decimal] = {}
    for key, value in raw.items():
        minute = int(str(key))
        if minute < 0:
            raise ValueError("minimal_roi keys must be nonnegative minutes")
        ratio = _decimal(value, "0")
        if ratio < ZERO:
            raise ValueError("minimal_roi values cannot be negative")
        result[minute] = ratio
    return result or {0: Decimal("0.03")}


def policies_from_config(config: Mapping[str, Any]) -> dict[tuple[HedgeSide, HedgeBucket], HedgeExitPolicy]:
    """Translate official defaults and optional Hedge overrides into four policies."""

    hedge_raw = config.get("hedge", {})
    hedge = dict(hedge_raw) if isinstance(hedge_raw, Mapping) else {}
    native_raw = hedge.get("native_convergence", {})
    native = dict(native_raw) if isinstance(native_raw, Mapping) else {}
    exit_raw = native.get("exits", {})
    exits = dict(exit_raw) if isinstance(exit_raw, Mapping) else {}

    default = {
        "stoploss": config.get("stoploss", "-0.10"),
        "minimal_roi": config.get("minimal_roi", {"0": "0.03"}),
        "trailing_enabled": config.get("trailing_stop", False),
        "trailing_positive": config.get("trailing_stop_positive", "0.01"),
        "trailing_offset": config.get("trailing_stop_positive_offset", "0.02"),
        "trailing_only_offset_reached": config.get(
            "trailing_only_offset_is_reached", True
        ),
        "exit_fraction": exits.get("default_exit_fraction", "1"),
    }

    result: dict[tuple[HedgeSide, HedgeBucket], HedgeExitPolicy] = {}
    for side in HedgeSide:
        side_raw = exits.get(side.value.lower(), {})
        side_values = dict(side_raw) if isinstance(side_raw, Mapping) else {}
        for bucket in HedgeBucket:
            bucket_raw = side_values.get(bucket.value.lower(), {})
            bucket_values = dict(bucket_raw) if isinstance(bucket_raw, Mapping) else {}
            values = {**default, **side_values, **bucket_values}
            # Remove nested bucket mappings which may have arrived from side_values.
            values.pop("core", None)
            values.pop("tactical", None)
            result[(side, bucket)] = HedgeExitPolicy(
                stoploss=_decimal(values.get("stoploss"), "-0.10"),
                minimal_roi=_roi_mapping(values.get("minimal_roi")),
                trailing_enabled=bool(values.get("trailing_enabled", False)),
                trailing_positive=_decimal(values.get("trailing_positive"), "0.01"),
                trailing_offset=_decimal(values.get("trailing_offset"), "0.02"),
                trailing_only_offset_reached=bool(
                    values.get("trailing_only_offset_reached", True)
                ),
                exit_fraction=_decimal(values.get("exit_fraction"), "1"),
            )
    return result


class NativeExitOverlay:
    """Append deterministic reduce intents selected by the native exit engine."""

    def __init__(self, engine: HedgeExitPolicyEngine) -> None:
        self.engine = engine

    @staticmethod
    def _leg_snapshots(app: Any, market: Any) -> tuple[LegSnapshot, ...]:
        rows: list[LegSnapshot] = []
        for pside, hside in (
            (PositionSide.LONG, HedgeSide.LONG),
            (PositionSide.SHORT, HedgeSide.SHORT),
        ):
            bucket = app._bucket[pside]
            for pbucket, hbucket, quantity, average, opened_at in (
                (
                    PositionBucket.CORE,
                    HedgeBucket.CORE,
                    bucket.core_quantity,
                    bucket.core_average,
                    bucket.core_opened_at,
                ),
                (
                    PositionBucket.TACTICAL,
                    HedgeBucket.TACTICAL,
                    bucket.tactical_quantity,
                    bucket.tactical_average,
                    bucket.tactical_opened_at,
                ),
            ):
                if quantity <= ZERO:
                    continue
                rows.append(
                    LegSnapshot(
                        pair=app.symbol,
                        side=hside,
                        bucket=hbucket,
                        quantity=quantity,
                        average_price=average,
                        mark_price=market.mark,
                        opened_at=opened_at,
                    )
                )
        return tuple(rows)

    def apply(self, planning: Any, *, app: Any, market: Any) -> tuple[Any, tuple[str, ...]]:
        exits: list[OrderIntent] = []
        diagnostics: list[str] = []
        existing_keys = {
            (item.position_side.value, item.bucket.value)
            for item in tuple(getattr(planning, "submit_orders", ()))
            if item.reduce_only
        }
        at = utc_datetime(getattr(market, "timestamp", None))
        for leg in self._leg_snapshots(app, market):
            decision = self.engine.evaluate(leg, at=at)
            if not decision.should_exit:
                continue
            key = (leg.side.value, leg.bucket.value)
            if key in existing_keys:
                diagnostics.append(
                    f"NATIVE_EXIT_SUPERSEDED_BY_PLANNER:{leg.side.value}:{leg.bucket.value}"
                )
                continue
            quantity = leg.quantity * decision.fraction
            quantity = (quantity // market.qty_step) * market.qty_step
            if quantity <= ZERO:
                diagnostics.append(
                    f"NATIVE_EXIT_BELOW_STEP:{leg.side.value}:{leg.bucket.value}"
                )
                continue
            pside = PositionSide(leg.side.value)
            pbucket = PositionBucket(leg.bucket.value)
            price = market.bid if pside is PositionSide.LONG else market.ask
            order_side = OrderSide.SELL if pside is PositionSide.LONG else OrderSide.BUY
            action = IntentAction.CLOSE if quantity >= leg.quantity else IntentAction.REDUCE
            exits.append(
                OrderIntent.deterministic(
                    symbol=app.symbol,
                    position_side=pside,
                    order_side=order_side,
                    action=action,
                    bucket=pbucket,
                    quantity=min(quantity, leg.quantity),
                    price=price,
                    reduce_only=True,
                    reason=f"NATIVE_EXIT:{decision.reason}",
                    epoch=at.isoformat(),
                )
            )
            diagnostics.append(
                f"NATIVE_EXIT:{leg.side.value}:{leg.bucket.value}:{decision.reason}"
            )
        if not exits:
            return planning, tuple(diagnostics)
        submit = tuple(getattr(planning, "submit_orders")) + tuple(exits)
        ideal = tuple(getattr(planning, "ideal_orders")) + tuple(exits)
        merged_diagnostics = tuple(getattr(planning, "diagnostics", ())) + tuple(diagnostics)
        return replace(
            planning,
            submit_orders=submit,
            ideal_orders=ideal,
            diagnostics=merged_diagnostics,
        ), tuple(diagnostics)
