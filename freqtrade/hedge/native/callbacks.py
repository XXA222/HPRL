"""Strategy callback compatibility for dual-leg Hedge state.

Native callbacks use explicit Hedge views.  Optional legacy-view mode calls selected
Freqtrade callbacks with a read-only trade-like object, never a persisted Trade row.
The adapter is fail-closed for confirmations and fail-safe for optional advisory hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Callable, Mapping

from .models import (
    AdmissionCode,
    AdmissionDecision,
    HedgeAction,
    HedgeBucket,
    HedgeSide,
    LegSnapshot,
    NativeOrderIntent,
    ONE,
    ZERO,
    finite_decimal,
    utc_datetime,
)


class CallbackCompatibilityMode(StrEnum):
    NATIVE_ONLY = "native_only"
    LEGACY_VIEW = "legacy_view"
    BOTH_CONSERVATIVE = "both_conservative"


@dataclass(frozen=True, slots=True)
class HedgeTradeView:
    pair: str
    position_side: HedgeSide
    bucket: HedgeBucket
    amount: Decimal
    open_rate: Decimal
    current_rate: Decimal
    open_date_utc: datetime | None
    is_short: bool
    stake_amount: Decimal
    leverage: Decimal
    realized_profit: Decimal
    funding_fees: Decimal
    fee_total: Decimal
    is_open: bool = True

    @classmethod
    def from_leg(cls, leg: LegSnapshot, *, leverage: object = ONE) -> "HedgeTradeView":
        leverage_value = finite_decimal(leverage, field_name="leverage")
        if leverage_value <= ZERO:
            raise ValueError("leverage must be positive")
        return cls(
            pair=leg.pair,
            position_side=leg.side,
            bucket=leg.bucket,
            amount=leg.quantity,
            open_rate=leg.average_price,
            current_rate=leg.mark_price,
            open_date_utc=leg.opened_at,
            is_short=leg.side is HedgeSide.SHORT,
            stake_amount=(leg.quantity * leg.average_price / leverage_value),
            leverage=leverage_value,
            realized_profit=leg.realized_pnl,
            funding_fees=leg.funding,
            fee_total=leg.fees,
            is_open=leg.quantity > ZERO,
        )

    def calc_profit_ratio(self, rate: float | Decimal) -> float:
        current = finite_decimal(rate, field_name="rate")
        if self.open_rate <= ZERO:
            return 0.0
        if self.is_short:
            return float(self.open_rate / current - ONE)
        return float(current / self.open_rate - ONE)


@dataclass(frozen=True, slots=True)
class HedgeOrderView:
    intent_id: str
    pair: str
    position_side: HedgeSide
    bucket: HedgeBucket
    action: HedgeAction
    amount: Decimal
    price: Decimal
    reduce_only: bool
    tag: str = ""

    @classmethod
    def from_intent(cls, intent: NativeOrderIntent) -> "HedgeOrderView":
        return cls(
            intent_id=intent.intent_id,
            pair=intent.pair,
            position_side=intent.side,
            bucket=intent.bucket,
            action=intent.action,
            amount=intent.quantity,
            price=intent.price,
            reduce_only=intent.reduce_only,
            tag=str(intent.metadata.get("tag", "")),
        )


@dataclass(frozen=True, slots=True)
class HedgeCallbackContext:
    current_time: datetime
    leg: LegSnapshot | None = None
    intent: NativeOrderIntent | None = None
    wallet_equity: Decimal = ZERO
    available_balance: Decimal = ZERO
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_time", utc_datetime(self.current_time))
        object.__setattr__(
            self,
            "wallet_equity",
            finite_decimal(self.wallet_equity, field_name="wallet_equity"),
        )
        object.__setattr__(
            self,
            "available_balance",
            finite_decimal(self.available_balance, field_name="available_balance"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class HedgeStrategyCallbackAdapter:
    """Call native Hedge callbacks and selected upstream callbacks conservatively."""

    def __init__(
        self,
        strategy: Any,
        *,
        mode: CallbackCompatibilityMode | str = CallbackCompatibilityMode.NATIVE_ONLY,
        fail_closed_confirmations: bool = True,
    ) -> None:
        self.strategy = strategy
        self.mode = CallbackCompatibilityMode(mode)
        self.fail_closed_confirmations = bool(fail_closed_confirmations)

    def _legacy_callback_available(self, name: str) -> bool:
        """Return true only when a strategy class overrides an upstream callback.

        Calling IStrategy defaults would silently force values such as leverage=1.0,
        so compatibility mode must distinguish inherited defaults from user logic.
        """

        callback = getattr(self.strategy, name, None)
        if callback is None or not callable(callback):
            return False
        owner = next(
            (cls for cls in type(self.strategy).__mro__ if name in cls.__dict__),
            None,
        )
        return owner is not None and owner.__name__ != "IStrategy"

    @property
    def legacy_enabled(self) -> bool:
        return self.mode in {
            CallbackCompatibilityMode.LEGACY_VIEW,
            CallbackCompatibilityMode.BOTH_CONSERVATIVE,
        }

    @property
    def native_enabled(self) -> bool:
        return self.mode in {
            CallbackCompatibilityMode.NATIVE_ONLY,
            CallbackCompatibilityMode.BOTH_CONSERVATIVE,
        }

    def _call_optional(
        self,
        name: str,
        *,
        kwargs: Mapping[str, Any],
        default: Any = None,
        fail_closed: bool = False,
    ) -> Any:
        callback = getattr(self.strategy, name, None)
        if callback is None or not callable(callback):
            return default
        try:
            return callback(**dict(kwargs))
        except Exception:
            if fail_closed:
                raise
            return default

    @staticmethod
    def _finite_or_default(value: object, *, field_name: str, default: Decimal) -> Decimal:
        if value is None:
            return default
        try:
            return finite_decimal(value, field_name=field_name)
        except (TypeError, ValueError):
            return default

    def confirm_intent(
        self,
        intent: NativeOrderIntent,
        *,
        leg: LegSnapshot | None = None,
        current_time: datetime | None = None,
    ) -> AdmissionDecision:
        context = HedgeCallbackContext(
            current_time=utc_datetime(current_time),
            leg=leg,
            intent=intent,
        )
        decisions: list[bool] = []
        try:
            if self.native_enabled:
                native = self._call_optional(
                    "hedge_confirm_order",
                    kwargs={
                        "context": context,
                        "intent": HedgeOrderView.from_intent(intent),
                        "leg": None if leg is None else HedgeTradeView.from_leg(leg),
                    },
                    default=True,
                    fail_closed=self.fail_closed_confirmations,
                )
                decisions.append(bool(native))
            if self.legacy_enabled:
                if intent.reduce_only:
                    legacy_name = "confirm_trade_exit"
                    legacy_kwargs = {
                        "pair": intent.pair,
                        "trade": None if leg is None else HedgeTradeView.from_leg(leg),
                        "order_type": "limit",
                        "amount": float(intent.quantity),
                        "rate": float(intent.price),
                        "time_in_force": "GTC",
                        "exit_reason": intent.action.value,
                        "current_time": context.current_time,
                    }
                else:
                    legacy_name = "confirm_trade_entry"
                    legacy_kwargs = {
                        "pair": intent.pair,
                        "order_type": "limit",
                        "amount": float(intent.quantity),
                        "rate": float(intent.price),
                        "time_in_force": "GTC",
                        "current_time": context.current_time,
                        "entry_tag": str(intent.metadata.get("tag", "")),
                        "side": intent.side.pairlock_side,
                    }
                if self._legacy_callback_available(legacy_name):
                    legacy = self._call_optional(
                        legacy_name,
                        kwargs=legacy_kwargs,
                        default=True,
                        fail_closed=self.fail_closed_confirmations,
                    )
                    decisions.append(bool(legacy))
                else:
                    decisions.append(True)
        except Exception as exc:
            return AdmissionDecision.block(
                AdmissionCode.STRATEGY_REJECTED,
                f"strategy confirmation failed closed: {type(exc).__name__}",
            )
        if all(decisions):
            return AdmissionDecision.allow(reason="STRATEGY_CONFIRMED")
        return AdmissionDecision.block(
            AdmissionCode.STRATEGY_REJECTED,
            "strategy callback rejected Hedge order intent",
        )

    def custom_stake_amount(
        self,
        intent: NativeOrderIntent,
        *,
        proposed_stake: object,
        max_stake: object,
        leg: LegSnapshot | None = None,
        current_time: datetime | None = None,
        leverage: object = ONE,
    ) -> Decimal:
        proposed = finite_decimal(proposed_stake, field_name="proposed_stake")
        maximum = finite_decimal(max_stake, field_name="max_stake")
        leverage_value = finite_decimal(leverage, field_name="leverage")
        if proposed < ZERO or maximum < ZERO or leverage_value <= ZERO:
            raise ValueError("stake amounts cannot be negative and leverage must be positive")
        context = HedgeCallbackContext(
            current_time=utc_datetime(current_time),
            leg=leg,
            intent=intent,
        )
        candidates: list[Decimal] = [proposed]
        if self.native_enabled:
            value = self._call_optional(
                "hedge_custom_stake_amount",
                kwargs={
                    "context": context,
                    "intent": HedgeOrderView.from_intent(intent),
                    "proposed_stake": proposed,
                    "max_stake": maximum,
                },
                default=proposed,
            )
            candidates.append(self._finite_or_default(value, field_name="hedge_custom_stake_amount", default=proposed))
        if self.legacy_enabled and self._legacy_callback_available("custom_stake_amount"):
            value = self._call_optional(
                "custom_stake_amount",
                kwargs={
                    "pair": intent.pair,
                    "current_time": context.current_time,
                    "current_rate": float(intent.price),
                    "proposed_stake": float(proposed),
                    "min_stake": None,
                    "max_stake": float(maximum),
                    "leverage": float(leverage_value),
                    "entry_tag": str(intent.metadata.get("tag") or intent.metadata.get("reason") or "") or None,
                    "side": intent.side.pairlock_side,
                },
                default=float(proposed),
            )
            candidates.append(self._finite_or_default(value, field_name="custom_stake_amount", default=proposed))
        result = min(candidates)
        return min(max(result, ZERO), maximum, proposed)

    def leverage(
        self,
        *,
        pair: str,
        side: HedgeSide,
        proposed_leverage: object,
        max_leverage: object,
        current_time: datetime | None = None,
        current_rate: object = ZERO,
        entry_tag: str | None = None,
    ) -> Decimal:
        proposed = finite_decimal(proposed_leverage, field_name="proposed_leverage")
        maximum = finite_decimal(max_leverage, field_name="max_leverage")
        rate = finite_decimal(current_rate, field_name="current_rate")
        context = HedgeCallbackContext(current_time=utc_datetime(current_time))
        candidates: list[Decimal] = [proposed]
        if self.native_enabled:
            value = self._call_optional(
                "hedge_leverage",
                kwargs={
                    "context": context,
                    "pair": pair,
                    "side": HedgeSide.parse(side),
                    "proposed_leverage": proposed,
                    "max_leverage": maximum,
                },
                default=proposed,
            )
            candidates.append(self._finite_or_default(value, field_name="hedge_leverage", default=proposed))
        if self.legacy_enabled and self._legacy_callback_available("leverage"):
            value = self._call_optional(
                "leverage",
                kwargs={
                    "pair": pair,
                    "current_time": context.current_time,
                    "current_rate": float(rate),
                    "proposed_leverage": float(proposed),
                    "max_leverage": float(maximum),
                    "entry_tag": entry_tag,
                    "side": HedgeSide.parse(side).pairlock_side,
                },
                default=float(proposed),
            )
            candidates.append(self._finite_or_default(value, field_name="leverage", default=proposed))
        result = min(candidates)
        if result <= ZERO:
            raise ValueError("leverage callback must return a positive value")
        return min(result, maximum, proposed)

    def custom_order_price(
        self,
        intent: NativeOrderIntent,
        *,
        current_time: datetime | None = None,
        max_distance_ratio: object = Decimal("0.02"),
    ) -> Decimal:
        """Translate custom entry/exit price callbacks with a hard distance clamp."""

        limit = finite_decimal(max_distance_ratio, field_name="max_distance_ratio")
        if limit < ZERO or limit >= ONE:
            raise ValueError("custom price max distance ratio must be in [0, 1)")
        context = HedgeCallbackContext(current_time=utc_datetime(current_time), intent=intent)
        proposed = intent.price
        candidates: list[Decimal] = []
        native_callback = getattr(self.strategy, "hedge_custom_order_price", None)
        if self.native_enabled and callable(native_callback):
            value = self._call_optional(
                "hedge_custom_order_price",
                kwargs={
                    "context": context,
                    "intent": HedgeOrderView.from_intent(intent),
                    "proposed_rate": proposed,
                },
                default=proposed,
            )
            candidates.append(self._finite_or_default(value, field_name="hedge_custom_order_price", default=proposed))
        callback_name = "custom_exit_price" if intent.reduce_only else "custom_entry_price"
        if self.legacy_enabled and self._legacy_callback_available(callback_name):
            tag = str(intent.metadata.get("tag") or intent.metadata.get("reason") or "") or None
            if intent.reduce_only:
                synthetic_leg = LegSnapshot(
                    pair=intent.pair,
                    side=intent.side,
                    bucket=intent.bucket,
                    quantity=intent.quantity,
                    average_price=proposed,
                    mark_price=proposed,
                    opened_at=context.current_time,
                )
                legacy_kwargs = {
                    "pair": intent.pair,
                    "trade": HedgeTradeView.from_leg(synthetic_leg),
                    "current_time": context.current_time,
                    "proposed_rate": float(proposed),
                    "current_profit": 0.0,
                    "exit_tag": tag,
                }
            else:
                legacy_kwargs = {
                    "pair": intent.pair,
                    "trade": None,
                    "current_time": context.current_time,
                    "proposed_rate": float(proposed),
                    "entry_tag": tag,
                    "side": intent.side.pairlock_side,
                }
            legacy = self._call_optional(
                callback_name,
                kwargs=legacy_kwargs,
                default=float(proposed),
            )
            candidates.append(self._finite_or_default(legacy, field_name=callback_name, default=proposed))
        if not candidates:
            return proposed
        # In BOTH mode choose the candidate closest to the planner price, then clamp.
        selected = min(candidates, key=lambda item: abs(item - proposed))
        lower = proposed * (ONE - limit)
        upper = proposed * (ONE + limit)
        return min(max(selected, lower), upper)

    def custom_exit_reason(
        self,
        leg: LegSnapshot,
        *,
        current_time: datetime | None = None,
    ) -> str | None:
        context = HedgeCallbackContext(current_time=utc_datetime(current_time), leg=leg)
        reasons: list[str] = []
        if self.native_enabled:
            result = self._call_optional(
                "hedge_custom_exit",
                kwargs={"context": context, "leg": HedgeTradeView.from_leg(leg)},
                default=None,
            )
            if result is True:
                reasons.append("HEDGE_CUSTOM_EXIT")
            elif result not in (None, False):
                reasons.append(str(result)[:128])
        if self.legacy_enabled and self._legacy_callback_available("custom_exit"):
            result = self._call_optional(
                "custom_exit",
                kwargs={
                    "pair": leg.pair,
                    "trade": HedgeTradeView.from_leg(leg),
                    "current_time": context.current_time,
                    "current_rate": float(leg.mark_price),
                    "current_profit": float(leg.profit_ratio),
                },
                default=None,
            )
            if result is True:
                reasons.append("CUSTOM_EXIT")
            elif result not in (None, False):
                reasons.append(str(result)[:128])
        return None if not reasons else reasons[0]

    @staticmethod
    def _normalize_stoploss(value: object, *, field_name: str) -> Decimal:
        result = finite_decimal(value, field_name=field_name)
        if result > ZERO:
            result = -result
        if result < -ONE or result >= ZERO:
            raise ValueError("custom stoploss must be in [-1, 0)")
        return result

    def custom_stoploss(
        self,
        leg: LegSnapshot,
        *,
        current_time: datetime | None = None,
    ) -> Decimal | None:
        context = HedgeCallbackContext(current_time=utc_datetime(current_time), leg=leg)
        candidates: list[Decimal] = []
        if self.native_enabled:
            value = self._call_optional(
                "hedge_custom_stoploss",
                kwargs={
                    "context": context,
                    "leg": HedgeTradeView.from_leg(leg),
                    "current_profit": leg.profit_ratio,
                },
                default=None,
            )
            if value is not None:
                try:
                    candidates.append(
                        self._normalize_stoploss(value, field_name="hedge_custom_stoploss")
                    )
                except (TypeError, ValueError):
                    pass
        if (
            self.legacy_enabled
            and bool(getattr(self.strategy, "use_custom_stoploss", False))
            and self._legacy_callback_available("custom_stoploss")
        ):
            value = self._call_optional(
                "custom_stoploss",
                kwargs={
                    "pair": leg.pair,
                    "trade": HedgeTradeView.from_leg(leg),
                    "current_time": context.current_time,
                    "current_rate": float(leg.mark_price),
                    "current_profit": float(leg.profit_ratio),
                    "after_fill": False,
                },
                default=None,
            )
            if value is not None:
                try:
                    candidates.append(self._normalize_stoploss(value, field_name="custom_stoploss"))
                except (TypeError, ValueError):
                    pass
        # Closer to zero exits earlier and is therefore the conservative intersection.
        return None if not candidates else max(candidates)

    def custom_roi(
        self,
        leg: LegSnapshot,
        *,
        current_time: datetime | None = None,
    ) -> Decimal | None:
        now = utc_datetime(current_time)
        duration = 0 if leg.opened_at is None else max(0, int((now - leg.opened_at).total_seconds() // 60))
        context = HedgeCallbackContext(current_time=now, leg=leg)
        candidates: list[Decimal] = []
        if self.native_enabled:
            value = self._call_optional(
                "hedge_custom_roi",
                kwargs={
                    "context": context,
                    "leg": HedgeTradeView.from_leg(leg),
                    "trade_duration": duration,
                },
                default=None,
            )
            if value is not None:
                normalized = self._finite_or_default(
                    value, field_name="hedge_custom_roi", default=Decimal("-1")
                )
                if normalized >= ZERO:
                    candidates.append(normalized)
        if (
            self.legacy_enabled
            and bool(getattr(self.strategy, "use_custom_roi", False))
            and self._legacy_callback_available("custom_roi")
        ):
            value = self._call_optional(
                "custom_roi",
                kwargs={
                    "pair": leg.pair,
                    "trade": HedgeTradeView.from_leg(leg),
                    "current_time": now,
                    "trade_duration": duration,
                    "entry_tag": None,
                    "side": leg.side.pairlock_side,
                },
                default=None,
            )
            if value is not None:
                normalized = self._finite_or_default(
                    value, field_name="custom_roi", default=Decimal("-1")
                )
                if normalized >= ZERO:
                    candidates.append(normalized)
        if any(value < ZERO for value in candidates):
            raise ValueError("custom ROI cannot be negative")
        return None if not candidates else min(candidates)

    def order_filled(
        self,
        *,
        intent: NativeOrderIntent,
        fill_price: object,
        fill_quantity: object,
        current_time: datetime | None = None,
    ) -> None:
        at = utc_datetime(current_time)
        price = finite_decimal(fill_price, field_name="fill_price")
        quantity = finite_decimal(fill_quantity, field_name="fill_quantity")
        if self.native_enabled:
            callback: Callable[..., Any] | None = getattr(self.strategy, "hedge_order_filled", None)
            if callback is not None and callable(callback):
                callback(
                    context=HedgeCallbackContext(current_time=at, intent=intent),
                    intent=HedgeOrderView.from_intent(intent),
                    fill_price=price,
                    fill_quantity=quantity,
                )
        if self.legacy_enabled and self._legacy_callback_available("order_filled"):
            leg = LegSnapshot(
                pair=intent.pair,
                side=intent.side,
                bucket=intent.bucket,
                quantity=quantity,
                average_price=price,
                mark_price=price,
                opened_at=at,
            )
            self._call_optional(
                "order_filled",
                kwargs={
                    "pair": intent.pair,
                    "trade": HedgeTradeView.from_leg(leg),
                    "order": HedgeOrderView.from_intent(intent),
                    "current_time": at,
                },
                default=None,
            )
