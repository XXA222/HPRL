"""Strategy directive contract and risk-reducing planner adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from freqtrade.hedge.planning.context import PlannerConfig

ZERO = Decimal(0)
ONE = Decimal(1)

HEDGE_SIGNAL_COLUMNS = (
    "hedge_long_score",
    "hedge_short_score",
    "hedge_target_net",
    "hedge_target_net_ratio",
    "hedge_confidence",
    "hedge_risk_scale",
    "hedge_long_exposure_scale",
    "hedge_short_exposure_scale",
    "hedge_allow_new_risk",
    "hedge_regime",
    "hedge_reason",
    "hedge_model_version",
)


def _decimal(
    value: object,
    default: Decimal,
    *,
    low: Decimal | None = None,
    high: Decimal | None = None,
) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        result = default
    if not result.is_finite():
        result = default
    if low is not None:
        result = max(low, result)
    if high is not None:
        result = min(high, result)
    return result


def _optional_decimal(
    value: object,
    *,
    low: Decimal | None = None,
    high: Decimal | None = None,
) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not result.is_finite():
        return None
    if low is not None:
        result = max(low, result)
    if high is not None:
        result = min(high, result)
    return result


def _bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float | Decimal):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


@runtime_checkable
class StrategyDirectiveLike(Protocol):
    long_score: Decimal
    short_score: Decimal
    target_net_quantity: Decimal | None
    target_net_ratio: Decimal | None
    confidence: Decimal
    risk_scale: Decimal
    long_exposure_scale: Decimal
    short_exposure_scale: Decimal
    allow_new_risk: bool
    regime: str
    reason: str
    model_version: str


@dataclass(frozen=True, slots=True)
class StrategyDirective:
    long_score: Decimal = ZERO
    short_score: Decimal = ZERO
    target_net_quantity: Decimal | None = None
    target_net_ratio: Decimal | None = None
    confidence: Decimal = ONE
    risk_scale: Decimal = ONE
    long_exposure_scale: Decimal = ONE
    short_exposure_scale: Decimal = ONE
    allow_new_risk: bool = True
    regime: str = "UNSPECIFIED"
    reason: str = ""
    model_version: str = "strategy"

    def __post_init__(self) -> None:
        bounded = (
            "long_score",
            "short_score",
            "confidence",
            "risk_scale",
            "long_exposure_scale",
            "short_exposure_scale",
        )
        for name in bounded:
            value = getattr(self, name)
            if not value.is_finite() or value < ZERO or value > ONE:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.target_net_quantity is not None and not self.target_net_quantity.is_finite():
            raise ValueError("target_net_quantity must be finite")
        if self.target_net_ratio is not None:
            if (
                not self.target_net_ratio.is_finite()
                or self.target_net_ratio < -ONE
                or self.target_net_ratio > ONE
            ):
                raise ValueError("target_net_ratio must be within [-1, 1]")
        if self.target_net_quantity is not None and self.target_net_ratio is not None:
            raise ValueError("exact target and target ratio are mutually exclusive")
        if not isinstance(self.allow_new_risk, bool):
            raise TypeError("allow_new_risk must be bool")
        if len(self.regime) > 64 or len(self.reason) > 256 or len(self.model_version) > 128:
            raise ValueError("strategy metadata exceeds contract limits")

    @property
    def effective_risk_scale(self) -> Decimal:
        return min(self.confidence, self.risk_scale)


@dataclass(frozen=True, slots=True)
class StrategyContract:
    schema_version: int = 1
    preferred_columns: tuple[str, ...] = HEDGE_SIGNAL_COLUMNS
    legacy_columns: tuple[str, ...] = ("enter_long", "enter_short")
    closed_candle_only: bool = True
    next_bar_activation: bool = True


def directive_from_values(values: Mapping[str, object]) -> StrategyDirective:
    exact = _optional_decimal(values.get("hedge_target_net"))
    ratio = _optional_decimal(
        values.get("hedge_target_net_ratio"),
        low=-ONE,
        high=ONE,
    )
    # The bounded ratio is safer than an unbounded exact quantity.
    if ratio is not None:
        exact = None
    return StrategyDirective(
        long_score=_decimal(
            values.get("hedge_long_score", values.get("enter_long", ZERO)),
            ZERO,
            low=ZERO,
            high=ONE,
        ),
        short_score=_decimal(
            values.get("hedge_short_score", values.get("enter_short", ZERO)),
            ZERO,
            low=ZERO,
            high=ONE,
        ),
        target_net_quantity=exact,
        target_net_ratio=ratio,
        confidence=_decimal(
            values.get("hedge_confidence", ONE),
            ONE,
            low=ZERO,
            high=ONE,
        ),
        risk_scale=_decimal(
            values.get("hedge_risk_scale", ONE),
            ONE,
            low=ZERO,
            high=ONE,
        ),
        long_exposure_scale=_decimal(
            values.get("hedge_long_exposure_scale", ONE),
            ONE,
            low=ZERO,
            high=ONE,
        ),
        short_exposure_scale=_decimal(
            values.get("hedge_short_exposure_scale", ONE),
            ONE,
            low=ZERO,
            high=ONE,
        ),
        allow_new_risk=_bool(values.get("hedge_allow_new_risk", True), True),
        regime=str(values.get("hedge_regime") or "UNSPECIFIED")[:64],
        reason=str(values.get("hedge_reason") or "")[:256],
        model_version=str(values.get("hedge_model_version") or "strategy")[:128],
    )


def planner_config_for_directive(
    base: PlannerConfig,
    directive: StrategyDirectiveLike,
) -> PlannerConfig:
    scale = min(directive.confidence, directive.risk_scale)
    long_scale = scale * directive.long_exposure_scale
    short_scale = scale * directive.short_exposure_scale
    return replace(
        base,
        core_wallet_exposure_long=base.core_wallet_exposure_long * long_scale,
        tactical_wallet_exposure_long=base.tactical_wallet_exposure_long * long_scale,
        max_wallet_exposure_long=base.max_wallet_exposure_long * long_scale,
        core_wallet_exposure_short=base.core_wallet_exposure_short * short_scale,
        tactical_wallet_exposure_short=base.tactical_wallet_exposure_short * short_scale,
        max_wallet_exposure_short=base.max_wallet_exposure_short * short_scale,
        max_gross_wallet_exposure=base.max_gross_wallet_exposure * scale,
        max_single_order_notional=base.max_single_order_notional * scale,
    )


def target_net_quantity_for_directive(
    *,
    directive: StrategyDirectiveLike,
    base: PlannerConfig,
    equity: Decimal,
    mark_price: Decimal,
) -> Decimal | None:
    if mark_price <= ZERO or equity < ZERO:
        raise ValueError("equity/mark_price invalid")

    target = directive.target_net_quantity
    if directive.target_net_ratio is not None:
        target = equity * directive.target_net_ratio / mark_price
    if target is None:
        return None

    scale = min(directive.confidence, directive.risk_scale)
    side_scale = (
        directive.long_exposure_scale if target >= ZERO else directive.short_exposure_scale
    )
    cap_ratio = (
        base.max_wallet_exposure_long
        if target >= ZERO
        else base.max_wallet_exposure_short
    )
    cap = equity * cap_ratio * scale * side_scale / mark_price
    return min(max(target, -cap), cap)
