"""Stateful evidence for progressive live canary promotion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from .canary import CanaryLevel


@dataclass(frozen=True, slots=True)
class CanaryRunMetrics:
    started_at: datetime
    observed_at: datetime
    level: CanaryLevel
    orders_submitted: int
    orders_filled: int
    realized_pnl: Decimal
    fees: Decimal
    funding: Decimal
    max_drawdown_ratio: Decimal
    unknown_orders: int
    reconciliation_divergences: int
    risk_limit_breaches: int
    manual_interventions: int

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.observed_at.tzinfo is None:
            raise ValueError("canary timestamps must be timezone-aware")
        if self.observed_at < self.started_at:
            raise ValueError("canary observed_at cannot precede started_at")
        for name in (
            "orders_submitted", "orders_filled", "unknown_orders",
            "reconciliation_divergences", "risk_limit_breaches", "manual_interventions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.orders_filled > self.orders_submitted:
            raise ValueError("orders_filled cannot exceed orders_submitted")
        for name in ("realized_pnl", "fees", "funding", "max_drawdown_ratio"):
            value = Decimal(getattr(self, name))
            if not value.is_finite():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.fees < 0 or self.max_drawdown_ratio < 0:
            raise ValueError("fees/drawdown must be nonnegative")
        object.__setattr__(self, "started_at", self.started_at.astimezone(UTC))
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))

    @property
    def duration(self) -> timedelta:
        return self.observed_at - self.started_at

    @property
    def net_realized_pnl(self) -> Decimal:
        return self.realized_pnl - self.fees - self.funding


@dataclass(frozen=True, slots=True)
class CanaryPromotionPolicy:
    micro_min_duration: timedelta = timedelta(hours=6)
    micro_min_fills: int = 10
    small_min_duration: timedelta = timedelta(hours=24)
    small_min_fills: int = 30
    bounded_min_duration: timedelta = timedelta(hours=72)
    bounded_min_fills: int = 100
    max_drawdown_ratio: Decimal = Decimal("0.03")
    max_manual_interventions: int = 0


@dataclass(frozen=True, slots=True)
class CanaryPromotionDecision:
    current_level: CanaryLevel
    next_level: CanaryLevel
    promote: bool
    reasons: tuple[str, ...]
    net_realized_pnl: Decimal


def evaluate_canary_promotion(
    metrics: CanaryRunMetrics,
    policy: CanaryPromotionPolicy | None = None,
) -> CanaryPromotionDecision:
    policy = policy or CanaryPromotionPolicy()
    reasons: list[str] = []
    if metrics.unknown_orders:
        reasons.append("CANARY_UNKNOWN_ORDERS")
    if metrics.reconciliation_divergences:
        reasons.append("CANARY_RECONCILIATION_DIVERGENCE")
    if metrics.risk_limit_breaches:
        reasons.append("CANARY_RISK_LIMIT_BREACH")
    if metrics.manual_interventions > policy.max_manual_interventions:
        reasons.append("CANARY_MANUAL_INTERVENTION")
    if metrics.max_drawdown_ratio > policy.max_drawdown_ratio:
        reasons.append("CANARY_DRAWDOWN_LIMIT")

    if metrics.level is CanaryLevel.MICRO:
        required_duration, required_fills, next_level = (
            policy.micro_min_duration, policy.micro_min_fills, CanaryLevel.SMALL
        )
    elif metrics.level is CanaryLevel.SMALL:
        required_duration, required_fills, next_level = (
            policy.small_min_duration, policy.small_min_fills, CanaryLevel.BOUNDED
        )
    elif metrics.level is CanaryLevel.BOUNDED:
        required_duration, required_fills, next_level = (
            policy.bounded_min_duration, policy.bounded_min_fills, CanaryLevel.BOUNDED
        )
    else:
        return CanaryPromotionDecision(
            metrics.level, metrics.level, False, ("CANARY_LEVEL_NOT_PROMOTABLE",), metrics.net_realized_pnl
        )
    if metrics.duration < required_duration:
        reasons.append("CANARY_DURATION_INSUFFICIENT")
    if metrics.orders_filled < required_fills:
        reasons.append("CANARY_FILL_COUNT_INSUFFICIENT")
    # Profitability is not required for plumbing promotion, but catastrophic negative
    # net PnL must already be caught by drawdown/daily-loss gates.  Preserve PnL as evidence.
    return CanaryPromotionDecision(
        metrics.level,
        next_level,
        not reasons,
        tuple(reasons),
        metrics.net_realized_pnl,
    )
