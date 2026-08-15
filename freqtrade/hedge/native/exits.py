"""Per-leg and per-bucket ROI, stoploss and trailing-stop policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Mapping

from .callbacks import HedgeStrategyCallbackAdapter
from .models import (
    ExitDecision,
    HedgeBucket,
    HedgeSide,
    LegSnapshot,
    ONE,
    ZERO,
    finite_decimal,
    utc_datetime,
)


@dataclass(frozen=True, slots=True)
class HedgeExitPolicy:
    stoploss: Decimal = Decimal("-0.10")
    minimal_roi: Mapping[int, Decimal] = field(default_factory=lambda: {0: Decimal("0.03")})
    trailing_enabled: bool = False
    trailing_positive: Decimal = Decimal("0.01")
    trailing_offset: Decimal = Decimal("0.02")
    trailing_only_offset_reached: bool = True
    exit_fraction: Decimal = ONE

    def __post_init__(self) -> None:
        stoploss = finite_decimal(self.stoploss, field_name="stoploss")
        if stoploss < -ONE or stoploss >= ZERO:
            raise ValueError("stoploss must be in [-1, 0)")
        trailing_positive = finite_decimal(
            self.trailing_positive,
            field_name="trailing_positive",
        )
        trailing_offset = finite_decimal(self.trailing_offset, field_name="trailing_offset")
        if min(trailing_positive, trailing_offset) < ZERO:
            raise ValueError("trailing values cannot be negative")
        if trailing_positive > trailing_offset and self.trailing_only_offset_reached:
            raise ValueError("trailing_positive cannot exceed trailing_offset")
        exit_fraction = finite_decimal(self.exit_fraction, field_name="exit_fraction")
        if exit_fraction <= ZERO or exit_fraction > ONE:
            raise ValueError("exit_fraction must be in (0, 1]")
        normalized_roi: dict[int, Decimal] = {}
        for minute, value in self.minimal_roi.items():
            if isinstance(minute, bool) or int(minute) < 0:
                raise ValueError("ROI minutes must be nonnegative integers")
            ratio = finite_decimal(value, field_name="minimal_roi")
            if ratio < ZERO:
                raise ValueError("minimal ROI values cannot be negative")
            normalized_roi[int(minute)] = ratio
        object.__setattr__(self, "stoploss", stoploss)
        object.__setattr__(self, "trailing_positive", trailing_positive)
        object.__setattr__(self, "trailing_offset", trailing_offset)
        object.__setattr__(self, "exit_fraction", exit_fraction)
        object.__setattr__(self, "minimal_roi", normalized_roi)

    def roi_for_age(self, age: timedelta) -> Decimal | None:
        minutes = max(0, int(age.total_seconds() // 60))
        eligible = [minute for minute in self.minimal_roi if minute <= minutes]
        if not eligible:
            return None
        return self.minimal_roi[max(eligible)]


class HedgeExitPolicyEngine:
    """Evaluate priority-ordered hard risk, custom exit, trailing and ROI rules."""

    PRIORITY_HARD_STOP = 1000
    PRIORITY_CUSTOM_STOP = 950
    PRIORITY_CUSTOM_EXIT = 900
    PRIORITY_TRAILING = 800
    PRIORITY_ROI = 700

    def __init__(
        self,
        policies: Mapping[tuple[HedgeSide, HedgeBucket], HedgeExitPolicy],
        *,
        callback_adapter: HedgeStrategyCallbackAdapter | None = None,
    ) -> None:
        self.policies = {
            (HedgeSide.parse(side), HedgeBucket(bucket)): policy
            for (side, bucket), policy in policies.items()
        }
        self.callback_adapter = callback_adapter
        self._best_profit: dict[tuple[str, HedgeSide, HedgeBucket], Decimal] = {}

    def reset(self, pair: str, side: HedgeSide, bucket: HedgeBucket) -> None:
        self._best_profit.pop((pair.upper(), HedgeSide.parse(side), HedgeBucket(bucket)), None)

    def evaluate(
        self,
        leg: LegSnapshot,
        *,
        at: datetime | None = None,
    ) -> ExitDecision:
        if leg.quantity <= ZERO:
            self.reset(leg.pair, leg.side, leg.bucket)
            return ExitDecision.hold("FLAT")
        policy = self.policies.get((leg.side, leg.bucket))
        if policy is None:
            return ExitDecision.hold("NO_EXIT_POLICY")
        now = utc_datetime(at)
        profit = leg.profit_ratio
        key = (leg.pair, leg.side, leg.bucket)
        best = max(self._best_profit.get(key, profit), profit)
        self._best_profit[key] = best

        if profit <= policy.stoploss:
            return ExitDecision(
                True,
                ONE,
                f"HARD_STOPLOSS:{profit}",
                self.PRIORITY_HARD_STOP,
                True,
            )

        if self.callback_adapter is not None:
            custom_stop = self.callback_adapter.custom_stoploss(leg, current_time=now)
            if custom_stop is not None and profit <= custom_stop:
                return ExitDecision(
                    True,
                    ONE,
                    f"CUSTOM_STOPLOSS:{profit}",
                    self.PRIORITY_CUSTOM_STOP,
                    True,
                )
            custom_reason = self.callback_adapter.custom_exit_reason(leg, current_time=now)
            if custom_reason:
                return ExitDecision(
                    True,
                    policy.exit_fraction,
                    custom_reason,
                    self.PRIORITY_CUSTOM_EXIT,
                    False,
                )

        if policy.trailing_enabled:
            activated = best >= policy.trailing_offset
            if not policy.trailing_only_offset_reached or activated:
                trailing_floor = best - policy.trailing_positive
                if profit <= trailing_floor and best > ZERO:
                    return ExitDecision(
                        True,
                        policy.exit_fraction,
                        f"TRAILING_STOP:{profit}:{best}",
                        self.PRIORITY_TRAILING,
                        False,
                    )

        age = timedelta(0) if leg.opened_at is None else max(now - leg.opened_at, timedelta(0))
        roi = policy.roi_for_age(age)
        if self.callback_adapter is not None:
            custom_roi = self.callback_adapter.custom_roi(leg, current_time=now)
            if custom_roi is not None:
                roi = custom_roi if roi is None else min(roi, custom_roi)
        if roi is not None and profit >= roi:
            return ExitDecision(
                True,
                policy.exit_fraction,
                f"MINIMAL_ROI:{roi}",
                self.PRIORITY_ROI,
                False,
            )
        return ExitDecision.hold("EXIT_RULES_CLEAR")
