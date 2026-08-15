"""Risk limit definitions with unit-safe validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from freqtrade.hedge.errors import HedgeConfigurationError
from freqtrade.hedge.numeric import require_nonnegative, require_positive, require_unit_interval


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_margin_utilization: Decimal
    min_liquidation_buffer_ratio: Decimal
    max_gross_notional: Decimal | None = None
    max_gross_exposure_ratio: Decimal | None = None
    max_single_order_notional: Decimal | None = None
    max_leg_notional: Decimal | None = None
    max_symbol_gross_notional: Decimal | None = None
    max_net_notional: Decimal | None = None
    max_net_exposure_ratio: Decimal | None = None
    max_long_notional: Decimal | None = None
    max_short_notional: Decimal | None = None
    max_long_exposure_ratio: Decimal | None = None
    max_short_exposure_ratio: Decimal | None = None
    max_pending_order_notional: Decimal | None = None
    max_pending_order_initial_margin: Decimal | None = None
    min_available_balance: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_margin_utilization",
            require_unit_interval(
                self.max_margin_utilization,
                field="max_margin_utilization",
            ),
        )
        object.__setattr__(
            self,
            "min_liquidation_buffer_ratio",
            require_unit_interval(
                self.min_liquidation_buffer_ratio,
                field="min_liquidation_buffer_ratio",
            ),
        )
        for field_name in (
            "max_gross_notional",
            "max_single_order_notional",
            "max_leg_notional",
            "max_symbol_gross_notional",
            "max_net_notional",
            "max_long_notional",
            "max_short_notional",
            "max_pending_order_notional",
            "max_pending_order_initial_margin",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_positive(value, field=field_name))
        for field_name in (
            "max_gross_exposure_ratio",
            "max_net_exposure_ratio",
            "max_long_exposure_ratio",
            "max_short_exposure_ratio",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_positive(value, field=field_name))
        object.__setattr__(
            self,
            "min_available_balance",
            require_nonnegative(self.min_available_balance, field="min_available_balance"),
        )
        if self.max_gross_notional is None and self.max_gross_exposure_ratio is None:
            raise HedgeConfigurationError(
                "Set max_gross_notional or max_gross_exposure_ratio."
            )


    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "RiskLimits":
        """Create limits from a configuration mapping with strict field names."""

        if not isinstance(values, Mapping):
            raise HedgeConfigurationError("Risk limits configuration must be a mapping.")
        allowed = frozenset(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise HedgeConfigurationError(
                f"Unknown risk limit fields: {sorted(unknown)!r}."
            )
        missing = {
            "max_margin_utilization",
            "min_liquidation_buffer_ratio",
        } - set(values)
        if missing:
            raise HedgeConfigurationError(
                f"Missing required risk limit fields: {sorted(missing)!r}."
            )
        return cls(**dict(values))

    def as_dict(self) -> dict[str, str | None]:
        return {
            field_name: (
                None
                if getattr(self, field_name) is None
                else str(getattr(self, field_name))
            )
            for field_name in self.__dataclass_fields__
        }
