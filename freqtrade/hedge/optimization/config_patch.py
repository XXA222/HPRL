"""Safe configuration patching for optimization trials.

Only explicitly allowlisted simulation and planner fields may be changed.  Account,
credential, database, network, and live-trading settings are intentionally outside
this module's writable surface.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from decimal import Decimal
from typing import Any

from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec, exact_decimal


PLANNER_FIELDS = frozenset(
    {
        "long_enabled",
        "short_enabled",
        "core_wallet_exposure_long",
        "core_wallet_exposure_short",
        "tactical_wallet_exposure_long",
        "tactical_wallet_exposure_short",
        "max_wallet_exposure_long",
        "max_wallet_exposure_short",
        "max_gross_wallet_exposure",
        "initial_entry_fraction",
        "max_grid_layers",
        "grid_spacing",
        "grid_spacing_growth",
        "grid_qty_growth",
        "trailing_rebound",
        "take_profit_spacing",
        "take_profit_layers",
        "tactical_reduce_fraction",
        "core_min_fraction",
        "cooldown_seconds",
        "replace_price_tolerance_ticks",
        "replace_qty_tolerance_steps",
        "replace_min_age_seconds",
        "unstuck_trigger_gross_exposure",
        "unstuck_reduce_fraction",
        "maintenance_margin_rate",
        "target_net_wallet_exposure",
        "net_repair_threshold",
        "trailing_trigger_distance",
        "trailing_timeout_seconds",
        "max_pending_entries",
        "max_single_order_notional",
        "unstuck_max_holding_seconds",
        "unstuck_daily_loss_budget",
        "unstuck_weekly_loss_budget",
        "unstuck_min_cooldown_seconds",
        "unstuck_min_risk_improvement",
        "unstuck_limit_only",
        "liquidation_fee_rate",
        "liquidation_buffer_warning_ratio",
    }
)

PAPER_FIELDS = frozenset(
    {
        "leverage",
        "long_signal",
        "short_signal",
        "volume_participation",
        "market_slippage_bps",
        "max_entry_layers_per_bar",
        "max_reduce_layers_per_bar",
        "max_fill_ratio_per_order",
        "max_fills_per_bar",
        "bar_volume",
    }
)

ALLOWED_PARAMETER_PATHS = frozenset(
    {f"hedge.planner.{name}" for name in PLANNER_FIELDS}
    | {f"hedge.paper.{name}" for name in PAPER_FIELDS}
)


def validate_parameter_path(path: str) -> None:
    if path not in ALLOWED_PARAMETER_PATHS:
        raise ValueError(
            f"optimization path {path!r} is not allowlisted; credentials, runtime gates, "
            "database settings, and live-trading controls cannot be optimized"
        )


def _normalize_value(spec: ParameterSpec, value: object) -> object:  # noqa: C901
    if spec.kind is ParameterKind.DECIMAL:
        normalized = exact_decimal(value, field_name=spec.name)
        low = spec.low
        high = spec.high
        if not isinstance(low, Decimal) or not isinstance(high, Decimal):
            raise TypeError(f"decimal parameter {spec.name} has invalid bounds")
        if not low <= normalized <= high:
            raise ValueError(f"{spec.name} is outside [{low}, {high}]")
        if spec.step is not None:
            step = spec.step
            if not isinstance(step, Decimal):
                raise TypeError(f"decimal parameter {spec.name} has an invalid step")
            quotient = (normalized - low) / step
            if quotient != quotient.to_integral_value():
                raise ValueError(f"{spec.name} does not align to step {step}")
        return normalized
    if spec.kind is ParameterKind.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{spec.name} must be an integer")
        if (
            not isinstance(spec.low, int)
            or isinstance(spec.low, bool)
            or not isinstance(spec.high, int)
            or isinstance(spec.high, bool)
        ):
            raise TypeError(f"integer parameter {spec.name} has invalid bounds")
        if not spec.low <= value <= spec.high:
            raise ValueError(f"{spec.name} is outside [{spec.low}, {spec.high}]")
        step = 1 if spec.step is None else spec.step
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError(f"integer parameter {spec.name} has an invalid step")
        if (value - spec.low) % step:
            raise ValueError(f"{spec.name} does not align to step {step}")
        return value
    if spec.kind is ParameterKind.BOOLEAN:
        if not isinstance(value, bool):
            raise TypeError(f"{spec.name} must be a boolean")
        return value
    if value not in spec.choices:
        raise ValueError(f"{spec.name} must be one of {spec.choices!r}")
    return deepcopy(value)


def _set_path(target: MutableMapping[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    cursor: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise TypeError(f"cannot patch {path!r}; {part!r} is not an object")
        cursor = child
    cursor[parts[-1]] = value


def apply_parameters(
    base_config: Mapping[str, Any],
    specs: Sequence[ParameterSpec],
    values: Mapping[str, object],
) -> dict[str, Any]:
    """Return a deep-copied configuration with one validated trial applied."""

    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("parameter names must be unique")
    unknown = sorted(set(values) - set(by_name))
    missing = sorted(set(by_name) - set(values))
    if unknown:
        raise ValueError(f"trial contains unknown parameters: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"trial is missing parameters: {', '.join(missing)}")
    patched = deepcopy(dict(base_config))
    for spec in specs:
        validate_parameter_path(spec.path)
        _set_path(patched, spec.path, _normalize_value(spec, values[spec.name]))
    return patched
