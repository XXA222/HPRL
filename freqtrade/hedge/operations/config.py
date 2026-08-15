"""Configuration validation for the durable Dry-run operations runtime."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from freqtrade.hedge.config_migration import has_legacy_operations_alias

_ALLOWED = {
    "enabled",
    "state_path",
    "market_max_age_seconds",
    "market_max_gap_candles",
    "mark_index_max_divergence_bps",
    "warmup_candles",
    "informative_warmup",
    "max_gross_ratio",
    "max_margin_ratio",
    "max_net_ratio",
    "drawdown_warning",
    "drawdown_pause",
    "drawdown_kill",
    "drawdown_recovery",
}


def _decimal(raw: object, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be decimal") from exc
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def operations_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the canonical Clean Mainline operations configuration."""

    if has_legacy_operations_alias(config):
        raise ValueError("OPERATIONS_CONFIG_NOT_NORMALIZED")
    hedge = config.get("hedge", {})
    if not isinstance(hedge, Mapping):
        return {}
    current = hedge.get("operations")
    return current if isinstance(current, Mapping) else {}


def validate_operations_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    hedge = config.get("hedge", {})
    hedge = hedge if isinstance(hedge, Mapping) else {}
    try:
        raw = operations_config(config)
    except ValueError as exc:
        if str(exc) == "OPERATIONS_CONFIG_NOT_NORMALIZED":
            return ("OPERATIONS_CONFIG_NOT_NORMALIZED",)
        raise
    if not raw:
        return ()

    unknown = sorted(set(raw) - _ALLOWED)
    if unknown:
        errors.append("OPERATIONS_UNKNOWN_KEYS:" + ",".join(unknown))

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append("OPERATIONS_ENABLED_MUST_BE_BOOL")

    if enabled:
        if config.get("dry_run") is not True:
            errors.append("OPERATIONS_REQUIRES_DRY_RUN")
        if hedge.get("read_only", True) is not True:
            errors.append("OPERATIONS_REQUIRES_READ_ONLY")
        if hedge.get("live_trading_enabled", False) is not False:
            errors.append("OPERATIONS_FORBIDS_LIVE_TRADING")

    path = str(raw.get("state_path", "user_data/hedge/operations/runtime-state.json"))
    if not path.strip():
        errors.append("OPERATIONS_STATE_PATH_REQUIRED")
    if ".." in Path(path).parts:
        errors.append("OPERATIONS_STATE_PATH_TRAVERSAL")

    integer_rules = (
        ("market_max_age_seconds", 90, 1),
        ("market_max_gap_candles", 2, 0),
        ("warmup_candles", 100, 1),
    )
    for key, default, minimum in integer_rules:
        try:
            value = int(raw.get(key, default))
            if value < minimum:
                errors.append(f"OPERATIONS_{key.upper()}_INVALID")
        except (TypeError, ValueError):
            errors.append(f"OPERATIONS_{key.upper()}_INVALID")

    informative = raw.get("informative_warmup", {})
    if not isinstance(informative, Mapping):
        errors.append("OPERATIONS_INFORMATIVE_WARMUP_INVALID")
    else:
        for timeframe, count in informative.items():
            try:
                if not str(timeframe).strip() or int(count) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"OPERATIONS_INFORMATIVE_WARMUP_INVALID:{timeframe}")

    try:
        gross = _decimal(raw.get("max_gross_ratio", "0.80"), "max_gross_ratio")
        margin = _decimal(raw.get("max_margin_ratio", "0.55"), "max_margin_ratio")
        net = _decimal(raw.get("max_net_ratio", "0.50"), "max_net_ratio")
        if any(value <= 0 or value > 1 for value in (gross, margin, net)):
            errors.append("OPERATIONS_RISK_RATIOS_INVALID")
    except ValueError:
        errors.append("OPERATIONS_RISK_RATIOS_INVALID")

    try:
        recovery = _decimal(raw.get("drawdown_recovery", "0.03"), "drawdown_recovery")
        warning = _decimal(raw.get("drawdown_warning", "0.05"), "drawdown_warning")
        pause = _decimal(raw.get("drawdown_pause", "0.10"), "drawdown_pause")
        kill = _decimal(raw.get("drawdown_kill", "0.20"), "drawdown_kill")
        if not Decimal(0) <= recovery < warning < pause < kill < Decimal(1):
            errors.append("OPERATIONS_DRAWDOWN_THRESHOLDS_INVALID")
    except ValueError:
        errors.append("OPERATIONS_DRAWDOWN_THRESHOLDS_INVALID")

    return tuple(dict.fromkeys(errors))
