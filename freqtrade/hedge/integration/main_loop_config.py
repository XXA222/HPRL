"""Typed configuration for the production-equivalent Hedge main loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.integration.production_main_loop import HedgeExecutionMode
from freqtrade.hedge.symbols import raw_symbol

_SUPPORTED_SETTLE_SUFFIXES = ("USDT", "USDC", "FDUSD")
_ALLOWED_KEYS = frozenset(
    {
        "enabled",
        "mode",
        "allowed_symbols",
        "state_backend",
        "state_path",
        "max_submissions_per_cycle",
        "max_cancellations_per_cycle",
        "block_new_risk_on_external_side",
        "require_stream_fresh",
        "require_rest_fresh",
        "require_reconciliation_consistent",
        "recover_on_start",
        "authoritative_execution_enabled",
    }
)


def _valid_perpetual_symbol(value: str) -> bool:
    return (
        value.isascii()
        and value.isalnum()
        and any(value.endswith(suffix) and len(value) > len(suffix) for suffix in _SUPPORTED_SETTLE_SUFFIXES)
    )


@dataclass(frozen=True, slots=True)
class ProductionMainLoopConfig:
    enabled: bool = False
    mode: HedgeExecutionMode = HedgeExecutionMode.HEDGE_PRODUCTION_LOCKED
    allowed_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    state_backend: str = "sql"
    state_path: str | None = None
    max_submissions_per_cycle: int = 32
    max_cancellations_per_cycle: int = 64
    block_new_risk_on_external_side: bool = True
    require_stream_fresh: bool = True
    require_rest_fresh: bool = True
    require_reconciliation_consistent: bool = True
    recover_on_start: bool = True
    authoritative_execution_enabled: bool = False

    def __post_init__(self) -> None:
        mode = HedgeExecutionMode(self.mode)
        if mode not in {
            HedgeExecutionMode.HEDGE_SIMULATED,
            HedgeExecutionMode.HEDGE_PRODUCTION_LOCKED,
            HedgeExecutionMode.HEDGE_PRODUCTION_ARMED,
        }:
            raise OperationalException("Unsupported hedge.main_loop.mode.")
        if (
            mode is HedgeExecutionMode.HEDGE_PRODUCTION_ARMED
            and not self.authoritative_execution_enabled
        ):
            raise OperationalException(
                "HEDGE_PRODUCTION_ARMED requires authoritative_execution_enabled=true."
            )
        symbols = tuple(dict.fromkeys(raw_symbol(item) for item in self.allowed_symbols))
        if not symbols or not all(_valid_perpetual_symbol(item) for item in symbols):
            raise OperationalException(
                "hedge.main_loop.allowed_symbols must contain valid USDT/USDC/FDUSD perpetuals."
            )
        backend = str(self.state_backend).strip().lower()
        if backend not in {"sql", "json", "memory"}:
            raise OperationalException(
                "hedge.main_loop.state_backend must be sql, json, or memory."
            )
        if backend == "json" and not str(self.state_path or "").strip():
            raise OperationalException(
                "hedge.main_loop.state_path is required for the json backend."
            )
        for name in ("max_submissions_per_cycle", "max_cancellations_per_cycle"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise OperationalException(f"hedge.main_loop.{name} must be a positive integer.")
        for name in (
            "enabled",
            "block_new_risk_on_external_side",
            "require_stream_fresh",
            "require_rest_fresh",
            "require_reconciliation_consistent",
            "recover_on_start",
            "authoritative_execution_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise OperationalException(f"hedge.main_loop.{name} must be a boolean.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "allowed_symbols", symbols)
        object.__setattr__(self, "state_backend", backend)
        if self.state_path is not None:
            object.__setattr__(self, "state_path", str(Path(self.state_path).expanduser()))


def production_main_loop_config_from_mapping(
    config: Mapping[str, Any],
) -> ProductionMainLoopConfig:
    hedge = config.get("hedge", {})
    if not isinstance(hedge, Mapping):
        raise OperationalException("hedge must be a JSON object.")
    raw = hedge.get("main_loop", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise OperationalException("hedge.main_loop must be a JSON object.")
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise OperationalException(
            "Unknown hedge.main_loop configuration key(s): " + ", ".join(unknown)
        )
    values = dict(raw)
    if "mode" in values:
        try:
            values["mode"] = HedgeExecutionMode(str(values["mode"]).strip().upper())
        except ValueError as exc:
            raise OperationalException("Invalid hedge.main_loop.mode.") from exc
    symbols = values.get("allowed_symbols")
    if symbols is not None:
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise OperationalException(
                "hedge.main_loop.allowed_symbols must be an array of strings."
            )
        values["allowed_symbols"] = tuple(symbols)
    result = ProductionMainLoopConfig(**values)
    managed_pair = config.get("managed_pair")
    if result.enabled and managed_pair is not None:
        managed = raw_symbol(str(managed_pair))
        if managed not in result.allowed_symbols:
            raise OperationalException(
                "managed_pair must be included in hedge.main_loop.allowed_symbols."
            )
    return result
