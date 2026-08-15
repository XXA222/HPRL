from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentBaseline:
    project_root: str
    python_version: str
    platform: str
    exchange: str
    hedge_mode_enabled: bool
    read_only: bool
    live_trading_enabled: bool
    operation_mode: str
    managed_symbols: tuple[str, ...]
    database_configured: bool
    credentials_present: bool

    @property
    def safe_for_runtime_acceptance(self) -> bool:
        return bool(
            self.exchange == "binance"
            and self.hedge_mode_enabled
            and self.read_only
            and not self.live_trading_enabled
            and self.operation_mode in {"readonly", "shadow", "dry_run", "dry-run"}
            and self.managed_symbols
        )


def audit_environment(config: Mapping[str, Any], *, project_root: Path) -> EnvironmentBaseline:
    exchange = config.get("exchange") if isinstance(config.get("exchange"), Mapping) else {}
    hedge = config.get("hedge") if isinstance(config.get("hedge"), Mapping) else {}
    symbols = hedge.get("managed_symbols") or exchange.get("pair_whitelist") or ()
    if isinstance(symbols, str):
        symbols = (symbols,)
    managed = tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
    key = str(exchange.get("key") or "").strip()
    secret = str(exchange.get("secret") or "").strip()
    database_url = str(config.get("db_url") or config.get("database_url") or "").strip()
    return EnvironmentBaseline(
        project_root=str(project_root.resolve()),
        python_version=platform.python_version(),
        platform=platform.platform(),
        exchange=str(exchange.get("name") or "").strip().lower(),
        hedge_mode_enabled=config.get("hedge_mode_enabled") is True,
        read_only=hedge.get("read_only", True) is True,
        live_trading_enabled=hedge.get("live_trading_enabled", False) is True,
        operation_mode=str(hedge.get("operation_mode") or "readonly").strip().lower(),
        managed_symbols=managed,
        database_configured=bool(database_url),
        credentials_present=bool(key and secret),
    )
