"""Explicit Hedge schema administration commands.

The native Freqtrade database migration path must remain unaware of H3.  This
module provides the only CLI surface that may plan, apply or verify Hedge schema.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


def start_hedge_db(args: dict[str, Any]) -> None:
    from freqtrade.configuration.config_setup import setup_utils_configuration
    from freqtrade.hedge.config import validate_hedge_config
    from freqtrade.persistence.hedge_bootstrap import (
        bootstrap_hedge_schema,
        plan_hedge_migration,
        verify_hedge_schema,
    )

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    hedge_config = validate_hedge_config(config)
    if not hedge_config.enabled:
        raise OperationalException("hedge-db requires hedge_mode_enabled=true")

    db_url = args.get("db_url") or config.get("db_url")
    if not isinstance(db_url, str) or not db_url.strip():
        raise OperationalException("hedge-db requires a database URL")
    engine = create_engine(db_url, future=True)
    action = str(args.get("hedge_db_action", "status")).strip().lower()
    backup = args.get("hedge_backup_directory")
    backup_directory = None if backup is None else Path(str(backup)).expanduser().resolve()

    try:
        if action == "plan":
            payload = asdict(plan_hedge_migration(engine, config))
        elif action == "migrate":
            report = bootstrap_hedge_schema(
                engine,
                config,
                backup_directory=backup_directory,
            )
            payload = {
                "plan": asdict(report.plan),
                "migration": asdict(report.migration),
                "recovered_accounts": report.recovered_accounts,
            }
        elif action in {"status", "verify"}:
            payload = {
                "action": action,
                "migrations": list(verify_hedge_schema(engine, config)),
            }
        else:  # pragma: no cover - argparse restricts this
            raise OperationalException(f"Unsupported hedge-db action: {action}")
    finally:
        engine.dispose()

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
