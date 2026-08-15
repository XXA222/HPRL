"""Explicit Hedge schema bootstrap, planning, verification and recovery.

This module is the only production entry point allowed to mutate Hedge schema.
Freqtrade's native ``check_migrate`` deliberately knows nothing about H3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import sessionmaker

from freqtrade.exceptions import OperationalException
from freqtrade.persistence.hedge_migrations import (
    HedgeMigrationReport,
    hedge_migrations_pending,
    migration_status,
    prepare_hedge_migration_backup,
    run_hedge_migrations,
)
from freqtrade.persistence.hedge_recovery import (
    LedgerRecoveryCoordinator,
    RecoveryReport,
)

logger = logging.getLogger(__name__)

_SUPPORTED_DIALECTS = frozenset({"sqlite", "postgresql"})


@dataclass(frozen=True, slots=True)
class HedgeMigrationPlan:
    enabled: bool
    dialect: str
    pending: bool
    existing_tables: tuple[str, ...]
    planned_changes: tuple[str, ...]
    backup_required: bool


@dataclass(frozen=True, slots=True)
class HedgeBootstrapReport:
    migration: HedgeMigrationReport
    recovered_accounts: int
    plan: HedgeMigrationPlan


@dataclass(frozen=True, slots=True)
class HedgePersistenceBootstrapReport:
    """Compatibility report for the pre-feature-gate bootstrap API."""

    migration: HedgeMigrationReport
    recovery: RecoveryReport | None


def _hedge_enabled(config: Mapping[str, Any]) -> bool:
    enabled = config.get("hedge_mode_enabled", False)
    if not isinstance(enabled, bool):
        raise OperationalException("hedge_mode_enabled must be a boolean")
    return enabled


def _dialect(engine: Engine) -> str:
    name = str(getattr(getattr(engine, "dialect", None), "name", "")).strip().lower()
    if name not in _SUPPORTED_DIALECTS:
        raise OperationalException(
            f"Hedge schema supports SQLite and PostgreSQL only, got {name!r}."
        )
    return name


def _additive_core_column_plan(engine: Engine) -> tuple[str, ...]:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    planned: list[str] = []
    if "trades" in tables:
        columns = {column["name"] for column in inspector.get_columns("trades")}
        if "position_side" not in columns:
            planned.append("trades.position_side")
    if "orders" in tables:
        columns = {column["name"] for column in inspector.get_columns("orders")}
        for name in ("position_side", "position_action", "action_group_id"):
            if name not in columns:
                planned.append(f"orders.{name}")
    return tuple(planned)


def plan_hedge_migration(
    engine: Engine,
    config: Mapping[str, Any],
) -> HedgeMigrationPlan:
    """Return a non-mutating migration plan."""

    enabled = _hedge_enabled(config)
    dialect = _dialect(engine)
    tables = tuple(sorted(inspect(engine).get_table_names()))
    if not enabled:
        return HedgeMigrationPlan(
            enabled=False,
            dialect=dialect,
            pending=False,
            existing_tables=tables,
            planned_changes=(),
            backup_required=False,
        )

    additive = _additive_core_column_plan(engine)
    pending = hedge_migrations_pending(engine)
    changes = (*additive, *(() if not pending else ("H3 ledger migrations",)))
    return HedgeMigrationPlan(
        enabled=True,
        dialect=dialect,
        pending=bool(changes),
        existing_tables=tables,
        planned_changes=tuple(changes),
        backup_required=bool(changes),
    )


def migrate_hedge_compatibility_columns(engine: Engine) -> tuple[str, ...]:
    """Add only legacy compatibility columns required by released H3 steps.

    New Hedge facts live in dedicated ``hedge_*`` tables. These columns are kept
    solely for databases already using the v1.x compatibility model and must not
    be expanded further.
    """

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    statements: list[str] = []
    applied: list[str] = []

    if "trades" in tables:
        columns = {column["name"] for column in inspector.get_columns("trades")}
        if "position_side" not in columns:
            statements.append(
                "ALTER TABLE trades ADD COLUMN position_side VARCHAR(10) "
                "NOT NULL DEFAULT 'BOTH'"
            )
            applied.append("trades.position_side")

    if "orders" in tables:
        columns = {column["name"] for column in inspector.get_columns("orders")}
        if "position_side" not in columns:
            statements.append(
                "ALTER TABLE orders ADD COLUMN position_side VARCHAR(10) "
                "NOT NULL DEFAULT 'BOTH'"
            )
            applied.append("orders.position_side")
        if "position_action" not in columns:
            statements.append("ALTER TABLE orders ADD COLUMN position_action VARCHAR(16)")
            applied.append("orders.position_action")
        if "action_group_id" not in columns:
            statements.append("ALTER TABLE orders ADD COLUMN action_group_id VARCHAR(64)")
            applied.append("orders.action_group_id")

    if not statements:
        return ()

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        if "trades" in tables:
            connection.execute(
                text("UPDATE trades SET position_side = 'BOTH' WHERE position_side IS NULL")
            )
        if "orders" in tables:
            connection.execute(
                text("UPDATE orders SET position_side = 'BOTH' WHERE position_side IS NULL")
            )
    return tuple(applied)


def bootstrap_hedge_schema(
    engine: Engine,
    config: Mapping[str, Any],
    *,
    backup_directory: str | Path | None = None,
    fail_after_step: str | None = None,
    lock_timeout_seconds: float = 60.0,
) -> HedgeBootstrapReport:
    """Apply H3 and recover the ledger under an explicit feature gate.

    The caller must validate the complete Hedge configuration before invoking
    this function. Disabled Hedge mode is a hard no-op and performs no inspection
    that creates tables or migration records.
    """

    plan = plan_hedge_migration(engine, config)
    if not plan.enabled:
        empty = HedgeMigrationReport(applied=(), skipped=(), backup_reference=None)
        return HedgeBootstrapReport(migration=empty, recovered_accounts=0, plan=plan)

    backup_reference = (
        prepare_hedge_migration_backup(engine, backup_directory)
        if plan.backup_required
        else None
    )
    compatibility_changes = migrate_hedge_compatibility_columns(engine)
    if compatibility_changes:
        logger.info(
            "Applied explicit Hedge compatibility columns: %s",
            ", ".join(compatibility_changes),
        )

    migration = run_hedge_migrations(
        engine,
        backup_directory=backup_directory,
        fail_after_step=fail_after_step,
        precreated_backup_reference=backup_reference,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    recovery = LedgerRecoveryCoordinator(
        sessionmaker(bind=engine, expire_on_commit=False)
    ).recover_all()
    recovered_accounts = len(recovery.accounts)

    logger.info(
        "Hedge schema bootstrap complete: applied=%s skipped=%s recovered_accounts=%s",
        len(migration.applied),
        len(migration.skipped),
        recovered_accounts,
    )
    return HedgeBootstrapReport(
        migration=migration,
        recovered_accounts=recovered_accounts,
        plan=plan,
    )


def bootstrap_hedge_persistence(
    engine: Engine,
    *,
    session_factory: object | None = None,
    recover: bool = True,
    backup_directory: str | Path | None = None,
    fail_after_step: str | None = None,
    lock_timeout_seconds: float = 60.0,
) -> HedgePersistenceBootstrapReport:
    """Apply H3 and optionally recover the ledger without a config feature gate.

    This preserves the public entry point used by earlier Hedge persistence
    integrations. New application startup code should prefer
    :func:`bootstrap_hedge_schema`, which enforces ``hedge_mode_enabled``.
    """

    _dialect(engine)
    migrate_hedge_compatibility_columns(engine)
    migration = run_hedge_migrations(
        engine,
        backup_directory=backup_directory,
        fail_after_step=fail_after_step,
        lock_timeout_seconds=lock_timeout_seconds,
    )

    recovery_report: RecoveryReport | None = None
    if recover:
        factory = session_factory
        if factory is None:
            factory = sessionmaker(bind=engine, expire_on_commit=False)
        if not callable(factory):
            raise OperationalException("session_factory must be callable")
        recovery_report = LedgerRecoveryCoordinator(factory).recover_all()  # type: ignore[arg-type]

    return HedgePersistenceBootstrapReport(
        migration=migration,
        recovery=recovery_report,
    )


def bootstrap_hedge_session_factory(
    session_factory: object,
    config: Mapping[str, Any],
    *,
    backup_directory: str | Path | None = None,
) -> HedgeBootstrapReport:
    """Resolve an Engine from a SQLAlchemy session factory and bootstrap H3."""

    keyword_args = getattr(session_factory, "kw", None)
    engine = keyword_args.get("bind") if isinstance(keyword_args, Mapping) else None
    if engine is None:
        engine = getattr(session_factory, "bind", None)
    if engine is None:
        try:
            session = session_factory()  # type: ignore[operator]
        except Exception as exc:
            raise OperationalException(
                "Unable to resolve SQLAlchemy engine for Hedge bootstrap."
            ) from exc
        try:
            engine = session.get_bind()
        finally:
            session.close()
    if not isinstance(engine, Engine):
        raise OperationalException("Resolved Hedge database binding is not an Engine.")
    return bootstrap_hedge_schema(
        engine,
        config,
        backup_directory=backup_directory,
    )


def verify_hedge_schema(engine: Engine, config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return immutable migration status after validating the feature gate."""

    if not _hedge_enabled(config):
        return ()
    _dialect(engine)
    return tuple(migration_status(engine))
