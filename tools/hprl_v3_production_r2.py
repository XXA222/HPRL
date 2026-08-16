#!/usr/bin/env python3
"""Operator CLI for HPRL V3 Production Integration R2.

The CLI is deliberately split between offline/source checks and environment evidence.
No command enables live exchange writes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.production.binance_dryrun import (
    BinanceDryRunPolicy, BinanceDryRunSafetyContext, evaluate_binance_dryrun,
)
from freqtrade.hedge.production.database_runtime import PostgresConcurrencyProbeRunner, PostgresProbeRunner
from freqtrade.hedge.production.postgres_acceptance import PostgresDurabilityProbeRunner
from freqtrade.hedge.production.source_convergence import build_canonical_source_snapshot
from freqtrade.hedge.telemetry.dryrun import JsonlDryRunTelemetryStore
from freqtrade.hedge.production.acceptance_r2 import HPRL_V3_PRODUCTION_RELEASE


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _write_or_print(payload: dict[str, object], output: str) -> None:
    text = _json(payload) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    print(text, end="")


def _connect_factory(dsn: str):
    try:
        import psycopg  # type: ignore
        return lambda: psycopg.connect(dsn)
    except ImportError:
        try:
            import psycopg2  # type: ignore
            return lambda: psycopg2.connect(dsn)
        except ImportError as exc:
            raise RuntimeError("psycopg or psycopg2 is required for PostgreSQL environment probes") from exc


def cmd_source(args: argparse.Namespace) -> int:
    snapshot = build_canonical_source_snapshot(args.root)
    payload = {"schema": "hprl-v3-production-r2-source-snapshot-v1", **asdict(snapshot), "passed": snapshot.passed}
    _write_or_print(payload, args.output)
    return 0 if snapshot.passed else 2


def cmd_dryrun(args: argparse.Namespace) -> int:
    path = Path(args.telemetry)
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    capacity = max(10, min(100000, lines + 10))
    store = JsonlDryRunTelemetryStore(path, capacity=capacity)
    items = store.list(capacity)
    safety = BinanceDryRunSafetyContext(
        exchange="binance", operation_mode="dry_run", real_market_data=True,
        exchange_write_capability=False, simulated_execution=True,
        hedge_mode_semantics=True, cross_margin_semantics=True,
        source_release=HPRL_V3_PRODUCTION_RELEASE, account_namespace=args.account_namespace,
    )
    policy = BinanceDryRunPolicy(
        minimum_cycles=args.minimum_cycles,
        minimum_duration=timedelta(minutes=args.minimum_minutes),
        maximum_cycle_gap=timedelta(seconds=args.maximum_gap_seconds),
        require_dual_leg_target=not args.allow_single_leg_only,
        maximum_risk_block_ratio=Decimal(str(args.maximum_risk_block_ratio)),
    )
    report = evaluate_binance_dryrun(items, safety=safety, policy=policy)
    payload = {
        "schema": "hprl-v3-production-r2-binance-dryrun-evidence-v1",
        "safety": asdict(safety), "policy": asdict(policy), "report": asdict(report),
        "passed": report.passed,
        "note": "Real Binance market data with simulated execution only; exchange write capability is false.",
    }
    _write_or_print(payload, args.output)
    return 0 if report.passed else 2


def cmd_postgres(args: argparse.Namespace) -> int:
    dsn = os.environ.get(args.dsn_env, "").strip()
    if not dsn:
        raise RuntimeError(f"PostgreSQL DSN environment variable is missing: {args.dsn_env}")
    factory = _connect_factory(dsn)
    now = datetime.now(UTC)
    primary = factory()
    try:
        basic = PostgresProbeRunner(primary).run(now=now)
    finally:
        close = getattr(primary, "close", None)
        if callable(close): close()
    concurrency = PostgresConcurrencyProbeRunner(factory).run(now=now)
    durability = PostgresDurabilityProbeRunner(factory, factory).run(now=now)
    passed = basic.passed and concurrency.passed and durability.passed
    payload = {
        "schema": "hprl-v3-production-r2-postgres-runtime-probes-v1",
        "observed_at": now.isoformat(), "dsn_env": args.dsn_env,
        "basic": asdict(basic), "concurrency": asdict(concurrency), "durability": asdict(durability),
        "passed": passed,
        "full_r2_postgres_ready": False,
        "remaining_required_evidence": ["controlled failover exercise", "isolated pg_dump/pg_restore verification"],
    }
    _write_or_print(payload, args.output)
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source-snapshot")
    source.add_argument("--root", default=str(ROOT)); source.add_argument("--output", default=""); source.set_defaults(func=cmd_source)
    dry = sub.add_parser("binance-dryrun")
    dry.add_argument("--telemetry", required=True); dry.add_argument("--output", default="")
    dry.add_argument("--account-namespace", default="dryrun"); dry.add_argument("--minimum-cycles", type=int, default=100)
    dry.add_argument("--minimum-minutes", type=int, default=30); dry.add_argument("--maximum-gap-seconds", type=int, default=300)
    dry.add_argument("--maximum-risk-block-ratio", default="0.25"); dry.add_argument("--allow-single-leg-only", action="store_true")
    dry.set_defaults(func=cmd_dryrun)
    pg = sub.add_parser("postgres-probes")
    pg.add_argument("--dsn-env", default="HPRL_POSTGRES_DSN"); pg.add_argument("--output", default=""); pg.set_defaults(func=cmd_postgres)
    args = parser.parse_args()
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
