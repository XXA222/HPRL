#!/usr/bin/env python3
"""HPRL V3 R3 real-environment acceptance operator.

Every command either produces measured evidence or fails closed.  No command promotes a
PostgreSQL node, creates/drops databases, changes Binance account configuration, or sends
Binance trading writes.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.production.backtest_real_environment import (
    JsonlBacktestEvidenceJournal,
    MeasuredBacktestCommand,
    qualify_r3_two_year_backtest,
    run_measured_backtest_command,
)
from freqtrade.hedge.production.backtest_stability import TwoYearBacktestPolicy
from freqtrade.hedge.production.binance_real_environment import run_binance_r3_real_market_acceptance
from freqtrade.hedge.production.binance_runtime_dryrun import acceptance_probe_targets
from freqtrade.hedge.production.postgres_real_environment import (
    PostgresCliCapability,
    PostgresLogicalBackupReport,
    PostgresNodeIdentity,
    PostgresR3FailoverToken,
    create_postgres_logical_backup,
    prepare_postgres_r3_failover_token,
    probe_postgres_cli,
    restore_postgres_logical_backup,
    run_postgres_r3_core,
    verify_postgres_r3_failover_token,
)
from freqtrade.hedge.production.postgres_runtime_closure import (
    PostgresRestoreSnapshot,
    PostgresSnapshotTable,
)
from freqtrade.hedge.production.risk_behavior import HprlBehaviorPolicy
from freqtrade.hedge.production.risk_behavior_real_environment import (
    JsonlR3BehaviorJournal,
    R3BehaviorObservation,
    qualify_r3_behavior,
)
from freqtrade.hedge.production.runtime_closure import (
    EvidenceState,
    evaluate_runtime_closure_acceptance,
    load_runtime_closure_evidence_registry,
    record_runtime_closure_evidence,
)
from freqtrade.hedge.production.runtime_test_capability import bootstrap_postgres_driver, run_runtime_pytest_suites
from freqtrade.hedge.production.shadow import ShadowMetrics
from freqtrade.hedge.production.shadow_real_environment import (
    JsonlR3ShadowJournal,
    MeasuredR3ShadowCommand,
    R3ShadowWindowEvidence,
    qualify_r3_shadow,
    run_measured_r3_shadow_command,
)
from freqtrade.hedge.production.shadow_runtime import ShadowWindow


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"


def _emit(value: object, output: str = "") -> None:
    text = _json(value)
    if output:
        Path(output).write_text(text, encoding="utf-8")
    print(text, end="")


def _load(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _secret_line(path: str) -> str:
    rows = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"secret file is empty: {path}")
    return rows[0]


def _dsn(args: argparse.Namespace, *, file_attr: str = "dsn_file", env_attr: str = "dsn_env") -> str:
    path = str(getattr(args, file_attr, "") or "").strip()
    if path:
        return _secret_line(path)
    name = str(getattr(args, env_attr, "HPRL_POSTGRES_DSN"))
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing PostgreSQL DSN: --{file_attr.replace('_','-')} or env {name}")
    return value


def _connect_factories(dsn: str):
    import psycopg  # type: ignore
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    connection_factory = lambda: psycopg.connect(dsn)
    sql_dsn = dsn
    if sql_dsn.startswith("postgres://"):
        sql_dsn = "postgresql://" + sql_dsn[len("postgres://"):]
    if sql_dsn.startswith("postgresql://"):
        sql_dsn = "postgresql+psycopg://" + sql_dsn[len("postgresql://"):]
    engine = create_engine(sql_dsn, pool_pre_ping=True, future=True)
    sessions = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return connection_factory, engine, sessions


def cmd_postgres_bootstrap(args: argparse.Namespace) -> int:
    report = bootstrap_postgres_driver(args.root, timeout_seconds=args.timeout)
    _emit(asdict(report), args.output)
    return 0 if report.ready_for_postgres_acceptance else 2



def cmd_r3_pytest(args: argparse.Namespace) -> int:
    report = run_runtime_pytest_suites(
        args.root,
        suites=(
            ("hprl", ("tests/hedge/hprl",), 530),
            ("execution", ("tests/hedge/execution",), 126),
            ("production", ("tests/hedge/production",), 139),
        ),
        timeout_seconds=args.timeout,
        use_xdist=False,
    )
    strict_no_skips = all(item.skipped == 0 for item in report.suites)
    payload = asdict(report) | {
        "passed": report.passed and strict_no_skips,
        "strict_no_skips": strict_no_skips,
        "tests": report.tests,
    }
    _emit(payload, args.output)
    return 0 if payload["passed"] else 2

def cmd_postgres_cli(args: argparse.Namespace) -> int:
    report = probe_postgres_cli()
    _emit(asdict(report) | {"passed": report.passed}, args.output)
    return 0 if report.passed else 2


def cmd_postgres_core(args: argparse.Namespace) -> int:
    dsn = _dsn(args)
    factory, engine, sessions = _connect_factories(dsn)
    try:
        report = run_postgres_r3_core(
            connection_factory=factory,
            session_factory=sessions,
            engine=engine,
            symbol=args.symbol,
            now=datetime.now(UTC),
        )
        _emit(asdict(report) | {"passed": report.passed}, args.output)
        return 0 if report.passed else 2
    finally:
        engine.dispose()


def _snapshot(raw: dict[str, object]) -> PostgresRestoreSnapshot:
    return PostgresRestoreSnapshot(
        database_name=str(raw["database_name"]),
        server_version=str(raw["server_version"]),
        tables=tuple(PostgresSnapshotTable(**row) for row in raw["tables"]),  # type: ignore[arg-type]
        snapshot_sha256=str(raw["snapshot_sha256"]),
        observed_at=datetime.fromisoformat(str(raw["observed_at"])),
    )


def _backup_report(raw: dict[str, object]) -> PostgresLogicalBackupReport:
    return PostgresLogicalBackupReport(
        passed=bool(raw["passed"]),
        archive_path=str(raw["archive_path"]),
        archive_sha256=str(raw["archive_sha256"]),
        archive_bytes=int(raw["archive_bytes"]),
        source_snapshot=_snapshot(raw["source_snapshot"]),  # type: ignore[arg-type]
        cli=PostgresCliCapability(**raw["cli"]),  # type: ignore[arg-type]
        pg_dump_output_tail=str(raw.get("pg_dump_output_tail", "")),
        reasons=tuple(str(x) for x in raw.get("reasons", ())),
        source_snapshot_after=(
            _snapshot(raw["source_snapshot_after"])  # type: ignore[arg-type]
            if isinstance(raw.get("source_snapshot_after"), dict) else None
        ),
        source_stable_during_backup=bool(raw.get("source_stable_during_backup", False)),
        archive_list_verified=bool(raw.get("archive_list_verified", False)),
        pg_restore_list_output_tail=str(raw.get("pg_restore_list_output_tail", "")),
    )


def cmd_postgres_backup(args: argparse.Namespace) -> int:
    dsn = _dsn(args)
    factory, engine, _ = _connect_factories(dsn)
    try:
        report = create_postgres_logical_backup(
            source_dsn=dsn,
            source_connection_factory=factory,
            archive_path=args.archive,
            now=datetime.now(UTC),
            timeout_seconds=args.timeout,
        )
        _emit(asdict(report), args.output)
        return 0 if report.passed else 2
    finally:
        engine.dispose()


def cmd_postgres_restore(args: argparse.Namespace) -> int:
    backup_raw = _load(args.backup_report)
    if not isinstance(backup_raw, dict):
        raise ValueError("backup report must be an object")
    backup = _backup_report(backup_raw)
    target_dsn = _dsn(args, file_attr="target_dsn_file", env_attr="target_dsn_env")
    factory, engine, _ = _connect_factories(target_dsn)
    try:
        report = restore_postgres_logical_backup(
            backup=backup,
            target_dsn=target_dsn,
            target_connection_factory=factory,
            now=datetime.now(UTC),
            timeout_seconds=args.timeout,
        )
        _emit(asdict(report), args.output)
        return 0 if report.passed else 2
    finally:
        engine.dispose()


def _identity(raw: dict[str, object]) -> PostgresNodeIdentity:
    return PostgresNodeIdentity(
        database_name=str(raw["database_name"]), server_addr=str(raw["server_addr"]),
        server_port=None if raw.get("server_port") is None else int(raw["server_port"]),
        backend_pid=int(raw["backend_pid"]), in_recovery=bool(raw["in_recovery"]),
        transaction_read_only=bool(raw["transaction_read_only"]), server_version=str(raw["server_version"]),
        system_identifier=str(raw.get("system_identifier", "")), wal_position=str(raw.get("wal_position", "")),
        observed_at=datetime.fromisoformat(str(raw["observed_at"])),
    )


def _failover_token(raw: dict[str, object]) -> PostgresR3FailoverToken:
    return PostgresR3FailoverToken(
        probe_id=str(raw["probe_id"]), payload_sha256=str(raw["payload_sha256"]),
        schema=str(raw["schema"]), table=str(raw["table"]), primary=_identity(raw["primary"]),  # type: ignore[arg-type]
        writer_lock_key=int(raw["writer_lock_key"]), prepared_at=datetime.fromisoformat(str(raw["prepared_at"])),
    )


def cmd_failover_prepare(args: argparse.Namespace) -> int:
    dsn = _dsn(args)
    factory, engine, _ = _connect_factories(dsn)
    try:
        token = prepare_postgres_r3_failover_token(factory, now=datetime.now(UTC))
        _emit(asdict(token), args.output)
        return 0
    finally:
        engine.dispose()


def cmd_failover_verify(args: argparse.Namespace) -> int:
    token_raw = _load(args.token)
    if not isinstance(token_raw, dict):
        raise ValueError("failover token must be an object")
    token = _failover_token(token_raw)
    routed_dsn = _dsn(args)
    old_dsn = _dsn(args, file_attr="old_primary_dsn_file", env_attr="old_primary_dsn_env")
    routed_factory, routed_engine, _ = _connect_factories(routed_dsn)
    old_factory, old_engine, _ = _connect_factories(old_dsn)
    try:
        report = verify_postgres_r3_failover_token(
            token, routed_factory, old_primary_factory=old_factory, now=datetime.now(UTC),
        )
        _emit(asdict(report), args.output)
        return 0 if report.passed else 2
    finally:
        routed_engine.dispose(); old_engine.dispose()


def _read_targets(path: str, symbol: str) -> tuple[PlannedExecutionIntent, ...]:
    rows: list[PlannedExecutionIntent] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        metadata = {str(k): str(v) for k, v in dict(raw.get("metadata", {})).items()}
        metadata.setdefault("source", "model-target-feed")
        rows.append(PlannedExecutionIntent(
            symbol=str(raw.get("symbol", symbol)),
            target_long_exposure=float(raw["target_long_exposure"]),
            target_short_exposure=float(raw["target_short_exposure"]),
            confidence=float(raw.get("confidence", 1.0)),
            model_id=str(raw.get("model_id", "hprl-r3-model-target")),
            metadata=metadata,
        ))
    if not rows:
        raise ValueError(f"no model targets found in {path}")
    return tuple(rows)


def _credentials(args: argparse.Namespace) -> tuple[str, str]:
    if args.credentials_file:
        rows = [line.strip() for line in Path(args.credentials_file).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(rows) < 2:
            raise ValueError("Binance credential file requires key and secret on first two non-empty lines")
        return rows[0], rows[1]
    key = os.environ.get(args.key_env, "").strip(); secret = os.environ.get(args.secret_env, "").strip()
    if not key or not secret:
        raise RuntimeError("Binance credentials are missing")
    return key, secret


async def _binance(args: argparse.Namespace, *, probe: bool) -> object:
    from freqtrade.hedge.exchange.binance_readonly import AiohttpBinanceRestTransport, BinanceReadonlyClient
    key, secret = _credentials(args)
    proxy = os.environ.get(args.proxy_env, "").strip() or None
    transport = AiohttpBinanceRestTransport(api_key=key, api_secret=secret, proxy_url=proxy, trust_env_proxy=False)
    client = BinanceReadonlyClient(transport=transport, account_id=args.account_id, managed_symbols=(args.symbol,))
    targets = acceptance_probe_targets(args.symbol, args.cycles) if probe else _read_targets(args.targets_jsonl, args.symbol)
    try:
        report = await run_binance_r3_real_market_acceptance(
            client, symbol=args.symbol, targets=targets,
            journal_path=args.journal or None, checkpoint_path=args.checkpoint or None,
            cycle_interval_seconds=args.interval,
            require_model_target_feed=not probe,
        )
        if args.behavior_journal and report.behavior_rows and report.model_target_feed:
            rows = tuple(R3BehaviorObservation(
                cycle_id=item.cycle_id, model_id=item.model_id, target_source=item.target_source,
                target_sha256=item.target_sha256, market_evidence_sha256=item.market_evidence_sha256,
                observation=item.observation,
            ) for item in report.behavior_rows)
            JsonlR3BehaviorJournal(args.behavior_journal).append(rows)
        return report
    finally:
        await transport.close()


def cmd_binance_probe(args: argparse.Namespace) -> int:
    report = asyncio.run(_binance(args, probe=True))
    _emit(asdict(report) | {"passed": report.passed, "production_evidence_eligible": report.production_evidence_eligible}, args.output)
    return 0 if report.passed else 2


def cmd_binance_model(args: argparse.Namespace) -> int:
    if not args.targets_jsonl:
        raise ValueError("binance-model-dryrun requires --targets-jsonl")
    report = asyncio.run(_binance(args, probe=False))
    _emit(asdict(report) | {"passed": report.passed, "production_evidence_eligible": report.production_evidence_eligible}, args.output)
    return 0 if report.passed and report.production_evidence_eligible else 2


def _shadow_metrics(raw: dict[str, object], started: datetime, ended: datetime) -> ShadowMetrics:
    return ShadowMetrics(
        duration=ended - started,
        rest_ws_position_divergences=int(raw.get("rest_ws_position_divergences", 0)),
        unknown_orders_peak=int(raw.get("unknown_orders_peak", 0)),
        unresolved_unknown_orders=int(raw.get("unresolved_unknown_orders", 0)),
        sequence_gaps_unrecovered=int(raw.get("sequence_gaps_unrecovered", 0)),
        candle_gaps_unrecovered=int(raw.get("candle_gaps_unrecovered", 0)),
        duplicate_effects=int(raw.get("duplicate_effects", 0)),
        reconciliation_p99_seconds=float(raw.get("reconciliation_p99_seconds", 0.0)),
        loop_p99_ms=float(raw.get("loop_p99_ms", 0.0)),
        db_p99_ms=float(raw.get("db_p99_ms", 0.0)),
        model_p99_ms=float(raw.get("model_p99_ms", 0.0)),
        model_fallbacks=int(raw.get("model_fallbacks", 0)),
        memory_growth_ratio=float(raw.get("memory_growth_ratio", 0.0)),
        restart_recoveries=int(raw.get("restart_recoveries", 0)),
        restart_recovery_failures=int(raw.get("restart_recovery_failures", 0)),
        funding_cycles_observed=int(raw.get("funding_cycles_observed", 0)),
        planner_churn_ratio=float(raw.get("planner_churn_ratio", 0.0)),
        risk_reject_ratio=float(raw.get("risk_reject_ratio", 0.0)),
    )



def cmd_shadow_run(args: argparse.Namespace) -> int:
    raw = _load(args.command_json)
    if not isinstance(raw, dict) or not isinstance(raw.get("argv"), list):
        raise ValueError("shadow command JSON requires argv[]")
    command = MeasuredR3ShadowCommand(
        argv=tuple(str(x) for x in raw["argv"]),
        cwd=str(raw.get("cwd", args.root)),
        metrics_path=str(raw["metrics_path"]),
        real_market_evidence_path=str(raw["real_market_evidence_path"]),
        behavior_journal_path=str(raw["behavior_journal_path"]),
        source_release=str(raw.get("source_release", args.source_release)),
        model_id=str(raw["model_id"]),
        timeout_seconds=int(raw.get("timeout_seconds", args.timeout)),
        poll_interval_seconds=float(raw.get("poll_interval_seconds", 1.0)),
    )
    journal = JsonlR3ShadowJournal(args.journal, source_release=command.source_release)
    report = run_measured_r3_shadow_command(command, shadow_journal=journal, output_dir=args.run_output_dir)
    _emit(asdict(report), args.output)
    return 0 if report.passed else 2

def cmd_shadow_append(args: argparse.Namespace) -> int:
    raw = _load(args.window_json)
    if not isinstance(raw, dict):
        raise ValueError("shadow window must be an object")
    started = datetime.fromisoformat(str(raw["started_at"])); ended = datetime.fromisoformat(str(raw["ended_at"]))
    metrics_raw = raw.get("metrics", {})
    if not isinstance(metrics_raw, dict):
        raise ValueError("shadow metrics must be an object")
    window = ShadowWindow(
        started_at=started, ended_at=ended, metrics=_shadow_metrics(metrics_raw, started, ended),
        restart_boundary=bool(raw.get("restart_boundary", False)),
        source_cursor_start=int(raw.get("source_cursor_start", 0)), source_cursor_end=int(raw.get("source_cursor_end", 0)),
    )
    evidence = R3ShadowWindowEvidence(
        window=window, source_release=args.source_release, target_source=str(raw.get("target_source", "model-target-feed")),
        model_id=str(raw["model_id"]), model_observations=int(raw["model_observations"]),
        real_market_evidence_sha256=str(raw["real_market_evidence_sha256"]),
        behavior_chain_sha256=str(raw["behavior_chain_sha256"]),
        process_rss_start_bytes=int(raw["process_rss_start_bytes"]), process_rss_end_bytes=int(raw["process_rss_end_bytes"]),
        recorded_at=datetime.now(UTC),
    )
    journal = JsonlR3ShadowJournal(args.journal, source_release=args.source_release)
    record = journal.append(evidence)
    _emit({"passed": True, "sequence": record.sequence, "record_sha256": record.record_sha256}, args.output)
    return 0


def cmd_shadow_qualify(args: argparse.Namespace) -> int:
    report = qualify_r3_shadow(JsonlR3ShadowJournal(args.journal, source_release=args.source_release), target=args.target)
    pending = (not report.passed) and "SOAK_DURATION_INSUFFICIENT" in report.reasons
    _emit(asdict(report) | {"pending": pending}, args.output)
    return 0 if report.passed else (3 if pending else 2)


def cmd_backtest_measure(args: argparse.Namespace) -> int:
    raw = _load(args.command_json)
    if not isinstance(raw, dict) or not isinstance(raw.get("argv"), list):
        raise ValueError("backtest command JSON requires argv[]")
    command = MeasuredBacktestCommand(
        argv=tuple(str(x) for x in raw["argv"]), cwd=str(raw.get("cwd", args.root)),
        started_at=datetime.fromisoformat(str(raw["started_at"])), ended_at=datetime.fromisoformat(str(raw["ended_at"])),
        source_data_path=str(raw["source_data_path"]), result_path=str(raw["result_path"]), metrics_path=str(raw["metrics_path"]),
        timeout_seconds=int(raw.get("timeout_seconds", args.timeout)), poll_interval_seconds=float(raw.get("poll_interval_seconds", 0.2)),
    )
    report = run_measured_backtest_command(command, output_dir=args.run_output_dir)
    if report.chunk is not None:
        JsonlBacktestEvidenceJournal(args.journal).append(report)
    _emit(asdict(report) | {"passed": report.passed}, args.output)
    return 0 if report.passed else 2


def cmd_backtest_qualify(args: argparse.Namespace) -> int:
    policy = TwoYearBacktestPolicy(
        maximum_peak_rss_bytes=args.max_rss_gib * 1024**3,
        maximum_total_elapsed_seconds=args.max_hours * 3600,
    )
    primary = JsonlBacktestEvidenceJournal(args.journal)
    repeat = JsonlBacktestEvidenceJournal(args.repeat_journal)
    report = qualify_r3_two_year_backtest(primary, repeat, policy=policy)
    pending_reasons = {
        "REPEAT_NO_BACKTEST_EVIDENCE",
        "TWO_YEAR_REPEAT_CHUNK_COUNT_MISMATCH",
        "TWO_YEAR_DETERMINISTIC_REPEAT_SEMANTIC_MISMATCH",
    }
    pending = (
        not report.passed
        and not repeat.path.exists()
        and set(report.reasons).issubset(pending_reasons)
    )
    _emit(asdict(report) | {"pending": pending}, args.output)
    return 0 if report.passed else (3 if pending else 2)


def cmd_behavior_qualify(args: argparse.Namespace) -> int:
    policy = HprlBehaviorPolicy(minimum_observations=args.minimum_observations)
    report = qualify_r3_behavior(JsonlR3BehaviorJournal(args.journal), policy=policy)
    _emit(asdict(report), args.output)
    return 0 if report.passed else 2


def cmd_registry_record(args: argparse.Namespace) -> int:
    digest = ""
    if args.evidence_file:
        path = Path(args.evidence_file)
        digest = sha256(path.read_bytes()).hexdigest()
    state = EvidenceState(args.state)
    if state in {EvidenceState.PASS, EvidenceState.FAIL} and not digest:
        raise ValueError("PASS/FAIL registry record requires --evidence-file")
    reg = record_runtime_closure_evidence(args.registry, name=args.name, state=state, digest=digest, detail=args.detail)
    _emit({"passed": True, "registry_sha256": reg, "name": args.name, "state": state.value, "evidence_sha256": digest}, args.output)
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    evidence = load_runtime_closure_evidence_registry(args.registry)
    report = evaluate_runtime_closure_acceptance(evidence)
    _emit(asdict(report) | {"passed": report.passed}, args.output)
    return 0 if report.passed else 2


def _add_dsn(x: argparse.ArgumentParser) -> None:
    x.add_argument("--dsn-file", default=""); x.add_argument("--dsn-env", default="HPRL_POSTGRES_DSN")


def _add_binance(x: argparse.ArgumentParser) -> None:
    x.add_argument("--symbol", default="BTC/USDT:USDT"); x.add_argument("--account-id", default="binance-runtime-closure-r3")
    x.add_argument("--credentials-file", default=""); x.add_argument("--key-env", default="HPRL_BINANCE_API_KEY"); x.add_argument("--secret-env", default="HPRL_BINANCE_API_SECRET"); x.add_argument("--proxy-env", default="HPRL_PROXY_URL")
    x.add_argument("--cycles", type=int, default=100); x.add_argument("--interval", type=float, default=0.0)
    x.add_argument("--targets-jsonl", default=""); x.add_argument("--journal", default=""); x.add_argument("--checkpoint", default=""); x.add_argument("--behavior-journal", default="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False); common.add_argument("--output", default="")

    x = sub.add_parser("postgres-bootstrap", parents=[common]); x.add_argument("--root", default=str(ROOT)); x.add_argument("--timeout", type=int, default=1200); x.set_defaults(func=cmd_postgres_bootstrap)
    x = sub.add_parser("pytest", parents=[common]); x.add_argument("--root", default=str(ROOT)); x.add_argument("--timeout", type=int, default=3600); x.set_defaults(func=cmd_r3_pytest)
    x = sub.add_parser("postgres-cli", parents=[common]); x.set_defaults(func=cmd_postgres_cli)
    x = sub.add_parser("postgres-core", parents=[common]); _add_dsn(x); x.add_argument("--symbol", default="BTCUSDT"); x.set_defaults(func=cmd_postgres_core)
    x = sub.add_parser("postgres-backup", parents=[common]); _add_dsn(x); x.add_argument("--archive", required=True); x.add_argument("--timeout", type=int, default=1800); x.set_defaults(func=cmd_postgres_backup)
    x = sub.add_parser("postgres-restore", parents=[common]); x.add_argument("--backup-report", required=True); x.add_argument("--target-dsn-file", default=""); x.add_argument("--target-dsn-env", default="HPRL_POSTGRES_RESTORE_DSN"); x.add_argument("--timeout", type=int, default=1800); x.set_defaults(func=cmd_postgres_restore)
    x = sub.add_parser("postgres-failover-prepare", parents=[common]); _add_dsn(x); x.set_defaults(func=cmd_failover_prepare)
    x = sub.add_parser("postgres-failover-verify", parents=[common]); _add_dsn(x); x.add_argument("--old-primary-dsn-file", default=""); x.add_argument("--old-primary-dsn-env", default="HPRL_POSTGRES_OLD_PRIMARY_DSN"); x.add_argument("--token", required=True); x.set_defaults(func=cmd_failover_verify)

    x = sub.add_parser("binance-probe", parents=[common]); _add_binance(x); x.set_defaults(func=cmd_binance_probe)
    x = sub.add_parser("binance-model-dryrun", parents=[common]); _add_binance(x); x.set_defaults(func=cmd_binance_model)

    x = sub.add_parser("shadow-run", parents=[common]); x.add_argument("--command-json", required=True); x.add_argument("--journal", required=True); x.add_argument("--run-output-dir", required=True); x.add_argument("--root", default=str(ROOT)); x.add_argument("--source-release", default="freqtrade-hedge-hprl-v3-real-environment-r3"); x.add_argument("--timeout", type=int, default=46800); x.set_defaults(func=cmd_shadow_run)
    x = sub.add_parser("shadow-append", parents=[common]); x.add_argument("--journal", required=True); x.add_argument("--window-json", required=True); x.add_argument("--source-release", default="freqtrade-hedge-hprl-v3-real-environment-r3"); x.set_defaults(func=cmd_shadow_append)
    x = sub.add_parser("shadow-qualify", parents=[common]); x.add_argument("--journal", required=True); x.add_argument("--target", choices=("24h", "72h"), required=True); x.add_argument("--source-release", default="freqtrade-hedge-hprl-v3-real-environment-r3"); x.set_defaults(func=cmd_shadow_qualify)

    x = sub.add_parser("backtest-measure", parents=[common]); x.add_argument("--command-json", required=True); x.add_argument("--journal", required=True); x.add_argument("--run-output-dir", required=True); x.add_argument("--root", default=str(ROOT)); x.add_argument("--timeout", type=int, default=21600); x.set_defaults(func=cmd_backtest_measure)
    x = sub.add_parser("backtest-qualify", parents=[common]); x.add_argument("--journal", required=True); x.add_argument("--repeat-journal", required=True); x.add_argument("--max-rss-gib", type=int, default=12); x.add_argument("--max-hours", type=float, default=6.0); x.set_defaults(func=cmd_backtest_qualify)
    x = sub.add_parser("behavior-qualify", parents=[common]); x.add_argument("--journal", required=True); x.add_argument("--minimum-observations", type=int, default=10000); x.set_defaults(func=cmd_behavior_qualify)

    x = sub.add_parser("registry-record", parents=[common]); x.add_argument("--registry", required=True); x.add_argument("--name", required=True, choices=("container_pytest","postgres_core","postgres_failover","postgres_restore","binance_real_market_dryrun","fault_campaign","shadow_24h","shadow_72h","two_year_backtest","position_behavior")); x.add_argument("--state", required=True, choices=("PASS","FAIL","PENDING")); x.add_argument("--evidence-file", default=""); x.add_argument("--detail", default=""); x.set_defaults(func=cmd_registry_record)
    x = sub.add_parser("acceptance", parents=[common]); x.add_argument("--registry", required=True); x.set_defaults(func=cmd_acceptance)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
