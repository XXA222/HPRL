#!/usr/bin/env python3
"""HPRL V3 Runtime Closure R2 operator CLI.

No command in this utility enables Binance exchange writes.  Real Binance commands build
only ``BinanceReadonlyClient`` and use local simulated execution.
"""
from __future__ import annotations

import argparse
import asyncio
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

from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.production.backtest_stability import (
    BacktestChunkEvidence, TwoYearBacktestPolicy, evaluate_two_year_backtest_stability,
)
from freqtrade.hedge.production.binance_runtime_dryrun import (
    acceptance_probe_targets, run_binance_real_market_dryrun,
)
from freqtrade.hedge.production.postgres_runtime_closure import (
    PostgresFailoverToken, PostgresRuntimeClosureRunner,
    capture_postgres_restore_snapshot, prepare_postgres_failover_token,
    verify_postgres_failover_token, verify_postgres_restore,
)
from freqtrade.hedge.production.risk_behavior import (
    HprlBehaviorObservation, HprlBehaviorPolicy, analyze_hprl_position_behavior,
)
from freqtrade.hedge.production.runtime_closure import (
    EvidenceState, RuntimeClosureEvidence, RuntimeClosurePolicy,
    evaluate_runtime_closure_acceptance, initialize_runtime_closure_evidence_registry,
    load_runtime_closure_evidence_registry, record_runtime_closure_evidence,
)
from freqtrade.hedge.production.runtime_fault_injection import run_focused_runtime_fault_campaign
from freqtrade.hedge.production.runtime_test_capability import (
    bootstrap_postgres_driver, bootstrap_runtime_test_dependencies, probe_runtime_test_capability,
    report_json, run_runtime_pytest_suites,
)
from freqtrade.hedge.production.shadow import ShadowMetrics
from freqtrade.hedge.production.shadow_runtime import ShadowWindow
from freqtrade.hedge.production.shadow_soak_runtime import JsonlShadowWindowJournal


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"


def _emit(value: object, output: str = "") -> None:
    text = _json(value)
    if output:
        Path(output).write_text(text, encoding="utf-8")
    print(text, end="")


def _load_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _connect_factories(dsn: str):
    driver = ""
    try:
        import psycopg  # type: ignore
        connection_factory = lambda: psycopg.connect(dsn)
        driver = "psycopg"
    except ImportError:
        try:
            import psycopg2  # type: ignore
            connection_factory = lambda: psycopg2.connect(dsn)
            driver = "psycopg2"
        except ImportError as exc:
            raise RuntimeError("psycopg or psycopg2 is required for PostgreSQL runtime closure") from exc
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    sqlalchemy_dsn = dsn
    if dsn.startswith("postgres://"):
        sqlalchemy_dsn = "postgresql://" + dsn[len("postgres://"):]
    if sqlalchemy_dsn.startswith("postgresql://"):
        sqlalchemy_dsn = f"postgresql+{driver}://" + sqlalchemy_dsn[len("postgresql://"):]
    engine = create_engine(sqlalchemy_dsn, pool_pre_ping=True, future=True)
    sessions = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return connection_factory, engine, sessions


def cmd_test_capability(args: argparse.Namespace) -> int:
    report = probe_runtime_test_capability(args.root)
    _emit(asdict(report), args.output)
    return 0 if report.ready_for_serial_pytest else 2


def cmd_bootstrap_tests(args: argparse.Namespace) -> int:
    report = bootstrap_runtime_test_dependencies(
        args.root, include_all_test_plugins=not args.pytest_only, timeout_seconds=args.timeout,
    )
    _emit(asdict(report), args.output)
    return 0 if report.ready_for_serial_pytest else 2


def cmd_run_tests(args: argparse.Namespace) -> int:
    report = run_runtime_pytest_suites(
        args.root, timeout_seconds=args.timeout, use_xdist=args.xdist,
    )
    _emit(asdict(report) | {"passed": report.passed, "tests": report.tests}, args.output)
    return 0 if report.passed else 2


def cmd_bootstrap_postgres(args: argparse.Namespace) -> int:
    report = bootstrap_postgres_driver(args.root, timeout_seconds=args.timeout)
    _emit(asdict(report), args.output)
    return 0 if report.ready_for_postgres_acceptance else 2


def _read_secret_file(path: str, *, lines: int = 1) -> tuple[str, ...]:
    if not path:
        return ()
    values = tuple(line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if len(values) < lines:
        raise RuntimeError(f"secret file does not contain {lines} non-empty line(s): {path}")
    return values[:lines]


def _postgres_dsn(args: argparse.Namespace) -> str:
    dsn_file = str(getattr(args, "dsn_file", "") or "").strip()
    if dsn_file:
        return _read_secret_file(dsn_file, lines=1)[0]
    dsn = os.environ.get(args.dsn_env, "").strip()
    if not dsn:
        raise RuntimeError(f"missing PostgreSQL DSN: provide --dsn-file or env {args.dsn_env}")
    return dsn


def cmd_postgres_core(args: argparse.Namespace) -> int:
    factory, engine, sessions = _connect_factories(_postgres_dsn(args))
    try:
        report = PostgresRuntimeClosureRunner(
            connection_factory=factory, session_factory=sessions, engine=engine,
            symbol=args.symbol,
        ).run(now=datetime.now(UTC))
        _emit(asdict(report) | {"passed": report.passed}, args.output)
        return 0 if report.passed else 2
    finally:
        engine.dispose()


def cmd_postgres_snapshot(args: argparse.Namespace) -> int:
    factory, engine, _sessions = _connect_factories(_postgres_dsn(args))
    try:
        report = capture_postgres_restore_snapshot(factory, now=datetime.now(UTC))
        _emit(asdict(report), args.output)
        return 0
    finally:
        engine.dispose()


def _snapshot_from(raw: dict[str, object]):
    from freqtrade.hedge.production.postgres_runtime_closure import PostgresRestoreSnapshot, PostgresSnapshotTable
    return PostgresRestoreSnapshot(
        database_name=str(raw["database_name"]), server_version=str(raw["server_version"]),
        tables=tuple(PostgresSnapshotTable(**item) for item in raw["tables"]),  # type: ignore[arg-type]
        snapshot_sha256=str(raw["snapshot_sha256"]),
        observed_at=datetime.fromisoformat(str(raw["observed_at"])),
    )


def cmd_postgres_verify_restore(args: argparse.Namespace) -> int:
    source = _snapshot_from(_load_json(args.source_snapshot))  # type: ignore[arg-type]
    factory, engine, _sessions = _connect_factories(_postgres_dsn(args))
    try:
        restored = capture_postgres_restore_snapshot(factory, now=datetime.now(UTC))
        report = verify_postgres_restore(source, restored)
        _emit(asdict(report), args.output)
        return 0 if report.passed else 2
    finally:
        engine.dispose()


def cmd_postgres_failover_prepare(args: argparse.Namespace) -> int:
    factory, engine, _sessions = _connect_factories(_postgres_dsn(args))
    try:
        token = prepare_postgres_failover_token(factory, now=datetime.now(UTC))
        _emit(asdict(token), args.output)
        return 0
    finally:
        engine.dispose()


def _failover_token(raw: dict[str, object]) -> PostgresFailoverToken:
    return PostgresFailoverToken(
        probe_id=str(raw["probe_id"]), payload_sha256=str(raw["payload_sha256"]),
        schema=str(raw["schema"]), table=str(raw["table"]), database_name=str(raw["database_name"]),
        primary_backend_pid=int(raw["primary_backend_pid"]),
        prepared_at=datetime.fromisoformat(str(raw["prepared_at"])),
    )


def cmd_postgres_failover_verify(args: argparse.Namespace) -> int:
    token = _failover_token(_load_json(args.token))  # type: ignore[arg-type]
    routed_dsn = _postgres_dsn(args)
    routed, routed_engine, _ = _connect_factories(routed_dsn)
    old_factory = None
    old_engine = None
    old_file = str(getattr(args, "old_primary_dsn_file", "") or "").strip()
    old_dsn = _read_secret_file(old_file, lines=1)[0] if old_file else os.environ.get(args.old_primary_dsn_env, "").strip()
    if old_dsn:
        old_factory, old_engine, _ = _connect_factories(old_dsn)
    try:
        report = verify_postgres_failover_token(
            token, routed, now=datetime.now(UTC), old_primary_factory=old_factory,
        )
        _emit(asdict(report) | {"passed": report.passed}, args.output)
        return 0 if report.passed else 2
    finally:
        routed_engine.dispose()
        if old_engine is not None:
            old_engine.dispose()


def _load_targets(path: str, symbol: str) -> tuple[PlannedExecutionIntent, ...]:
    if not path:
        return ()
    rows: list[PlannedExecutionIntent] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(PlannedExecutionIntent(
            symbol=str(raw.get("symbol", symbol)),
            target_long_exposure=float(raw["target_long_exposure"]),
            target_short_exposure=float(raw["target_short_exposure"]),
            confidence=float(raw.get("confidence", 1.0)),
            model_id=str(raw.get("model_id", "hprl-target-feed")),
            metadata={str(k): str(v) for k, v in dict(raw.get("metadata", {})).items()},
        ))
    return tuple(rows)


async def _binance_dryrun(args: argparse.Namespace):
    from freqtrade.hedge.exchange.binance_readonly import AiohttpBinanceRestTransport, BinanceReadonlyClient
    credential_file = str(getattr(args, "credentials_file", "") or "").strip()
    if credential_file:
        key, secret = _read_secret_file(credential_file, lines=2)
    else:
        key = os.environ.get(args.key_env, "").strip()
        secret = os.environ.get(args.secret_env, "").strip()
    if not key or not secret:
        raise RuntimeError("Binance credentials are missing: provide --credentials-file or configured environment variables")
    proxy = os.environ.get(args.proxy_env, "").strip() or None
    transport = AiohttpBinanceRestTransport(
        api_key=key, api_secret=secret, proxy_url=proxy, trust_env_proxy=False,
    )
    client = BinanceReadonlyClient(
        transport=transport,
        account_id=args.account_id,
        managed_symbols=(args.symbol,),
    )
    targets = _load_targets(args.targets_jsonl, args.symbol)
    source = "model-target-feed"
    if not targets:
        targets = acceptance_probe_targets(args.symbol, args.cycles)
        source = "acceptance-probe"
    try:
        return await run_binance_real_market_dryrun(
            client,
            symbol=args.symbol,
            targets=targets,
            journal_path=args.journal or None,
            checkpoint_path=args.checkpoint or None,
            cycle_interval_seconds=args.interval,
            source=source,
        )
    finally:
        await transport.close()


def cmd_binance_real_dryrun(args: argparse.Namespace) -> int:
    report = asyncio.run(_binance_dryrun(args))
    _emit(asdict(report) | {"passed": report.passed}, args.output)
    return 0 if report.passed else 2


def cmd_fault_campaign(args: argparse.Namespace) -> int:
    report = run_focused_runtime_fault_campaign()
    _emit(asdict(report) | {"passed": report.passed}, args.output)
    return 0 if report.passed else 2



def _shadow_window_from_json(raw: dict[str, object]) -> ShadowWindow:
    metrics_raw = raw.get("metrics")
    if not isinstance(metrics_raw, dict):
        raise ValueError("shadow window evidence must contain metrics{}")
    started_at = datetime.fromisoformat(str(raw["started_at"]))
    ended_at = datetime.fromisoformat(str(raw["ended_at"]))
    duration_seconds = float(metrics_raw.get("duration_seconds", (ended_at - started_at).total_seconds()))
    metrics = ShadowMetrics(
        duration=timedelta(seconds=duration_seconds),
        rest_ws_position_divergences=int(metrics_raw.get("rest_ws_position_divergences", 0)),
        unknown_orders_peak=int(metrics_raw.get("unknown_orders_peak", 0)),
        unresolved_unknown_orders=int(metrics_raw.get("unresolved_unknown_orders", 0)),
        sequence_gaps_unrecovered=int(metrics_raw.get("sequence_gaps_unrecovered", 0)),
        candle_gaps_unrecovered=int(metrics_raw.get("candle_gaps_unrecovered", 0)),
        duplicate_effects=int(metrics_raw.get("duplicate_effects", 0)),
        reconciliation_p99_seconds=float(metrics_raw.get("reconciliation_p99_seconds", 0.0)),
        loop_p99_ms=float(metrics_raw.get("loop_p99_ms", 0.0)),
        db_p99_ms=float(metrics_raw.get("db_p99_ms", 0.0)),
        model_p99_ms=float(metrics_raw.get("model_p99_ms", 0.0)),
        model_fallbacks=int(metrics_raw.get("model_fallbacks", 0)),
        memory_growth_ratio=float(metrics_raw.get("memory_growth_ratio", 0.0)),
        restart_recoveries=int(metrics_raw.get("restart_recoveries", 0)),
        restart_recovery_failures=int(metrics_raw.get("restart_recovery_failures", 0)),
        funding_cycles_observed=int(metrics_raw.get("funding_cycles_observed", 0)),
        planner_churn_ratio=float(metrics_raw.get("planner_churn_ratio", 0.0)),
        risk_reject_ratio=float(metrics_raw.get("risk_reject_ratio", 0.0)),
    )
    return ShadowWindow(
        started_at=started_at, ended_at=ended_at, metrics=metrics,
        restart_boundary=bool(raw.get("restart_boundary", False)),
        source_cursor_start=int(raw.get("source_cursor_start", 0)),
        source_cursor_end=int(raw.get("source_cursor_end", 0)),
    )


def cmd_shadow_append(args: argparse.Namespace) -> int:
    raw = _load_json(args.window_json)
    if not isinstance(raw, dict):
        raise ValueError("shadow window evidence must be a JSON object")
    window = _shadow_window_from_json(raw)
    journal = JsonlShadowWindowJournal(args.journal, source_release=args.source_release)
    record = journal.append(window, observed_at=datetime.now(UTC))
    state = journal.load()
    payload = {
        "record": record.payload(),
        "record_sha256": record.record_sha256,
        "journal_valid": state.valid,
        "journal_tip_sha256": state.tip_sha256,
        "records": len(state.records),
        "passed": state.valid,
    }
    _emit(payload, args.output)
    return 0 if state.valid else 2

def cmd_shadow_qualify(args: argparse.Namespace) -> int:
    journal = JsonlShadowWindowJournal(args.journal, source_release=args.source_release)
    state = journal.load()
    qualification = journal.qualify(target=args.target)
    payload = {
        "journal_valid": state.valid, "journal_reasons": state.reasons,
        "journal_tip_sha256": state.tip_sha256, "records": len(state.records),
        "qualification": asdict(qualification),
        "passed": state.valid and qualification.passed,
    }
    _emit(payload, args.output)
    return 0 if payload["passed"] else 2


def _backtest_chunk(raw: dict[str, object]) -> BacktestChunkEvidence:
    return BacktestChunkEvidence(
        started_at=datetime.fromisoformat(str(raw["started_at"])),
        ended_at=datetime.fromisoformat(str(raw["ended_at"])),
        bars=int(raw["bars"]), events=int(raw["events"]),
        elapsed_seconds=float(raw["elapsed_seconds"]), peak_rss_bytes=int(raw["peak_rss_bytes"]),
        exit_code=int(raw["exit_code"]), result_sha256=str(raw["result_sha256"]),
        source_data_sha256=str(raw["source_data_sha256"]),
    )


def cmd_backtest_2y(args: argparse.Namespace) -> int:
    raw = _load_json(args.evidence)
    if not isinstance(raw, dict) or not isinstance(raw.get("chunks"), list):
        raise ValueError("backtest evidence must contain chunks[]")
    chunks = tuple(_backtest_chunk(item) for item in raw["chunks"])
    policy = TwoYearBacktestPolicy(
        maximum_peak_rss_bytes=args.max_rss_gib * 1024**3,
        maximum_total_elapsed_seconds=args.max_hours * 3600,
    )
    report = evaluate_two_year_backtest_stability(
        chunks, repeat_result_sha256=str(raw.get("repeat_result_sha256") or "") or None,
        policy=policy,
    )
    _emit(asdict(report), args.output)
    return 0 if report.passed else 2


def cmd_behavior(args: argparse.Namespace) -> int:
    rows: list[HprlBehaviorObservation] = []
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        raw = json.loads(line)
        rows.append(HprlBehaviorObservation(
            timestamp=datetime.fromisoformat(str(raw["timestamp"])),
            long_margin_ratio=Decimal(str(raw["long_margin_ratio"])),
            short_margin_ratio=Decimal(str(raw["short_margin_ratio"])),
            equity_return=float(raw.get("equity_return", 0.0)),
            drawdown=float(raw.get("drawdown", 0.0)),
            uncertainty=float(raw.get("uncertainty", 0.0)),
        ))
    report = analyze_hprl_position_behavior(rows, policy=HprlBehaviorPolicy(minimum_observations=args.minimum_observations))
    _emit(asdict(report), args.output)
    return 0 if report.passed else 2



def _sha256_path(path: str) -> str:
    from hashlib import sha256
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_evidence_init(args: argparse.Namespace) -> int:
    digest = initialize_runtime_closure_evidence_registry(args.registry)
    evidence = load_runtime_closure_evidence_registry(args.registry)
    _emit({
        "registry": str(Path(args.registry)),
        "registry_sha256": digest,
        "states": {name: item.state.value for name, item in evidence.items()},
        "passed": True,
    }, args.output)
    return 0


def cmd_evidence_record(args: argparse.Namespace) -> int:
    evidence_digest = ""
    evidence_file = str(args.evidence_file or "").strip()
    if evidence_file:
        evidence_digest = _sha256_path(evidence_file)
    state = EvidenceState(args.state)
    if state in {EvidenceState.PASS, EvidenceState.FAIL} and not evidence_digest:
        raise ValueError("PASS/FAIL evidence recording requires --evidence-file")
    registry_digest = record_runtime_closure_evidence(
        args.registry, name=args.name, state=state,
        digest=evidence_digest, detail=args.detail,
    )
    _emit({
        "name": args.name, "state": state.value,
        "evidence_sha256": evidence_digest,
        "registry_sha256": registry_digest,
        "passed": True,
    }, args.output)
    return 0

def cmd_acceptance(args: argparse.Namespace) -> int:
    raw = _load_json(args.evidence)
    if not isinstance(raw, dict):
        raise ValueError("acceptance evidence must be a JSON object")
    if raw.get("schema") == "hprl-runtime-closure-r2-evidence-registry-v1":
        evidence = load_runtime_closure_evidence_registry(args.evidence)
    else:
        evidence = {
            name: RuntimeClosureEvidence(
                name=name,
                state=EvidenceState(str(item.get("state", "PENDING"))),
                digest=str(item.get("digest", "")), detail=str(item.get("detail", "")),
            )
            for name, item in raw.items() if isinstance(item, dict)
        }
    report = evaluate_runtime_closure_acceptance(evidence, policy=RuntimeClosurePolicy())
    _emit(asdict(report) | {"passed": report.passed}, args.output)
    return 0 if report.passed else 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common_out = argparse.ArgumentParser(add_help=False); common_out.add_argument("--output", default="")

    x=sub.add_parser("test-capability",parents=[common_out]); x.add_argument("--root",default=str(ROOT)); x.set_defaults(func=cmd_test_capability)
    x=sub.add_parser("bootstrap-tests",parents=[common_out]); x.add_argument("--root",default=str(ROOT)); x.add_argument("--pytest-only",action="store_true"); x.add_argument("--timeout",type=int,default=900); x.set_defaults(func=cmd_bootstrap_tests)
    x=sub.add_parser("run-tests",parents=[common_out]); x.add_argument("--root",default=str(ROOT)); x.add_argument("--timeout",type=int,default=3600); x.add_argument("--xdist",action="store_true"); x.set_defaults(func=cmd_run_tests)

    x=sub.add_parser("bootstrap-postgres",parents=[common_out]); x.add_argument("--root",default=str(ROOT)); x.add_argument("--timeout",type=int,default=900); x.set_defaults(func=cmd_bootstrap_postgres)

    for name, func in (("postgres-core",cmd_postgres_core),("postgres-snapshot",cmd_postgres_snapshot)):
        x=sub.add_parser(name,parents=[common_out]); x.add_argument("--dsn-env",default="HPRL_POSTGRES_DSN"); x.add_argument("--dsn-file",default=""); x.add_argument("--symbol",default="BTCUSDT"); x.set_defaults(func=func)
    x=sub.add_parser("postgres-verify-restore",parents=[common_out]); x.add_argument("--dsn-env",default="HPRL_POSTGRES_DSN"); x.add_argument("--dsn-file",default=""); x.add_argument("--source-snapshot",required=True); x.set_defaults(func=cmd_postgres_verify_restore)
    x=sub.add_parser("postgres-failover-prepare",parents=[common_out]); x.add_argument("--dsn-env",default="HPRL_POSTGRES_DSN"); x.add_argument("--dsn-file",default=""); x.set_defaults(func=cmd_postgres_failover_prepare)
    x=sub.add_parser("postgres-failover-verify",parents=[common_out]); x.add_argument("--dsn-env",default="HPRL_POSTGRES_DSN"); x.add_argument("--dsn-file",default=""); x.add_argument("--old-primary-dsn-env",default="HPRL_POSTGRES_OLD_PRIMARY_DSN"); x.add_argument("--old-primary-dsn-file",default=""); x.add_argument("--token",required=True); x.set_defaults(func=cmd_postgres_failover_verify)

    x=sub.add_parser("binance-real-dryrun",parents=[common_out]); x.add_argument("--symbol",default="BTC/USDT:USDT"); x.add_argument("--account-id",default="binance-runtime-closure"); x.add_argument("--cycles",type=int,default=100); x.add_argument("--interval",type=float,default=0.0); x.add_argument("--targets-jsonl",default=""); x.add_argument("--journal",default=""); x.add_argument("--checkpoint",default=""); x.add_argument("--credentials-file",default=""); x.add_argument("--key-env",default="HPRL_BINANCE_API_KEY"); x.add_argument("--secret-env",default="HPRL_BINANCE_API_SECRET"); x.add_argument("--proxy-env",default="HPRL_PROXY_URL"); x.set_defaults(func=cmd_binance_real_dryrun)
    x=sub.add_parser("fault-campaign",parents=[common_out]); x.set_defaults(func=cmd_fault_campaign)
    x=sub.add_parser("shadow-append",parents=[common_out]); x.add_argument("--journal",required=True); x.add_argument("--source-release",default="freqtrade-hedge-hprl-v3-runtime-closure-r2"); x.add_argument("--window-json",required=True); x.set_defaults(func=cmd_shadow_append)
    x=sub.add_parser("shadow-qualify",parents=[common_out]); x.add_argument("--journal",required=True); x.add_argument("--source-release",default="freqtrade-hedge-hprl-v3-runtime-closure-r2"); x.add_argument("--target",choices=("24h","72h"),required=True); x.set_defaults(func=cmd_shadow_qualify)
    x=sub.add_parser("backtest-2y",parents=[common_out]); x.add_argument("--evidence",required=True); x.add_argument("--max-rss-gib",type=int,default=12); x.add_argument("--max-hours",type=float,default=6.0); x.set_defaults(func=cmd_backtest_2y)
    x=sub.add_parser("behavior",parents=[common_out]); x.add_argument("--jsonl",required=True); x.add_argument("--minimum-observations",type=int,default=10000); x.set_defaults(func=cmd_behavior)
    x=sub.add_parser("evidence-init",parents=[common_out]); x.add_argument("--registry",required=True); x.set_defaults(func=cmd_evidence_init)
    x=sub.add_parser("evidence-record",parents=[common_out]); x.add_argument("--registry",required=True); x.add_argument("--name",required=True,choices=("container_pytest","postgres_core","postgres_failover","postgres_restore","binance_real_market_dryrun","fault_campaign","shadow_24h","shadow_72h","two_year_backtest","position_behavior")); x.add_argument("--state",required=True,choices=("PASS","FAIL","PENDING")); x.add_argument("--evidence-file",default=""); x.add_argument("--detail",default=""); x.set_defaults(func=cmd_evidence_record)
    x=sub.add_parser("acceptance",parents=[common_out]); x.add_argument("--evidence",required=True); x.set_defaults(func=cmd_acceptance)
    args=p.parse_args(); return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
