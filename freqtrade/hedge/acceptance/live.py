from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import time_ns
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from freqtrade.hedge.exchange.base import CalibrationKind
from freqtrade.hedge.integration.repository import PersistenceMirroringReadonlyRepository
from freqtrade.hedge.acceptance.acceptance import RuntimeAcceptanceEngine
from freqtrade.hedge.acceptance.clock import ClockSample, evaluate_clock
from freqtrade.hedge.acceptance.events import EventEnvelope
from freqtrade.hedge.acceptance.facts import build_fact_plane
from freqtrade.hedge.acceptance.models import AcceptancePolicy
from freqtrade.hedge.acceptance.persistence import RuntimeAcceptanceStore
from freqtrade.hedge.acceptance.projection import (
    build_database_plane,
    build_memory_plane,
    count_unrecovered_unknown_orders,
)
from freqtrade.hedge.readonly import build_binance_readonly_runtime_from_freqtrade_config
from freqtrade.persistence.hedge_models import HedgeModelBase
from freqtrade.persistence.hedge_service import HedgePersistenceService


logger = logging.getLogger(__name__)


def _require_round_passed(engine: RuntimeAcceptanceEngine) -> None:
    """Surface the first failed round instead of masking it with the next gate."""
    engine.session.require_last_passed()


def _epoch_ms() -> int:
    return time_ns() // 1_000_000


async def _clock_audit(client: Any, policy: AcceptancePolicy, *, sample_count: int = 5):
    samples: list[ClockSample] = []
    for _ in range(sample_count):
        before = _epoch_ms()
        server = await client.fetch_server_time()
        after = _epoch_ms()
        samples.append(ClockSample(before, server, after))
        await asyncio.sleep(0.05)
    return evaluate_clock(
        samples,
        max_abs_skew_ms=policy.max_clock_skew_ms,
        max_rtt_ms=policy.max_clock_rtt_ms,
    )


async def _preflight_signed_access(client: Any, permission_policy: Any) -> Any:
    """Synchronize Binance time before the first signed private request."""
    await client.synchronize_clock()
    return await client.preflight_permissions(permission_policy)


def _synthetic_schema_event(
    event_type: str, *, now_ms: int, trade_id: str = ""
) -> EventEnvelope:
    """Build a schema-only event; this is never reported as exchange evidence."""
    return EventEnvelope(
        account_id="schema-probe",
        event_type=event_type,
        event_time_ms=now_ms,
        transaction_time_ms=now_ms,
        symbol="SCHEMA",
        position_side="LONG",
        order_id="schema",
        trade_id=trade_id,
        payload={"e": event_type, "E": now_ms, "T": now_ms, "tranId": f"schema-{now_ms}"},
    )


def _raw_user_stream_payload(event: Any) -> Mapping[str, Any]:
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    nested = payload.get("event")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _latest_user_stream_event(repository: Any, event_type: str) -> Any | None:
    for event in reversed(repository.account_events):
        raw = _raw_user_stream_payload(event)
        if str(raw.get("e") or "").upper() == event_type:
            return event
    return None


def _event_envelope(event: Any, *, expected_type: str) -> EventEnvelope:
    raw = _raw_user_stream_payload(event)
    symbol = "ACCOUNT"
    position_side = ""
    order_id = event.identity
    trade_id = ""
    if expected_type == "ACCOUNT_UPDATE":
        account = raw.get("a")
        if isinstance(account, Mapping):
            positions = account.get("P")
            if isinstance(positions, list) and positions and isinstance(positions[0], Mapping):
                symbol = str(positions[0].get("s") or symbol).upper()
                position_side = str(positions[0].get("ps") or "").upper()
    elif expected_type == "ORDER_TRADE_UPDATE":
        order = raw.get("o")
        if isinstance(order, Mapping):
            symbol = str(order.get("s") or symbol).upper()
            position_side = str(order.get("ps") or "").upper()
            order_id = str(order.get("i") or order_id)
            trade_id = str(order.get("t") or "")
            if trade_id in {"0", "-1"}:
                trade_id = ""
    return EventEnvelope(
        account_id=str(event.account_id),
        event_type=expected_type,
        event_time_ms=int(event.event_time_ms),
        transaction_time_ms=int(event.transaction_time_ms),
        symbol=symbol,
        position_side=position_side,
        order_id=order_id,
        trade_id=trade_id,
        payload=raw,
    )


def _reconnect_is_calibrated(snapshot: Any) -> bool:
    health = snapshot.stream_health
    if health.last_connected_at is None:
        return False
    if health.last_calibration_at is None:
        return False
    return health.last_calibration_at >= health.last_connected_at


def _sample_payload(snapshot: Any, *, sample_index: int, observed_seconds: float) -> dict[str, Any]:
    health = snapshot.stream_health
    return {
        "sample_index": sample_index,
        "observed_seconds": observed_seconds,
        "service_state": snapshot.status.state.value,
        "service_reason": snapshot.status.reason,
        "stream_connected": health.connected,
        "stream_fresh": snapshot.freshness.fresh,
        "freshness_reason": snapshot.freshness.reason,
        "last_connected_at": (
            None if health.last_connected_at is None else health.last_connected_at.isoformat()
        ),
        "last_event_at": None if health.last_event_at is None else health.last_event_at.isoformat(),
        "last_calibration_at": (
            None if health.last_calibration_at is None else health.last_calibration_at.isoformat()
        ),
        "reconnect_count": health.reconnect_count,
        "duplicate_count": health.duplicate_count,
        "out_of_order_count": health.out_of_order_count,
        "gap_count": health.gap_count,
        "reconciliation_consistent": snapshot.direction2_health.reconciliation_consistent,
    }


async def _observe_runtime(
    runtime: Any,
    store: RuntimeAcceptanceStore,
    *,
    observe_seconds: float,
    started: datetime,
) -> tuple[Any, int, int, int]:
    """Collect periodic soak evidence and safety-gate violations."""
    duration = max(0.0, float(observe_seconds))
    interval = 60.0 if duration >= 60.0 else max(0.25, min(5.0, duration or 0.25))
    deadline = runtime.clock.monotonic() + duration
    sample_index = 0
    reconnect_without_reconciliation = 0
    ready_while_stale = 0

    while True:
        sample_index += 1
        snapshot = runtime.snapshot()
        elapsed = max(0.0, (datetime.now(UTC) - started).total_seconds())
        store.save_evidence(
            f"SOAK-{sample_index:06d}",
            _sample_payload(snapshot, sample_index=sample_index, observed_seconds=elapsed),
        )
        is_ready = snapshot.status.state.value == "READY"
        if is_ready and snapshot.stream_health.reconnect_count > 0:
            reconnect_without_reconciliation += int(not _reconnect_is_calibrated(snapshot))
        if is_ready and not snapshot.freshness.fresh:
            ready_while_stale += 1

        remaining = deadline - runtime.clock.monotonic()
        if remaining <= 0:
            return (
                snapshot,
                reconnect_without_reconciliation,
                ready_while_stale,
                sample_index,
            )
        await asyncio.sleep(min(interval, remaining))


def _synchronize_acceptance_stream_gate(engine: RuntimeAcceptanceEngine, snapshot: Any) -> None:
    health = snapshot.stream_health
    healthy = (
        health.connected
        and snapshot.freshness.fresh
        and snapshot.direction2_health.reconciliation_consistent
        and _reconnect_is_calibrated(snapshot)
    )
    if not healthy:
        engine.stream_gate.disconnected()
        return
    engine.stream_gate.connected(at=datetime.now(UTC))
    engine.stream_gate.reconciliation_passed()


def _prepare_database_paths(database_path: Path) -> tuple[Path, Path]:
    """Normalize acceptance database paths outside the async runtime."""
    normalized = database_path.expanduser().resolve()
    normalized.parent.mkdir(parents=True, exist_ok=True)
    return normalized, normalized.with_name(normalized.stem + "-ledger.sqlite")


def _fresh_ledger_recovery(ledger_path: Path, *, account_id: str) -> tuple[Any, int]:
    """Reopen the authoritative mirror through a new SQLAlchemy engine/session."""
    recovery_engine = create_engine(
        f"sqlite:///{ledger_path.as_posix()}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    recovery_sessions = sessionmaker(
        bind=recovery_engine, autoflush=False, expire_on_commit=False
    )
    try:
        recovery_service = HedgePersistenceService(recovery_sessions)
        recovery = recovery_service.recover_account(account_id)
        unknown = count_unrecovered_unknown_orders(recovery_sessions, account_id=account_id)
        return recovery, unknown
    finally:
        recovery_engine.dispose()


async def run_live_acceptance(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    database_path: Path,
    observe_seconds: float,
    target_soak_stage: str,
) -> Any:
    """Run read-only Binance evidence. No order/mode/leverage write endpoint is invoked."""
    database_path, ledger_path = _prepare_database_paths(database_path)
    sql_engine = create_engine(
        f"sqlite:///{ledger_path.as_posix()}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    HedgeModelBase.metadata.create_all(sql_engine)
    session_factory = sessionmaker(bind=sql_engine, autoflush=False, expire_on_commit=False)
    persistence = HedgePersistenceService(session_factory)
    repository = PersistenceMirroringReadonlyRepository(persistence)
    runtime = build_binance_readonly_runtime_from_freqtrade_config(
        config=config, repository=repository
    )
    store = RuntimeAcceptanceStore(database_path)
    policy = AcceptancePolicy(
        max_clock_skew_ms=float(runtime.config.max_clock_skew_ms),
        quantity_tolerance=runtime.config.quantity_tolerance,
        financial_tolerance=runtime.config.financial_tolerance,
        stale_after=runtime.config.event_stale_after,
    )
    engine = RuntimeAcceptanceEngine(
        config=config,
        project_root=project_root,
        account_id=runtime.config.account_id,
        managed_symbols=runtime.config.managed_symbols,
        store=store,
        policy=policy,
        live_evidence=True,
    )
    started = datetime.now(UTC)
    try:
        engine.round01_baseline()
        _require_round_passed(engine)
        engine.round02_clock(await _clock_audit(runtime.client, policy))
        _require_round_passed(engine)
        await _preflight_signed_access(runtime.client, runtime.config.permission_policy)
        fill_start = int((datetime.now(UTC) - runtime.config.fill_lookback).timestamp() * 1000)
        bundle = await runtime.client.fetch_bundle(
            include_fills=True, fill_start_time_ms=fill_start
        )
        engine.round03_assets(bundle.account_snapshot, bundle.balances)
        _require_round_passed(engine)
        engine.round04_configuration(
            bundle.configuration, target_leverage=runtime.config.target_leverage
        )
        _require_round_passed(engine)
        engine.round05_identity(bundle.positions, bundle.configuration)
        _require_round_passed(engine)
        engine.round06_orders(
            bundle.open_orders,
            bundle.order_history,
            query_recovered_ids=bundle.order_history_query_recovered_ids,
            snapshot_fallback_ids=bundle.order_history_snapshot_fallback_ids,
        )
        _require_round_passed(engine)
        engine.round07_trades(bundle.fills)
        _require_round_passed(engine)
        engine.round08_income(bundle.income_events)
        _require_round_passed(engine)
        engine.round09_rest_snapshot(
            positions=bundle.positions,
            balances=bundle.balances,
            orders=bundle.open_orders,
            fills=bundle.fills,
            income=bundle.income_events,
            observed_at=bundle.collection_completed_at,
        )
        _require_round_passed(engine)

        await runtime.start()
        initial_snapshot = await runtime.wait_until_ready(
            timeout_seconds=max(30.0, min(180.0, observe_seconds + 30.0))
        )
        engine.round10_user_stream(
            evidence_source="BINANCE_USER_STREAM",
            stream_metrics={
                "connected": initial_snapshot.stream_health.connected,
                "fresh": initial_snapshot.freshness.fresh,
                "listen_key_generation": initial_snapshot.listen_key_generation,
                "reconnect_count": initial_snapshot.stream_health.reconnect_count,
                "calibrated_after_connect": _reconnect_is_calibrated(initial_snapshot),
            },
        )
        _require_round_passed(engine)

        runtime_snapshot, reconnect_violations, stale_ready_violations, sample_count = (
            await _observe_runtime(
                runtime,
                store,
                observe_seconds=observe_seconds,
                started=started,
            )
        )
        engine.session.set_metric(
            "ws_reconnect_without_reconciliation",
            max(
                engine.session.hard_metrics.ws_reconnect_without_reconciliation,
                reconnect_violations,
            ),
        )
        engine.session.set_metric(
            "new_risk_while_stale",
            max(engine.session.hard_metrics.new_risk_while_stale, stale_ready_violations),
        )

        account_event = _latest_user_stream_event(repository, "ACCOUNT_UPDATE")
        order_event = _latest_user_stream_event(repository, "ORDER_TRADE_UPDATE")
        now_ms = _epoch_ms()
        if account_event is None:
            engine.round11_account_update(
                _synthetic_schema_event("ACCOUNT_UPDATE", now_ms=now_ms),
                evidence_source="SCHEMA_PROBE_NO_LIVE_ACCOUNT_UPDATE",
            )
        else:
            engine.round11_account_update(
                _event_envelope(account_event, expected_type="ACCOUNT_UPDATE"),
                evidence_source="BINANCE_USER_STREAM",
            )
        _require_round_passed(engine)
        if order_event is None:
            engine.round12_order_trade_update(
                _synthetic_schema_event(
                    "ORDER_TRADE_UPDATE", now_ms=now_ms + 1, trade_id="schema-fill"
                ),
                evidence_source="SCHEMA_PROBE_NO_LIVE_ORDER_TRADE_UPDATE",
            )
        else:
            engine.round12_order_trade_update(
                _event_envelope(order_event, expected_type="ORDER_TRADE_UPDATE"),
                evidence_source="BINANCE_USER_STREAM",
            )
        _require_round_passed(engine)
        engine.round13_duplicates(
            _synthetic_schema_event(
                "ORDER_TRADE_UPDATE", now_ms=now_ms + 2, trade_id="schema-dedupe"
            ),
            _synthetic_schema_event("ACCOUNT_UPDATE", now_ms=now_ms + 3),
        )
        _require_round_passed(engine)
        engine.round14_out_of_order(
            _synthetic_schema_event(
                "ORDER_TRADE_UPDATE", now_ms=now_ms + 20, trade_id="newer"
            ),
            _synthetic_schema_event(
                "ORDER_TRADE_UPDATE", now_ms=now_ms + 10, trade_id="older"
            ),
        )
        _require_round_passed(engine)
        engine.round15_gap_recovery()
        _require_round_passed(engine)

        await runtime.calibrate_now(CalibrationKind.FULL)
        latest = runtime.calibration.last_bundle
        if latest is None:
            raise RuntimeError("FULL calibration produced no REST bundle")
        rest = build_fact_plane(
            account_id=runtime.config.account_id,
            observed_at=latest.collection_completed_at,
            positions=latest.positions,
            balances=latest.balances,
            orders=latest.open_orders,
            fills=latest.fills,
            income=latest.income_events,
        )
        runtime_snapshot = runtime.snapshot()
        memory = build_memory_plane(
            repository, runtime_snapshot, account_id=runtime.config.account_id
        )
        database = build_database_plane(session_factory, account_id=runtime.config.account_id)
        store.save_plane("MEMORY", memory)
        store.save_plane("DB", database)
        engine.reseed_rest_plane(rest)
        engine.round16_fast_reconciliation(memory, database)
        _require_round_passed(engine)
        engine.round17_deep_reconciliation(memory, database)
        _require_round_passed(engine)

        recovery, unknown_orders = _fresh_ledger_recovery(
            ledger_path, account_id=runtime.config.account_id
        )
        expected_positions = {
            (value.symbol, value.position_side): value.quantity
            for value in rest.positions.values()
            if value.quantity != 0
        }
        recovered_positions = {
            (value.symbol, value.position_side): Decimal(value.quantity)
            for value in recovery.positions
            if Decimal(value.quantity) != 0
        }
        engine.round18_crash_recovery(
            state_hash=rest.fingerprint(),
            external_recovery_ok=expected_positions == recovered_positions,
            external_unknown_orders=unknown_orders,
        )
        _require_round_passed(engine)
        engine.round19_fault_injection()
        _require_round_passed(engine)

        final_snapshot = runtime.snapshot()
        _synchronize_acceptance_stream_gate(engine, final_snapshot)
        observed = datetime.now(UTC) - started
        engine.session.note(
            f"Live soak samples={sample_count}; "
            f"reconnect={final_snapshot.stream_health.reconnect_count}; "
            f"duplicate={final_snapshot.stream_health.duplicate_count}; "
            f"out_of_order={final_snapshot.stream_health.out_of_order_count}; "
            f"gap={final_snapshot.stream_health.gap_count}."
        )
        if account_event is None or order_event is None:
            engine.session.note(
                "A quiet account produced no live ACCOUNT_UPDATE and/or ORDER_TRADE_UPDATE; "
                "those round schemas were verified by deterministic probes and are not claimed "
                "as live business-event evidence."
            )
        engine.round20_readiness(observed_duration=observed, target_stage=target_soak_stage)
        _require_round_passed(engine)
        return engine.session.finalize()
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.exception("Failed to stop readonly acceptance runtime cleanly")
        store.close()
        sql_engine.dispose()
