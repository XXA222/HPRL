from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from freqtrade.hedge.acceptance.baseline import audit_environment
from freqtrade.hedge.acceptance.clock import ClockSample, evaluate_clock
from freqtrade.hedge.acceptance.events import EventSequenceTracker, ExactlyOnceEffectJournal
from freqtrade.hedge.acceptance.facts import build_fact_plane, fill_values, income_values, order_values
from freqtrade.hedge.acceptance.faults import FaultClass, classify_http_fault
from freqtrade.hedge.acceptance.identity import build_leg_identities, identity_mismatch_count
from freqtrade.hedge.acceptance.models import (
    AcceptancePolicy,
    HardMetrics,
    ReconciliationDepth,
    RuntimeSnapshotSet,
    RuntimeStage,
)
from freqtrade.hedge.acceptance.persistence import RuntimeAcceptanceStore
from freqtrade.hedge.acceptance.readiness import evaluate_readiness, required_soak_for_stage
from freqtrade.hedge.acceptance.reconciliation import reconcile_planes, reconciliation_issue_metrics
from freqtrade.hedge.acceptance.scenario import (
    _balances,
    _configuration,
    _event,
    _fills,
    _income,
    _now,
    _orders,
    _positions,
    _snapshot,
    run_deterministic_acceptance,
)
from freqtrade.hedge.acceptance.session import (
    RuntimeAcceptanceRoundFailure,
    RuntimeAcceptanceSession,
)
from freqtrade.hedge.acceptance.stream import StreamRecoveryGate




def _safe_config() -> dict[str, object]:
    return {
        "exchange": {"name": "binance", "key": "x", "secret": "y", "pair_whitelist": ["BTCUSDT"]},
        "hedge_mode_enabled": True,
        "db_url": "sqlite:///runtime.db",
        "hedge": {
            "read_only": True,
            "live_trading_enabled": False,
            "operation_mode": "readonly",
            "managed_symbols": ["BTCUSDT"],
        },
    }


def test_round01_environment_baseline_is_fail_closed(tmp_path: Path) -> None:
    baseline = audit_environment(_safe_config(), project_root=tmp_path)
    assert baseline.safe_for_runtime_acceptance
    unsafe = _safe_config()
    unsafe["hedge"] = dict(unsafe["hedge"], live_trading_enabled=True)
    assert not audit_environment(unsafe, project_root=tmp_path).safe_for_runtime_acceptance


def test_live_round_failure_surfaces_original_round_evidence() -> None:
    session = RuntimeAcceptanceSession(live_evidence=True)
    session.record(
        "ACCEPT-01",
        passed=False,
        checks=("BASELINE",),
        metrics={"reason": "example"},
        detail="original failure detail",
    )

    with pytest.raises(RuntimeAcceptanceRoundFailure) as captured:
        session.require_last_passed()

    assert "ACCEPT-01 FAIL" in str(captured.value)
    assert captured.value.evidence.round_id == "ACCEPT-01"
    assert captured.value.evidence.metrics == {"reason": "example"}
    assert "original failure detail" in str(captured.value)


def test_round02_clock_midpoint_detects_skew_and_rtt() -> None:
    good = evaluate_clock((ClockSample(1000, 1005, 1010),), max_abs_skew_ms=10, max_rtt_ms=20)
    bad = evaluate_clock((ClockSample(1000, 2000, 1010),), max_abs_skew_ms=10, max_rtt_ms=20)
    assert good.synchronized
    assert not bad.synchronized


def test_round03_account_assets_are_unique() -> None:
    now = _now()
    balances = _balances(now)
    assert len({item.asset for item in balances}) == len(balances)
    assert _snapshot(now).total_initial_margin >= 0


def test_round04_hedge_cross_leverage_configuration() -> None:
    configuration = _configuration(_now())
    assert configuration.hedge_mode
    assert configuration.active_margin_modes == ("cross",)
    assert set(configuration.leverage_by_symbol_side) == {"BTCUSDT:LONG", "BTCUSDT:SHORT"}


def test_round05_zero_leg_still_has_long_short_identity() -> None:
    now = _now()
    identities = build_leg_identities(
        account_id="acct",
        managed_symbols=("BTCUSDT",),
        positions=(_positions(now)[0],),
        configuration=_configuration(now),
    )
    assert len(identities) == 2
    assert identity_mismatch_count(identities, managed_symbols=("BTCUSDT",), account_id="acct") == 0
    short = next(item for item in identities if item.position_side == "SHORT")
    assert short.quantity == 0
    assert not short.present_in_rest


def test_round06_open_order_must_be_in_history() -> None:
    now = _now()
    open_map = order_values(_orders(now))
    history_map = order_values(_orders(now))
    assert set(open_map).issubset(history_map)


def test_round07_trade_id_deduplicates_exact_copy_and_rejects_conflict() -> None:
    now = _now()
    fill = _fills(now)[0]
    assert len(fill_values((fill, fill))) == 1
    conflict = replace(fill, quantity=Decimal("0.02"))
    with pytest.raises(ValueError, match="conflicting duplicate fill"):
        fill_values((fill, conflict))


def test_round08_income_deduplicates_funding_identity() -> None:
    income = _income()[0]
    assert len(income_values("acct", (income, income))) == 1
    conflict = dict(income, income="0.99")
    with pytest.raises(ValueError, match="conflicting duplicate income"):
        income_values("acct", (income, conflict))


def test_round09_rest_snapshot_fingerprint_is_order_independent() -> None:
    now = _now()
    kwargs = {
        "account_id": "acct",
        "observed_at": now,
        "positions": _positions(now),
        "balances": _balances(now),
        "orders": _orders(now),
        "fills": _fills(now),
        "income": _income(),
    }
    left = build_fact_plane(**kwargs)
    right = build_fact_plane(**dict(kwargs, positions=tuple(reversed(kwargs["positions"]))))
    assert left.fingerprint() == right.fingerprint()


def test_round10_reconnect_requires_reconciliation_before_new_risk() -> None:
    gate = StreamRecoveryGate(stale_after=timedelta(minutes=1))
    gate.connected()
    assert not gate.assess().new_risk_enabled
    gate.reconciliation_passed()
    assert gate.assess().new_risk_enabled


def test_final14_quiet_stream_can_disable_business_event_staleness() -> None:
    connected_at = _now()
    gate = StreamRecoveryGate(stale_after=None)
    gate.connected(at=connected_at)
    gate.reconciliation_passed()

    quiet = gate.assess(now=connected_at + timedelta(hours=72))
    assert quiet.stage is RuntimeStage.READY
    assert quiet.reason == "READY"
    assert quiet.new_risk_enabled

    gate.disconnected()
    disconnected = gate.assess(now=connected_at + timedelta(hours=72))
    assert disconnected.stage is RuntimeStage.STREAM_STALE
    assert disconnected.reason == "DISCONNECTED"
    assert not disconnected.new_risk_enabled


def test_final14_explicit_business_event_staleness_remains_fail_closed() -> None:
    connected_at = _now()
    gate = StreamRecoveryGate(stale_after=timedelta(seconds=10))
    gate.connected(at=connected_at)
    gate.reconciliation_passed()

    stale = gate.assess(now=connected_at + timedelta(seconds=11))
    assert stale.stage is RuntimeStage.STREAM_STALE
    assert stale.reason == "USER_STREAM_STALE"
    assert not stale.new_risk_enabled


def test_final14_acceptance_policy_allows_runtime_quiet_stream_semantics() -> None:
    policy = AcceptancePolicy(stale_after=None)
    assert policy.stale_after is None


def test_round11_account_update_sequence_is_committed_once() -> None:
    tracker = EventSequenceTracker()
    event = _event("ACCOUNT_UPDATE", transaction_ms=100, event_ms=100)
    assert tracker.inspect(event).apply
    tracker.commit(event)
    assert tracker.inspect(event).duplicate


def test_round12_order_trade_update_requires_stable_identity() -> None:
    event = _event("ORDER_TRADE_UPDATE", transaction_ms=100, event_ms=100, trade_id="1")
    assert event.symbol == "BTCUSDT"
    assert event.position_side == "LONG"
    assert event.trade_id == "1"
    assert event.fingerprint == event.fingerprint


def test_round13_sql_exactly_once_fill_and_funding_effects(tmp_path: Path) -> None:
    store = RuntimeAcceptanceStore(tmp_path / "effects.sqlite")
    try:
        journal = ExactlyOnceEffectJournal(store)
        fill = _event("ORDER_TRADE_UPDATE", transaction_ms=100, event_ms=100, trade_id="1")
        funding = _event("ACCOUNT_UPDATE", transaction_ms=101, event_ms=101)
        assert journal.apply_fill(fill)
        assert not journal.apply_fill(fill)
        assert journal.apply_funding(funding)
        assert not journal.apply_funding(funding)
        assert store.effect_count("FILL") == 1
        assert store.effect_count("FUNDING") == 1
    finally:
        store.close()


def test_round14_out_of_order_event_is_fail_closed() -> None:
    tracker = EventSequenceTracker()
    newer = _event("ORDER_TRADE_UPDATE", transaction_ms=200, event_ms=200, trade_id="1")
    older = _event("ORDER_TRADE_UPDATE", transaction_ms=100, event_ms=100, trade_id="2")
    tracker.commit(newer)
    decision = tracker.inspect(older)
    assert decision.out_of_order
    assert not decision.apply


def test_round15_disconnect_gap_requires_rest_reconciliation() -> None:
    gate = StreamRecoveryGate(stale_after=timedelta(minutes=1))
    gate.connected()
    gate.reconciliation_passed()
    gate.disconnected()
    assert not gate.assess().new_risk_enabled
    gate.connected()
    assert not gate.assess().new_risk_enabled
    gate.reconciliation_passed()
    assert gate.assess().new_risk_enabled


def test_round16_fast_reconciliation_detects_position_drift() -> None:
    now = _now()
    rest = build_fact_plane(
        account_id="acct",
        observed_at=now,
        positions=_positions(now),
        balances=_balances(now),
        orders=_orders(now),
    )
    long_key = "acct:BTC/USDT:USDT:LONG"
    memory = replace(
        rest,
        positions={
            **rest.positions,
            long_key: replace(rest.positions[long_key], quantity=Decimal("0.02")),
        },
    )
    outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=memory, database=rest),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal("0.00000001"),
        financial_tolerance=Decimal("0.00000001"),
        wallet_drift_tolerance=Decimal("0.000001"),
    )
    assert not outcome.passed
    assert any(item.entity_type == "POSITION" for item in outcome.unexplained)

    # Wallet tolerance is deliberately independent from price/PnL tolerance.
    balance_key = "acct:USDT"
    within_wallet_tolerance = replace(
        rest,
        balances={
            **rest.balances,
            balance_key: replace(
                rest.balances[balance_key],
                wallet_balance=rest.balances[balance_key].wallet_balance
                + Decimal("0.0000005"),
            ),
        },
    )
    tolerated = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=within_wallet_tolerance, database=rest),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
        wallet_drift_tolerance=Decimal("0.000001"),
    )
    assert tolerated.passed

    order_key = next(iter(rest.active_orders))
    order_drift = replace(
        rest,
        active_orders={
            **rest.active_orders,
            order_key: replace(
                rest.active_orders[order_key], cumulative_filled_quantity=Decimal("0.001")
            ),
        },
    )
    order_outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=order_drift, database=rest),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    assert any(
        item.entity_type == "ACTIVE_ORDER"
        and item.reason == "CUMULATIVE_FILLED_QUANTITY_MEMORY"
        for item in order_outcome.unexplained
    )


def test_round16_order_status_aliases_match_persistence_lifecycle() -> None:
    now = _now()
    base = build_fact_plane(
        account_id="acct",
        observed_at=now,
        orders=_orders(now),
    )
    prototype = next(iter(base.active_orders.values()))
    aliases = (
        ("NEW", "ACKNOWLEDGED"),
        ("SUBMITTED", "ACKNOWLEDGED"),
        ("PARTIALLY_FILLED", "PARTIAL"),
        ("CANCELLED", "CANCELED"),
        ("FAILED", "UNKNOWN"),
    )
    rest_orders = {}
    database_orders = {}
    for index in range(16):
        exchange_status, lifecycle_status = aliases[index % len(aliases)]
        rest_order = replace(
            prototype,
            exchange_order_id=str(1000 + index),
            client_order_id=f"alias-{index}",
            status=exchange_status,
        )
        database_order = replace(rest_order, status=lifecycle_status)
        rest_orders[rest_order.key] = rest_order
        database_orders[database_order.key] = database_order
    rest = replace(base, active_orders=rest_orders)
    database = replace(base, active_orders=database_orders)
    outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=rest, database=database),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    assert outcome.passed
    assert outcome.unexplained == ()


def test_round16_unknown_order_status_remains_fail_closed() -> None:
    now = _now()
    rest = build_fact_plane(
        account_id="acct",
        observed_at=now,
        orders=_orders(now),
    )
    key = next(iter(rest.active_orders))
    unknown = replace(
        rest,
        active_orders={
            **rest.active_orders,
            key: replace(rest.active_orders[key], status="FUTURE_BINANCE_STATUS"),
        },
    )
    outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=unknown, memory=unknown, database=unknown),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    assert not outcome.passed
    assert {item.reason for item in outcome.unexplained} == {"STATUS_MEMORY", "STATUS_DB"}


def test_reconciliation_issue_metrics_expose_reason_and_entity() -> None:
    now = _now()
    rest = build_fact_plane(
        account_id="acct",
        observed_at=now,
        orders=_orders(now),
    )
    key = next(iter(rest.active_orders))
    database = replace(
        rest,
        active_orders={
            **rest.active_orders,
            key: replace(rest.active_orders[key], status="FILLED"),
        },
    )
    outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=rest, database=database),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    metrics = reconciliation_issue_metrics(outcome)
    assert metrics["issues"] == 1
    assert metrics["reason_counts"] == {"STATUS_DB": 1}
    assert metrics["issue_samples"][0]["entity_type"] == "ACTIVE_ORDER"
    assert metrics["issue_samples"][0]["entity_key"] == key
    assert metrics["issue_samples"][0]["expected"] == {
        "raw": "NEW",
        "canonical": "ACKNOWLEDGED",
    }
    assert metrics["issue_samples"][0]["observed"] == {
        "raw": "FILLED",
        "canonical": "FILLED",
    }


def test_round16_exchange_and_canonical_symbols_share_one_identity() -> None:
    now = _now()
    prototype = _orders(now)[0]
    live_orders = tuple(
        replace(
            prototype,
            exchange_order_id=str(8389766220000000000 + index),
            client_order_id=f"live-{index}",
            position_side="LONG" if index % 2 == 0 else "SHORT",
            status="NEW",
        )
        for index in range(6)
    )
    rest = build_fact_plane(
        account_id="acct",
        observed_at=now,
        positions=_positions(now),
        balances=_balances(now),
        orders=live_orders,
    )
    database = replace(
        rest,
        positions={
            key: replace(value, symbol="BTC/USDT:USDT")
            for key, value in rest.positions.items()
        },
        active_orders={
            key: replace(value, symbol="BTC/USDT:USDT", status="ACKNOWLEDGED")
            for key, value in rest.active_orders.items()
        },
    )
    database = replace(
        database,
        positions={value.key: value for value in database.positions.values()},
        active_orders={value.key: value for value in database.active_orders.values()},
    )
    outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=rest, database=database),
        depth=ReconciliationDepth.FAST,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    assert outcome.passed
    assert outcome.unexplained == ()
    assert len(rest.positions) == 2
    assert len(rest.active_orders) == 6
    assert all(":BTC/USDT:USDT:" in key for key in rest.positions)
    assert all(":BTC/USDT:USDT:" in key for key in rest.active_orders)


def test_round09_fact_plane_fingerprint_is_symbol_spelling_invariant() -> None:
    now = _now()
    venue = build_fact_plane(
        account_id="acct",
        observed_at=now,
        positions=_positions(now),
        orders=_orders(now),
        fills=_fills(now),
    )
    canonical = build_fact_plane(
        account_id="acct",
        observed_at=now,
        positions=tuple(replace(item, symbol="BTC/USDT:USDT") for item in _positions(now)),
        orders=tuple(replace(item, symbol="BTC/USDT:USDT") for item in _orders(now)),
        fills=tuple(replace(item, symbol="BTC/USDT:USDT") for item in _fills(now)),
    )
    assert venue == canonical
    assert venue.fingerprint() == canonical.fingerprint()


def test_round17_fill_symbol_identity_is_canonical() -> None:
    now = _now()
    plane = build_fact_plane(account_id="acct", observed_at=now, fills=_fills(now))
    assert plane.fills
    assert all(":BTC/USDT:USDT:" in key for key in plane.fills)
    assert all(value.symbol == "BTC/USDT:USDT" for value in plane.fills.values())


def test_round17_deep_reconciliation_includes_fill_and_income_keysets() -> None:
    now = _now()
    rest = build_fact_plane(
        account_id="acct", observed_at=now, fills=_fills(now), income=_income()
    )
    empty = build_fact_plane(account_id="acct", observed_at=now)
    outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=empty, database=rest),
        depth=ReconciliationDepth.DEEP,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    assert {item.entity_type for item in outcome.unexplained} == {"FILL", "INCOME"}

    fill_key = next(iter(rest.fills))
    income_key = next(iter(rest.income))
    value_drift = replace(
        rest,
        fills={
            **rest.fills,
            fill_key: replace(rest.fills[fill_key], commission=Decimal("9.9")),
        },
        income={
            **rest.income,
            income_key: replace(rest.income[income_key], amount=Decimal("9.9")),
        },
    )
    value_outcome = reconcile_planes(
        RuntimeSnapshotSet(rest=rest, memory=value_drift, database=rest),
        depth=ReconciliationDepth.DEEP,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )
    reasons = {item.reason for item in value_outcome.unexplained}
    assert "COMMISSION_MEMORY" in reasons
    assert "AMOUNT_MEMORY" in reasons


def test_round17_deep_reconciliation_allows_local_append_only_history_superset() -> None:
    now = _now()
    historical = build_fact_plane(
        account_id="acct", observed_at=now, fills=_fills(now), income=_income()
    )
    current_rest_window = build_fact_plane(account_id="acct", observed_at=now)

    outcome = reconcile_planes(
        RuntimeSnapshotSet(
            rest=current_rest_window,
            memory=historical,
            database=historical,
        ),
        depth=ReconciliationDepth.DEEP,
        quantity_tolerance=Decimal(0),
        financial_tolerance=Decimal(0),
    )

    assert outcome.passed
    assert outcome.unexplained == ()


def test_round18_sqlite_checkpoint_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "recovery.sqlite"
    store = RuntimeAcceptanceStore(path)
    store.save_checkpoint("crash", "abc", {"x": 1})
    store.close()
    reopened = RuntimeAcceptanceStore(path)
    try:
        assert reopened.checkpoint_hash("crash") == "abc"
        assert reopened.unknown_order_count() == 0
    finally:
        reopened.close()


def test_round19_network_429_5xx_are_retryable_but_new_risk_locked() -> None:
    decisions = (
        classify_http_fault(None, network_error=True),
        classify_http_fault(429),
        classify_http_fault(503),
    )
    assert [item.fault_class for item in decisions] == [
        FaultClass.NETWORK,
        FaultClass.RATE_LIMIT,
        FaultClass.SERVER,
    ]
    assert all(
        item.retryable and item.requires_reconciliation and not item.new_risk_allowed
        for item in decisions
    )


def test_round20_smoke_stage_is_distinct_from_production_soak() -> None:
    assert required_soak_for_stage("smoke") == timedelta(seconds=60)
    assert required_soak_for_stage("1h") == timedelta(hours=1)

    gate = StreamRecoveryGate(stale_after=timedelta(days=10))
    gate.connected(at=datetime.now(UTC))
    gate.reconciliation_passed()
    too_short = evaluate_readiness(
        hard_metrics=HardMetrics(),
        stream_state=gate.assess(),
        observed_duration=timedelta(seconds=59),
        target_stage="smoke",
    )
    smoke_ready = evaluate_readiness(
        hard_metrics=HardMetrics(),
        stream_state=gate.assess(),
        observed_duration=timedelta(seconds=60),
        target_stage="smoke",
    )
    assert not too_short.ready
    assert smoke_ready.ready


def test_round20_production_readiness_requires_real_soak_duration() -> None:
    gate = StreamRecoveryGate(stale_after=timedelta(days=10))
    gate.connected(at=datetime.now(UTC))
    gate.reconciliation_passed()
    not_ready = evaluate_readiness(
        hard_metrics=HardMetrics(),
        stream_state=gate.assess(),
        observed_duration=timedelta(minutes=59),
        target_stage="1h",
    )
    ready = evaluate_readiness(
        hard_metrics=HardMetrics(),
        stream_state=gate.assess(),
        observed_duration=timedelta(hours=1),
        target_stage="1h",
    )
    assert not not_ready.ready
    assert ready.ready


def test_complete_deterministic_20_round_acceptance(tmp_path: Path) -> None:
    report = run_deterministic_acceptance(project_root=tmp_path, output_db=tmp_path / "full.sqlite")
    assert report.passed
    assert len(report.rounds) == 20
    assert report.hard_metrics.passed
    assert not report.live_evidence


def test_acceptance_policy_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        AcceptancePolicy(quantity_tolerance=Decimal(-1))
