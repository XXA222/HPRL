from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from freqtrade.enums.hedge import PositionMode, PositionSide
from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config import HedgeRuntimeConfig, normalize_hedge_config
from freqtrade.hedge.execution.integrated_fake import build_integrated_fake_runtime
from freqtrade.hedge.integration.market_data import exchange_market_rules
from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
from freqtrade.hedge.integration.paper_state import JsonPaperStateStore
from freqtrade.hedge.integration.signal_provider import FreqtradeStrategySignalProvider
from freqtrade.hedge.planning.context import MarketSnapshot
from freqtrade.hedge.simulation.exchange import BarEvent
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.risk.models import AccountRiskSnapshot
from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime
from freqtrade.persistence.hedge_bootstrap import bootstrap_hedge_schema


ROOT = Path(__file__).resolve().parents[3]


def _runtime_config(mode: str = "shadow") -> HedgeRuntimeConfig:
    return HedgeRuntimeConfig(
        position_mode=PositionMode.HEDGE,
        enabled=True,
        managed_pair="ETH/USDT:USDT",
        account_id="main",
        exchange_adapter="binance",
        operation_mode=mode,
    )


def _position(side: PositionSide, amount: str, *, source: str) -> PositionRecord:
    return PositionRecord(
        symbol="ETH/USDT:USDT",
        position_side=side,
        amount=Decimal(amount),
        entry_price=Decimal("2000"),
        mark_price=Decimal("2001"),
        exchange="binance" if source == "EXCHANGE" else "paper",
        account_id="main",
        source=source,
    )


def _risk() -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("8000"),
        initial_margin=Decimal("1000"),
        maintenance_margin=Decimal("100"),
        gross_long_notional=Decimal("2001"),
        gross_short_notional=Decimal("0"),
        net_notional=Decimal("2001"),
    )


def _exchange_checks() -> dict[str, bool]:
    return {
        "common.persistence_healthy": True,
        "exchange.readonly_service_bound": True,
        "exchange.rest_calibrated": True,
        "exchange.user_stream_fresh": True,
        "exchange.reconciliation_converged": True,
        "exchange.risk_snapshot_valid": True,
    }


def _paper_checks() -> dict[str, bool]:
    return {
        "common.persistence_healthy": True,
        "paper.market_data_fresh": True,
        "paper.funding_source_healthy": True,
        "paper.account_events_durable": True,
        "paper.simulation_engine_healthy": True,
        "paper.ledger_durable": True,
        "paper.risk_snapshot_valid": True,
    }


def test_shadow_keeps_exchange_and_paper_projections_separate() -> None:
    runtime = HedgeRuntime(_runtime_config("shadow"))
    observed = datetime.now(UTC)
    runtime.publish(
        source=HedgeProjectionSource.EXCHANGE,
        positions=(_position(PositionSide.LONG, "1", source="EXCHANGE"),),
        risk=_risk(),
        reconciliation_status="HEALTHY",
        reconciliation_at=observed,
        stream_state="CONNECTED",
        stream_last_event_at=observed,
        stream_reconnect_count=0,
        checks=_exchange_checks(),
        source_version="exchange-v1",
    )
    runtime.halt("EXCHANGE_FACTS_STALE", source=HedgeProjectionSource.EXCHANGE)
    runtime.publish(
        source=HedgeProjectionSource.PAPER,
        positions=(_position(PositionSide.SHORT, "2", source="PAPER"),),
        risk=_risk(),
        reconciliation_status="NOT_APPLICABLE",
        reconciliation_at=None,
        stream_state="NOT_APPLICABLE",
        stream_last_event_at=None,
        stream_reconnect_count=0,
        checks=_paper_checks(),
        source_version="paper-v1",
    )

    effective = runtime.view()
    exchange = runtime.view(HedgeProjectionSource.EXCHANGE)
    paper = runtime.view(HedgeProjectionSource.PAPER)

    assert effective.source is HedgeProjectionSource.EXCHANGE
    assert effective.positions[0].position_side is PositionSide.LONG
    assert effective.halted is True
    assert "EXCHANGE_FACTS_STALE" in effective.reasons
    assert exchange.sequence == 1
    assert paper.positions[0].position_side is PositionSide.SHORT
    assert paper.ready is True
    assert set(paper.available_sources) == {
        HedgeProjectionSource.EXCHANGE,
        HedgeProjectionSource.PAPER,
    }


def test_paper_projection_cannot_publish_exchange_health_names() -> None:
    runtime = HedgeRuntime(_runtime_config("paper"))
    with pytest.raises(ValueError, match="namespaced set"):
        runtime.publish(
            source=HedgeProjectionSource.PAPER,
            positions=(),
            risk=_risk(),
            reconciliation_status="NOT_APPLICABLE",
            reconciliation_at=None,
            stream_state="NOT_APPLICABLE",
            stream_last_event_at=None,
            stream_reconnect_count=0,
            checks={
                **_paper_checks(),
                "exchange.user_stream_fresh": True,
            },
        )


def test_disabled_hedge_bootstrap_is_schema_noop() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE trades (id INTEGER PRIMARY KEY, pair VARCHAR(64))"))
        connection.execute(text("CREATE TABLE orders (id INTEGER PRIMARY KEY, ft_pair VARCHAR(64))"))
    before = {
        table: tuple(column["name"] for column in inspect(engine).get_columns(table))
        for table in inspect(engine).get_table_names()
    }

    report = bootstrap_hedge_schema(engine, {"hedge_mode_enabled": False})

    after = {
        table: tuple(column["name"] for column in inspect(engine).get_columns(table))
        for table in inspect(engine).get_table_names()
    }
    assert report.plan.enabled is False
    assert report.migration.applied == ()
    assert before == after
    assert not any(table.startswith("hedge_") for table in after)


def test_native_migration_hook_has_no_h3_calls_or_imports() -> None:
    tree = ast.parse((ROOT / "freqtrade/persistence/migrations.py").read_text(encoding="utf-8"))
    forbidden = {
        "run_hedge_migrations",
        "prepare_hedge_migration_backup",
        "LedgerRecoveryCoordinator",
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    assert names.isdisjoint(forbidden)


def test_unknown_hedge_configuration_key_fails_closed() -> None:
    config: dict[str, object] = {
        "hedge_mode_enabled": True,
        "position_mode": "hedge",
        "managed_pair": "ETH/USDT:USDT",
        "trading_mode": "futures",
        "margin_mode": "cross",
        "dry_run": True,
        "exchange": {"name": "binance"},
        "hedge": {
            "operation_mode": "paper",
            "read_only": True,
            "live_trading_enabled": False,
            "paper": {"unknown_typo": True},
        },
    }
    with pytest.raises(OperationalException, match="unknown"):
        normalize_hedge_config(config)


def test_strategy_adapter_uses_latest_analyzed_row_without_order_submission() -> None:
    now = datetime.now(UTC)
    frame = pd.DataFrame(
        {
            "date": [now - timedelta(minutes=5), now],
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 12],
            "hedge_long_score": [0.1, 0.8],
            "hedge_short_score": [0.9, 0.2],
            "hedge_target_net": [0, 1.5],
            "hedge_model_version": ["old", "model-7"],
        }
    )
    provider = MagicMock()
    provider.get_analyzed_dataframe.return_value = (frame, now)
    strategy = SimpleNamespace(version=lambda: "fallback")

    signal = FreqtradeStrategySignalProvider(provider, strategy).signals(
        "ETH/USDT:USDT", "5m"
    )

    assert signal.long_score == Decimal("0.8")
    assert signal.short_score == Decimal("0.2")
    assert signal.target_net == Decimal("1.5")
    assert signal.model_version == "model-7"
    provider.get_analyzed_dataframe.assert_called_once_with("ETH/USDT:USDT", "5m")


def test_market_rules_prefer_exchange_metadata_over_config() -> None:
    exchange = SimpleNamespace(
        markets={
            "ETH/USDT:USDT": {
                "id": "ETHUSDT",
                "precision": {"price": 2, "amount": 3},
                "limits": {"amount": {"min": 0.005}, "cost": {"min": 10}},
                "last": 2000,
            }
        },
        price_get_one_pip=lambda pair, price: 0.01,
    )
    rules = exchange_market_rules(
        exchange=exchange,
        pair="ETH/USDT:USDT",
        fallback={"tick_size": "9", "qty_step": "9", "min_qty": "9", "min_notional": "9"},
    )
    assert rules.source == "EXCHANGE_MARKETS"
    assert rules.tick_size == Decimal("0.01")
    assert rules.qty_step == Decimal("0.001")
    assert rules.min_qty == Decimal("0.005")
    assert rules.min_notional == Decimal("10")


def test_strict_execution_composition_rejects_silent_test_defaults() -> None:
    with pytest.raises(ValueError, match="explicit dependencies"):
        build_integrated_fake_runtime(strict_dependencies=True)


def _paper_config() -> dict[str, object]:
    return {
        "managed_pair": "ETH/USDT:USDT",
        "hedge": {
            "paper": {
                "initial_balance": "1000",
                "leverage": "3",
                "auto_fill": True,
                "long_signal": "1",
                "short_signal": "0",
                "fill_model": "conservative",
                "tick_size": "0.01",
                "qty_step": "0.001",
                "min_qty": "0.001",
                "min_notional": "5",
                "bar_volume": "100",
                "volume_participation": "0.10",
                "max_fill_ratio_per_order": "0.25",
                "max_fills_per_bar": 1,
            },
            "planner": {"max_grid_layers": 2},
        },
    }


def _market(timestamp: datetime, mark: str) -> MarketSnapshot:
    value = Decimal(mark)
    return MarketSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=timestamp,
        bid=value - Decimal("1"),
        ask=value + Decimal("1"),
        mark=value,
        tick_size=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
    )


def _bar(market: MarketSnapshot) -> BarEvent:
    return BarEvent(
        timestamp=market.timestamp,
        symbol=market.symbol,
        open=market.mark,
        high=market.ask,
        low=market.bid,
        close=market.mark,
        volume=Decimal("100"),
    )


def test_partial_active_order_is_checkpointed_and_restored_without_resubmit(tmp_path) -> None:
    state = JsonPaperStateStore(tmp_path / "paper-state.json")
    now = datetime.now(UTC)
    first = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="main",
        symbol="ETH/USDT:USDT",
        state_store=state,
    )
    market = _market(now, "2000")
    submitted = first.run_market_cycle(market, bar=_bar(market))
    assert submitted.executions
    assert submitted.fills == ()
    fill_market = _market(now + timedelta(minutes=1), "2000")
    cycle = first.run_market_cycle(fill_market, bar=_bar(fill_market))
    active_before = first._active_execution_orders()
    assert cycle.fills
    assert active_before
    assert any(item.lifecycle.filled_quantity > 0 for item in active_before)
    assert any(
        item.lifecycle.filled_quantity < item.approved_quantity for item in active_before
    )

    recovered = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="main",
        symbol="ETH/USDT:USDT",
        state_store=state,
    )
    active_after = recovered._active_execution_orders()

    assert {item.client_order_id for item in active_after} == {
        item.client_order_id for item in active_before
    }
    assert {
        (item.client_order_id, item.lifecycle.filled_quantity)
        for item in active_after
    } == {
        (item.client_order_id, item.lifecycle.filled_quantity)
        for item in active_before
    }


def test_sql_paper_checkpoint_and_action_group_are_durable(tmp_path) -> None:
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.contracts.types import PositionSide as ContractPositionSide
    from freqtrade.hedge.execution.action_group_store import (
        ActionGroupMember,
        ActionGroupMemberState,
        ActionGroupRecord,
    )
    from freqtrade.persistence.hedge_execution_adapters import SqlActionGroupRepository
    from freqtrade.hedge.integration.paper_state import SqlPaperStateStore
    from freqtrade.persistence.hedge_models import (
        ActionGroupRow,
        EventOutbox,
        PaperRuntimeCheckpointRow,
        StrategySideState,
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'stage-a.sqlite'}")
    for model in (
        PaperRuntimeCheckpointRow,
        ActionGroupRow,
        EventOutbox,
        StrategySideState,
    ):
        model.__table__.create(engine, checkfirst=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    state = SqlPaperStateStore(
        factory,
        exchange="binance",
        account_id="main",
        symbol="ETH/USDT:USDT",
        source="PAPER",
    )
    state.save(
        {
            "account_id": "main",
            "symbol": "ETH/USDT:USDT",
            "active_orders": [],
        }
    )
    loaded = state.load()
    assert loaded is not None
    assert loaded["schema_version"] == 2

    from uuid import uuid4

    action_group_id = uuid4()
    group = ActionGroupRecord(
        action_group_id=action_group_id,
        action_type="CLOSE_BOTH",
        account_id="main",
        symbol="ETH/USDT:USDT",
        members=(
            ActionGroupMember(ContractPositionSide.LONG),
            ActionGroupMember(ContractPositionSide.SHORT),
        ),
    )
    groups = SqlActionGroupRepository(factory)
    groups.put(group)
    updated = groups.update_member(
        action_group_id,
        ActionGroupMember(
            ContractPositionSide.LONG,
            state=ActionGroupMemberState.SUBMITTED,
            client_order_id="client-long",
        ),
    )

    restarted = SqlActionGroupRepository(factory)
    restored = restarted.get(action_group_id)
    assert restored == updated
    assert restored is not None
    assert restored.member(ContractPositionSide.LONG).client_order_id == "client-long"


def test_sql_execution_ledger_commits_intent_fill_audit_and_outbox(tmp_path) -> None:
    from uuid import uuid4

    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.contracts.events import FillEvent as ContractFillEvent
    from freqtrade.hedge.contracts.events import OutboxEvent
    from freqtrade.hedge.contracts.types import (
        IntentAction as ContractIntentAction,
        OrderSide as ContractOrderSide,
        PositionKey,
        PositionSide as ContractPositionSide,
    )
    from freqtrade.hedge.execution.service import (
        ExecutionOrder,
        IntentAction,
        OrderIntent,
        PositionSide as ExecutionPositionSide,
    )
    from freqtrade.persistence.hedge_execution_adapters import SqlExecutionLedger
    from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
    from freqtrade.persistence.hedge_models import (
        AuditEvent,
        EventOutbox,
        ExecutionOrderStateRow,
        FillEvent,
        OrderIntent as OrderIntentRow,
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'execution-ledger.sqlite'}")
    for model in (OrderIntentRow, ExecutionOrderStateRow, FillEvent, EventOutbox, AuditEvent):
        model.__table__.create(engine, checkfirst=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    ledger = SqlExecutionLedger(factory)
    now = datetime.now(UTC)
    intent = OrderIntent(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=ExecutionPositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.1"),
        idempotency_key="idem-ledger-1",
        limit_price=Decimal("2000"),
        metadata={"exchange": "binance"},
    )
    lifecycle = OrderLifecycle(
        status=OrderState.PARTIAL,
        filled_quantity=Decimal("0.05"),
        average_price=Decimal("2000"),
        exchange_order_id="paper-order-1",
        version=2,
        updated_at=now,
    )
    order = ExecutionOrder(
        intent=intent,
        client_order_id="client-ledger-1",
        approved_quantity=Decimal("0.1"),
        lifecycle=lifecycle,
        created_at=now,
    )
    fill = ContractFillEvent(
        position_key=PositionKey(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=ContractPositionSide.LONG,
        ),
        trade_id="trade-ledger-1",
        client_order_id=order.client_order_id,
        action=ContractIntentAction.OPEN,
        order_side=ContractOrderSide.BUY,
        quantity=Decimal("0.05"),
        price=Decimal("2000"),
        fee=Decimal("0.1"),
        exchange_time=now,
        observed_time=now,
    )
    outbox = OutboxEvent(
        event_id=uuid4(),
        event_type="FILL_RECORDED",
        payload={"client_order_id": order.client_order_id},
        correlation_id=str(intent.intent_id),
        occurred_at=now,
    )

    ledger.record(
        order=order,
        event_type="ORDER_EVENT_APPLIED",
        fill=fill,
        outbox=outbox,
        payload={"client_order_id": order.client_order_id},
    )
    ledger.mark_published(str(outbox.event_id), published_at=now)

    with factory() as session:
        assert session.scalar(select(OrderIntentRow.intent_id)) == str(intent.intent_id)
        assert session.scalar(select(FillEvent.exchange_trade_id)) == "trade-ledger-1"
        assert session.scalar(select(AuditEvent.event_type)) == "ORDER_EVENT_APPLIED"
        persisted_outbox = session.scalar(select(EventOutbox))
        assert persisted_outbox is not None
        assert persisted_outbox.status == "PUBLISHED"
