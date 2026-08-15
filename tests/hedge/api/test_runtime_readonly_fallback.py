from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.enums.hedge import PositionMode
from freqtrade.hedge.config import HedgeRuntimeConfig
from freqtrade.hedge.exchange.base import OrderFact, ReadonlyAccountView
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.risk.models import AccountRiskSnapshot
from freqtrade.hedge.runtime import HedgeRuntime
from freqtrade.rpc.api_server.hedge_runtime import HedgeExecutionRuntimeQuery

NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _runtime() -> HedgeRuntime:
    runtime = HedgeRuntime(
        HedgeRuntimeConfig(
            position_mode=PositionMode.HEDGE,
            enabled=True,
            managed_pair="ETH/USDT:USDT",
            exchange_adapter="binance",
            account_id="hedge-main",
        )
    )
    runtime.publish(
        positions=(
            PositionRecord(
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                amount="2",
                entry_price="100",
                mark_price="101",
                unrealized_pnl="2",
                leverage="3",
                collateral="67.333333",
                source="BINANCE_REST",
                account_id="hedge-main",
            ),
            PositionRecord(
                symbol="ETH/USDT:USDT",
                position_side="SHORT",
                amount="1",
                entry_price="102",
                mark_price="101",
                unrealized_pnl="1",
                leverage="3",
                collateral="33.666667",
                source="BINANCE_REST",
                account_id="hedge-main",
            ),
        ),
        risk=AccountRiskSnapshot(
            account_id="hedge-main",
            equity=Decimal("1000"),
            wallet_balance=Decimal("997"),
            available_balance=Decimal("900"),
            initial_margin=Decimal("101"),
            maintenance_margin=Decimal("10"),
            gross_long_notional=Decimal("202"),
            gross_short_notional=Decimal("101"),
            net_notional=Decimal("101"),
            liquidation_buffer_ratio=Decimal("0.30"),
        ),
        reconciliation_status="HEALTHY",
        reconciliation_at=NOW,
        stream_state="CONNECTED",
        stream_last_event_at=NOW,
        stream_reconnect_count=0,
        checks={
            "readonly_service_bound": True,
            "rest_calibrated": True,
            "user_stream_fresh": True,
            "reconciliation_converged": True,
            "risk_snapshot_valid": True,
        },
    )
    return runtime


def _view() -> ReadonlyAccountView:
    return ReadonlyAccountView(
        account_id="hedge-main",
        observed_at=NOW,
        account_snapshot=None,
        balances=(),
        positions=(),
        active_orders=(
            OrderFact(
                account_id="hedge-main",
                symbol="ETHUSDT",
                position_side="LONG",
                exchange_order_id="123",
                client_order_id="external-123",
                side="BUY",
                order_type="LIMIT",
                status="NEW",
                original_quantity=Decimal("0.5"),
                cumulative_filled_quantity=Decimal("0.1"),
                average_price=Decimal("100"),
                reduce_only=False,
                update_time_ms=1,
                observed_at=NOW,
                source="BINANCE_REST",
                raw={"price": "99"},
            ),
        ),
        configuration=None,
        revision=1,
    )


def test_readonly_mode_execution_queries_fall_back_to_exchange_facts(monkeypatch):
    query = HedgeExecutionRuntimeQuery()
    monkeypatch.setattr(query, "_paper_adapter", lambda: None)
    monkeypatch.setattr(query, "_readonly_account_view", _view)
    monkeypatch.setattr(query, "_central_runtime", _runtime)

    orders = query.orders(
        account_id="hedge-main",
        symbol="ETH/USDT:USDT",
        status="ACKNOWLEDGED",
        limit=10,
    )
    assert orders.count == 1
    assert orders.orders[0].remaining_quantity == Decimal("0.4")
    assert orders.orders[0].action == "INCREASE"

    summary = query.pair_summary(
        account_id="hedge-main",
        symbol="ETH/USDT:USDT",
    )
    assert summary.long_quantity == Decimal("2")
    assert summary.short_quantity == Decimal("1")
    assert summary.net_quantity == Decimal("1")
    assert summary.pending_entry_quantity == Decimal("0.4")
