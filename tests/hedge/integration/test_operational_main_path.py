from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.exchange.base import (
    AccountSnapshotFact,
    BalanceFact,
    ExchangeFactBatch,
    FillFact,
    PositionFact,
)
from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
from freqtrade.hedge.integration.repository import PersistenceMirroringReadonlyRepository
from freqtrade.hedge.planning.context import MarketSnapshot
from freqtrade.hedge.simulation.exchange import BarEvent

NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _paper_config() -> dict[str, object]:
    return {
        "hedge": {
            "target_leverage": "3",
            "max_gross_notional": "10000",
            "max_gross_exposure_ratio": "0.80",
            "max_margin_utilization": "0.80",
            "max_single_order_notional": "10",
            "min_liquidation_buffer_ratio": "0.05",
            "planner": {"cooldown_seconds": 0, "max_grid_layers": 2},
            "paper": {
                "initial_balance": "1000",
                "auto_fill": True,
                "long_signal": "1",
                "short_signal": "1",
            },
        }
    }


def _market(price: str, offset: int = 0) -> MarketSnapshot:
    mark = Decimal(price)
    return MarketSnapshot(
        "ETH/USDT:USDT",
        NOW + timedelta(seconds=offset),
        mark - Decimal("0.1"),
        mark + Decimal("0.1"),
        mark,
        Decimal("0.1"),
        Decimal("0.001"),
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


def test_paper_runtime_keeps_planner_state_and_uses_direction_three_risk():
    application = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="hedge-main",
        symbol="ETH/USDT:USDT",
    )
    first_market = _market("100")
    first = application.run_market_cycle(first_market, bar=_bar(first_market))
    second_market = _market("100", 1)
    second = application.run_market_cycle(second_market, bar=_bar(second_market))

    assert first.executions
    assert first.fills == ()
    assert second.fills
    assert application.long_state.sequence > 0
    assert application.short_state.sequence > 0
    assert second.wallet.long.quantity > 0
    assert second.wallet.short.quantity > 0
    assert application.risk_portfolio().account.risk_data_valid is True
    # The old merge used AllowAllRiskApproval. A small D3 single-order limit
    # must clip at least one planner request before fake submission.
    assert any(
        result.order.approved_quantity < result.order.intent.quantity
        for result in first.executions
    )


def test_paper_runtime_can_publish_positions_to_central_runtime(tmp_path):
    from freqtrade.enums.hedge import PositionMode
    from freqtrade.hedge.config import HedgeRuntimeConfig
    from freqtrade.hedge.integration.paper_state import JsonPaperStateStore
    from freqtrade.hedge.runtime import HedgeRuntime

    application = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="hedge-main",
        symbol="ETH/USDT:USDT",
        state_store=JsonPaperStateStore(tmp_path / "paper-runtime.json"),
    )
    market = _market("100")
    application.run_market_cycle(market, bar=_bar(market))
    next_market = _market("100", 1)
    application.run_market_cycle(next_market, bar=_bar(next_market))
    runtime = HedgeRuntime(
        HedgeRuntimeConfig(
            position_mode=PositionMode.HEDGE,
            enabled=True,
            managed_pair="ETH/USDT:USDT",
            exchange_adapter="binance",
            account_id="hedge-main",
            operation_mode="paper",
        )
    )
    application.publish_runtime(runtime)
    view = runtime.view()
    assert view.ready is True
    assert {item.position_side.value for item in view.positions} == {"LONG", "SHORT"}
    assert view.risk is not None
    assert view.risk.gross_total_notional > 0


class _FakeLedger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __getattr__(self, name: str):
        def call(**kwargs):
            self.calls.append((name, dict(kwargs)))
            if name == "start_reconciliation":
                return type("Run", (), {"run_id": "persistent-run"})()
            return None
        return call


class _FakeService:
    def __init__(self, *, fail: bool = False) -> None:
        self.ledger = _FakeLedger()
        self.fail = fail

    def transaction(self, operation):
        if self.fail:
            raise RuntimeError("database unavailable")
        return operation(self.ledger)


def _exchange_batch() -> ExchangeFactBatch:
    return ExchangeFactBatch(
        account_id="hedge-main",
        source="BINANCE_REST",
        observed_at=NOW,
        account_snapshot=AccountSnapshotFact(
            account_id="hedge-main",
            total_wallet_balance=Decimal("1000"),
            total_available_balance=Decimal("900"),
            total_margin_balance=Decimal("1010"),
            total_initial_margin=Decimal("100"),
            total_maintenance_margin=Decimal("10"),
            total_unrealized_pnl=Decimal("10"),
            observed_at=NOW,
            collection_started_at=NOW,
            collection_completed_at=NOW,
        ),
        balances=(
            BalanceFact(
                account_id="hedge-main",
                asset="USDT",
                wallet_balance=Decimal("1000"),
                available_balance=Decimal("900"),
                cross_wallet_balance=Decimal("1000"),
                unrealized_pnl=Decimal("10"),
                observed_at=NOW,
                source="BINANCE_REST",
            ),
        ),
        positions=(
            PositionFact(
                account_id="hedge-main",
                symbol="ETHUSDT",
                position_side="LONG",
                quantity=Decimal("1"),
                entry_price=Decimal("100"),
                mark_price=Decimal("101"),
                unrealized_pnl=Decimal("1"),
                liquidation_price=Decimal("50"),
                leverage=3,
                margin_mode="CROSSED",
                update_time_ms=1000,
                observed_at=NOW,
                source="BINANCE_REST",
            ),
        ),
        fills=(
            FillFact(
                account_id="hedge-main",
                symbol="ETHUSDT",
                position_side="LONG",
                exchange_trade_id="trade-1",
                exchange_order_id="order-1",
                side="BUY",
                quantity=Decimal("1"),
                price=Decimal("100"),
                commission=Decimal("0.04"),
                commission_asset="USDT",
                realized_pnl=Decimal("0"),
                event_time_ms=1000,
                observed_at=NOW,
                source="BINANCE_REST",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_persistence_mirror_records_fill_balance_and_account_risk():
    service = _FakeService()
    repository = PersistenceMirroringReadonlyRepository(service)
    await repository.append_exchange_fact_batch(_exchange_batch())
    names = [name for name, _ in service.ledger.calls]
    assert "append_position_snapshot" in names
    assert "apply_fill" in names
    assert "record_audit_event" in names
    assert "append_account_risk_snapshot" in names
    assert await repository.has_fill("hedge-main", "ETHUSDT", "trade-1")


@pytest.mark.asyncio
async def test_persistence_failure_does_not_advance_fast_projection():
    repository = PersistenceMirroringReadonlyRepository(_FakeService(fail=True))
    with pytest.raises(RuntimeError, match="database unavailable"):
        await repository.append_exchange_fact_batch(_exchange_batch())
    assert await repository.load_active_positions("hedge-main") == ()
