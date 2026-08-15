from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
from freqtrade.hedge.integration.paper_state import JsonPaperStateStore
from freqtrade.hedge.planning.context import MarketSnapshot
from freqtrade.hedge.simulation.exchange import BarEvent


def _config() -> dict[str, object]:
    return {
        "managed_pair": "ETH/USDT:USDT",
        "hedge": {
            "paper": {
                "initial_balance": "1000",
                "leverage": "3",
                "auto_fill": True,
                "long_signal": "1",
                "short_signal": "1",
                "tick_size": "0.01",
                "qty_step": "0.001",
                "min_qty": "0.001",
                "min_notional": "5",
                "volume_participation": "0.10",
                "bar_volume": "100",
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


def test_paper_uses_shared_matcher_and_recovers_confirmed_state(tmp_path) -> None:
    state_path = tmp_path / "paper-state.json"
    store = JsonPaperStateStore(state_path)
    first = IntegratedPaperHedgeApplication(
        config=_config(),
        account_id="paper-main",
        symbol="ETH/USDT:USDT",
        state_store=store,
    )
    now = datetime.now(UTC)
    market = _market(now, "2000")
    submitted = first.run_market_cycle(market, bar=_bar(market))
    assert submitted.executions
    assert submitted.fills == ()
    fill_market = _market(now + timedelta(minutes=1), "2000")
    result = first.run_market_cycle(fill_market, bar=_bar(fill_market))
    assert result.fills
    first_wallet = result.wallet
    first_account = first.execution.exchange.account.snapshot()
    first_active = first._active_execution_orders()
    assert first_active
    assert all(Decimal(row["fees"]) > 0 for row in first_account)
    assert state_path.exists()

    recovered = IntegratedPaperHedgeApplication(
        config=_config(),
        account_id="paper-main",
        symbol="ETH/USDT:USDT",
        state_store=store,
    )
    recovered_wallet = recovered.wallet(_market(now + timedelta(minutes=2), "2000"))
    assert recovered_wallet.long.quantity == first_wallet.long.quantity
    assert recovered_wallet.short.quantity == first_wallet.short.quantity
    assert recovered_wallet.long.average_price == first_wallet.long.average_price
    assert recovered_wallet.short.average_price == first_wallet.short.average_price
    assert recovered.execution.exchange.account.snapshot() == first_account

    recovered_active = recovered._active_execution_orders()
    assert [
        (
            item.client_order_id,
            item.lifecycle.status,
            item.lifecycle.filled_quantity,
            item.approved_quantity,
        )
        for item in recovered_active
    ] == [
        (
            item.client_order_id,
            item.lifecycle.status,
            item.lifecycle.filled_quantity,
            item.approved_quantity,
        )
        for item in first_active
    ]
    assert recovered._active_orders() == first._active_orders()
