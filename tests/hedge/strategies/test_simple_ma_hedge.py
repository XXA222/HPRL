from datetime import UTC, datetime, timedelta
from decimal import Decimal

from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.simulation.exchange import BarEvent, SignalEvent
from freqtrade.hedge.strategies.simple_ma_hedge import (
    SimpleDualLegMaConfig,
    SimpleDualLegMaHedgeStrategy,
)
from freqtrade.optimize.hedge_backtesting import HedgeBacktesting

START = datetime(2026, 1, 1, tzinfo=UTC)
SYMBOL = "ETH/USDT:USDT"


def _bars() -> list[BarEvent]:
    prices = [Decimal("100") + Decimal(index % 12 - 6) for index in range(72)]
    result: list[BarEvent] = []
    for index, close in enumerate(prices):
        open_price = prices[index - 1] if index else close
        result.append(
            BarEvent(
                START + timedelta(minutes=5 * index),
                SYMBOL,
                open_price,
                max(open_price, close) + Decimal("2"),
                min(open_price, close) - Decimal("2"),
                close,
                Decimal("1000"),
            )
        )
    return result


def test_signals_are_bounded_and_directional() -> None:
    strategy = SimpleDualLegMaHedgeStrategy(
        SimpleDualLegMaConfig(fast_window=2, slow_window=4, sensitivity=Decimal("10"))
    )
    long_signal, short_signal = strategy.signal(
        [Decimal("100"), Decimal("100"), Decimal("90"), Decimal("90")]
    )
    assert Decimal("0") <= long_signal <= Decimal("1")
    assert Decimal("0") <= short_signal <= Decimal("1")
    assert long_signal > short_signal


def test_events_emit_signal_before_bar_without_using_current_close() -> None:
    strategy = SimpleDualLegMaHedgeStrategy(
        SimpleDualLegMaConfig(fast_window=2, slow_window=4)
    )
    bars = _bars()[:5]
    events = list(strategy.events(bars))
    assert isinstance(events[0], SignalEvent)
    assert events[0].long_signal == Decimal("0.20")
    assert events[0].timestamp == bars[0].timestamp
    assert events[1] is bars[0]


def test_simple_strategy_runs_dual_leg_backtest() -> None:
    config = PlannerConfig(
        core_wallet_exposure_long=Decimal("0.12"),
        core_wallet_exposure_short=Decimal("0.12"),
        tactical_wallet_exposure_long=Decimal("0.10"),
        tactical_wallet_exposure_short=Decimal("0.10"),
        max_wallet_exposure_long=Decimal("0.28"),
        max_wallet_exposure_short=Decimal("0.28"),
        max_gross_wallet_exposure=Decimal("0.50"),
        initial_entry_fraction=Decimal("0.60"),
        max_grid_layers=3,
        cooldown_seconds=0,
        trailing_rebound=Decimal("0"),
    )
    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=config,
        leverage=Decimal("3"),
    ).run(SimpleDualLegMaHedgeStrategy().events(_bars()))
    final = result.snapshots[-1]
    assert final.long_quantity > 0
    assert final.short_quantity > 0
    assert result.report["dual_leg_duration_seconds"] > 0
    assert result.report["pnl_reconciliation_error"] == Decimal("0")
    assert result.report["liquidated"] is False
