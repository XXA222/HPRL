from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
from freqtrade.hedge.planning.context import MarketSnapshot, PlannerConfig
from freqtrade.hedge.simulation.exchange import (
    BarEvent,
    FillEvent,
    FundingEvent,
    OrderAcceptedEvent,
    SignalEvent,
)
from freqtrade.hedge.simulation.replay import EventReplayEngine
from freqtrade.optimize.hedge_backtesting import events_from_analyzed_dataframe

START = datetime(2026, 7, 1, tzinfo=UTC)
PAIR = "ETH/USDT:USDT"


def _planner() -> PlannerConfig:
    return PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0.20"),
        tactical_wallet_exposure_long=Decimal("0"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=1,
        cooldown_seconds=0,
        trailing_rebound=Decimal("0"),
    )


def _bar(
    ts: datetime,
    *,
    open_: str = "100",
    high: str = "120",
    low: str = "80",
    close: str = "100",
) -> BarEvent:
    return BarEvent(
        timestamp=ts,
        symbol=PAIR,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def test_replay_activates_orders_on_next_bar_and_preserves_dynamic_target() -> None:
    engine = EventReplayEngine(
        initial_balance=Decimal("1000"),
        planner_config=_planner(),
        leverage=Decimal("3"),
    )
    first = engine.advance(
        [
            SignalEvent(
                START,
                PAIR,
                Decimal("1"),
                Decimal("0"),
                target_net=Decimal("2"),
                model_version="r21-test",
                reason="NO_LOOKAHEAD",
            ),
            _bar(START),
        ]
    )
    assert any(isinstance(item, OrderAcceptedEvent) for item in first.events)
    assert not any(isinstance(item, FillEvent) for item in first.events)
    assert first.report["target_net_quantity"] == Decimal("2")
    checkpoint = engine.checkpoint()
    assert checkpoint.signal_target_net_quantity == Decimal("2")

    second = engine.advance([_bar(START + timedelta(minutes=5))])
    assert any(isinstance(item, FillEvent) for item in second.events)


def test_dataframe_adapter_uses_same_hedge_columns_and_funding_stream() -> None:
    frame = pd.DataFrame(
        {
            "date": [START, START + timedelta(minutes=5)],
            "open": [100, 101],
            "high": [102, 104],
            "low": [99, 100],
            "close": [101, 103],
            "volume": [1000, 1200],
            "hedge_long_score": ["0.75", "0.25"],
            "hedge_short_score": ["0.10", "0.80"],
            "hedge_target_net": ["1.5", "-0.5"],
            "hedge_model_version": ["model-a", "model-b"],
        }
    )
    funding = pd.DataFrame(
        {
            "date": [START + timedelta(minutes=5)],
            "open_fund": ["0.0001"],
            "open_mark": ["102"],
        }
    )
    dataset = events_from_analyzed_dataframe(
        pair=PAIR,
        timeframe="5m",
        frame=frame,
        funding_frame=funding,
        strategy_version="fallback",
    )
    signals = [item for item in dataset.events if isinstance(item, SignalEvent)]
    fundings = [item for item in dataset.events if isinstance(item, FundingEvent)]
    assert dataset.bar_count == 2
    assert dataset.signal_count == 2
    assert dataset.funding_count == 1
    assert signals[0].target_net == Decimal("1.5")
    assert signals[1].model_version == "model-b"
    assert fundings[0].rate == Decimal("0.0001")


def _paper_config() -> dict[str, object]:
    return {
        "managed_pair": PAIR,
        "hedge": {
            "paper": {
                "ephemeral": True,
                "initial_balance": "1000",
                "leverage": "3",
                "auto_fill": True,
                "funding_source": "none",
                "account_events_enabled": False,
                "long_signal": "1",
                "short_signal": "0",
                "bar_volume": "1000",
            },
            "planner": {
                "short_enabled": False,
                "core_wallet_exposure_long": "0.20",
                "tactical_wallet_exposure_long": "0",
                "initial_entry_fraction": "1",
                "max_grid_layers": 1,
                "cooldown_seconds": 0,
                "trailing_rebound": "0",
            },
        },
    }


def _market(ts: datetime, close: str = "100") -> MarketSnapshot:
    mark = Decimal(close)
    return MarketSnapshot(
        symbol=PAIR,
        timestamp=ts,
        bid=mark - Decimal("0.1"),
        ask=mark + Decimal("0.1"),
        mark=mark,
        tick_size=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
    )


def test_paper_uses_next_bar_execution_and_rejects_duplicate_candle() -> None:
    app = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="paper-main",
        symbol=PAIR,
    )
    first_market = _market(START)
    first = app.run_market_cycle(first_market, bar=_bar(START))
    assert first.executions
    assert first.fills == ()

    second_time = START + timedelta(minutes=5)
    second_market = _market(second_time)
    second = app.run_market_cycle(second_market, bar=_bar(second_time))
    assert second.fills
    with pytest.raises(ValueError, match="duplicate candle"):
        app.run_market_cycle(second_market, bar=_bar(second_time))


def test_cli_registers_local_hedge_backtest_and_paper_commands() -> None:
    from freqtrade.commands.arguments import Arguments

    backtest = Arguments(
        [
            "hedge-backtesting",
            "--config",
            "config.json",
            "--strategy",
            "HedgeDualEmaExample",
            "--hedge-export-events",
        ]
    ).get_parsed_arg()
    paper = Arguments(
        [
            "hedge-paper",
            "--config",
            "config.json",
            "--strategy",
            "HedgeDualEmaExample",
        ]
    ).get_parsed_arg()

    assert backtest["command"] == "hedge-backtesting"
    assert backtest["hedge_export_events"] is True
    assert paper["command"] == "hedge-paper"


def test_durable_paper_command_rejects_memory_and_accepts_sqlite_file() -> None:
    from freqtrade.commands.hedge_runtime_commands import _validate_paper_runtime_config
    from freqtrade.exceptions import OperationalException

    config = {
        "hedge_mode_enabled": True,
        "position_mode": "hedge",
        "managed_pair": "ETH/USDT:USDT",
        "trading_mode": "futures",
        "margin_mode": "cross",
        "dry_run": True,
        "db_url": "sqlite:///:memory:",
        "exchange": {
            "name": "binance",
            "pair_whitelist": ["ETH/USDT:USDT"],
        },
        "hedge": {
            "operation_mode": "paper",
            "read_only": True,
            "live_trading_enabled": False,
            "paper": {
                "state_backend": "sql",
                "ohlcv_source": "dataprovider",
                "funding_source": "exchange",
                "account_events_enabled": True,
            },
        },
    }
    with pytest.raises(OperationalException, match="file SQLite"):
        _validate_paper_runtime_config(config)

    config["db_url"] = "sqlite:///user_data/hedge-paper.sqlite"
    _validate_paper_runtime_config(config)


def test_backtest_fails_closed_without_strategy_signal_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": datetime(2026, 8, 1, tzinfo=UTC),
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 10,
            }
        ]
    )
    with pytest.raises(OperationalException, match="produced no"):
        events_from_analyzed_dataframe(
            pair="ETH/USDT:USDT",
            timeframe="5m",
            frame=frame,
        )


def test_backtest_fails_closed_when_exchange_funding_data_was_not_downloaded() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": datetime(2026, 8, 1, tzinfo=UTC),
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 10,
                "hedge_long_score": 1,
                "hedge_short_score": 0,
            }
        ]
    )
    with pytest.raises(OperationalException, match="downloaded funding"):
        events_from_analyzed_dataframe(
            pair="ETH/USDT:USDT",
            timeframe="5m",
            frame=frame,
            require_funding_data=True,
        )


def test_backtest_command_executes_runner_and_prints_result(monkeypatch, capsys) -> None:
    import json
    from types import SimpleNamespace

    from freqtrade.commands.hedge_runtime_commands import start_hedge_backtesting
    import freqtrade.commands.optimize_commands as optimize_commands
    import freqtrade.optimize.hedge_backtesting as hedge_backtesting

    config = {"user_data_dir": "user_data"}
    monkeypatch.setattr(
        optimize_commands,
        "setup_optimize_configuration",
        lambda args, runmode: config,
    )
    fake_run = SimpleNamespace(
        strategy="HedgeDualEmaExample",
        dataset=SimpleNamespace(
            pair=PAIR,
            timeframe="5m",
            start=START,
            end=START + timedelta(minutes=5),
            bar_count=2,
            signal_count=2,
            funding_count=1,
        ),
        market_rule_source="EXCHANGE_MARKETS",
        market_rule_version="rules-v1",
        result=SimpleNamespace(report={"total_return_ratio": Decimal("0.01")}),
        export_path="user_data/backtest_results/hedge-r2.1.json",
    )
    monkeypatch.setattr(
        hedge_backtesting,
        "run_freqtrade_hedge_backtest",
        lambda config, export_path, export_events: fake_run,
    )

    start_hedge_backtesting(
        {
            "hedge_export_filename": "result.json",
            "hedge_export_events": True,
        }
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["execution_timing"] == "NEXT_BAR_NO_LOOKAHEAD"
    assert output["result_file"].endswith("hedge-r2.1.json")
    assert output["artifact_sha256"] == ""
    assert output["result_fingerprint"] == ""
    assert output["native_schema"] == ""
    assert output["freqtrade_projection_embedded"] is False


def test_paper_command_runs_worker_only_after_safety_preflight(monkeypatch) -> None:
    from freqtrade.commands.hedge_runtime_commands import start_hedge_paper
    import freqtrade.configuration as configuration_module
    import freqtrade.worker as worker_module

    config = {
        "hedge_mode_enabled": True,
        "position_mode": "hedge",
        "managed_pair": PAIR,
        "trading_mode": "futures",
        "margin_mode": "cross",
        "dry_run": True,
        "db_url": "sqlite:///user_data/hedge-paper.sqlite",
        "exchange": {"name": "binance", "pair_whitelist": [PAIR]},
        "hedge": {
            "operation_mode": "paper",
            "read_only": True,
            "live_trading_enabled": False,
            "paper": {
                "state_backend": "sql",
                "ohlcv_source": "dataprovider",
                "funding_source": "exchange",
                "account_events_enabled": True,
            },
        },
    }
    calls: list[str] = []

    class FakeConfiguration:
        def __init__(self, args, runmode) -> None:
            del args, runmode

        def get_config(self):
            return config

    class FakeWorker:
        def __init__(self, args, config) -> None:
            del args
            assert config is not None
            calls.append("init")

        def run(self) -> None:
            calls.append("run")

        def exit(self) -> None:
            calls.append("exit")

    monkeypatch.setattr(configuration_module, "Configuration", FakeConfiguration)
    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    monkeypatch.setattr("signal.signal", lambda *args, **kwargs: None)

    assert start_hedge_paper({"command": "hedge-paper"}) == 0
    assert calls == ["init", "run", "exit"]
