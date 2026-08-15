from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from argparse import ArgumentParser
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from freqtrade.hedge.simulation.exchange import FundingEvent


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")

    def timeframe_to_seconds(timeframe: str) -> int:
        value = timeframe.strip().lower()
        amount = int(value[:-1])
        factor = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[value[-1]]
        return amount * factor

    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


class Final14Bt20IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_exchange_stub()

    def _frame(self, long_score: str = "1") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": datetime(2026, 1, 1, tzinfo=UTC),
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                    "volume": "10",
                    "hedge_long_score": long_score,
                    "hedge_short_score": "0",
                },
                {
                    "date": datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                    "open": "100",
                    "high": "102",
                    "low": "98",
                    "close": "101",
                    "volume": "11",
                    "hedge_long_score": long_score,
                    "hedge_short_score": "0",
                },
            ]
        )

    def test_full_event_fingerprint_changes_when_signal_changes(self) -> None:
        from freqtrade.optimize.hedge_backtesting import events_from_analyzed_dataframe

        one = events_from_analyzed_dataframe(
            pair="ETH/USDT:USDT", timeframe="1m", frame=self._frame("1")
        )
        two = events_from_analyzed_dataframe(
            pair="ETH/USDT:USDT", timeframe="1m", frame=self._frame("0.5")
        )
        self.assertNotEqual(one.data_fingerprint, two.data_fingerprint)

    def test_funding_multiplier_is_backtest_only_and_fingerprinted(self) -> None:
        from freqtrade.optimize.hedge_backtesting import events_from_analyzed_dataframe

        funding = pd.DataFrame(
            [
                {
                    "date": datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                    "open_fund": "0.0001",
                    "open_mark": "101",
                }
            ]
        )
        one = events_from_analyzed_dataframe(
            pair="ETH/USDT:USDT",
            timeframe="1m",
            frame=self._frame(),
            funding_frame=funding,
            funding_rate_multiplier=Decimal(1),
        )
        two = events_from_analyzed_dataframe(
            pair="ETH/USDT:USDT",
            timeframe="1m",
            frame=self._frame(),
            funding_frame=funding,
            funding_rate_multiplier=Decimal(2),
        )
        rate_one = next(e.rate for e in one.events if isinstance(e, FundingEvent))
        rate_two = next(e.rate for e in two.events if isinstance(e, FundingEvent))
        self.assertEqual(rate_one, Decimal("0.0001"))
        self.assertEqual(rate_two, Decimal("0.0002"))
        self.assertNotEqual(one.data_fingerprint, two.data_fingerprint)

    def test_native_and_research_optimizer_commands_coexist(self) -> None:
        module_path = Path(__file__).parents[3] / "freqtrade" / "commands" / "hedge_cli.py"
        spec = importlib.util.spec_from_file_location("hedge_cli_bt20_isolated", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class DummyManager:
            def _build_args(self, optionlist, parser):
                return None

        root = ArgumentParser()
        subparsers = root.add_subparsers(dest="command")
        common = ArgumentParser(add_help=False)
        strategy = ArgumentParser(add_help=False)
        module.register_hedge_subcommands(
            DummyManager(),
            subparsers,
            common,
            strategy,
            trade_options=[],
            backtest_options=[],
        )
        self.assertIn("hedge-hyperopt", subparsers.choices)
        self.assertIn("hedge-research-optimize", subparsers.choices)

    def test_dependency_light_import_of_low_level_backtester(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktesting

        self.assertTrue(callable(HedgeBacktesting))


if __name__ == "__main__":
    unittest.main()
