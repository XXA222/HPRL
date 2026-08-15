import importlib.util
import unittest
from argparse import ArgumentParser
from pathlib import Path


MODULE_PATH = Path(__file__).parents[3] / "freqtrade" / "commands" / "hedge_cli.py"
SPEC = importlib.util.spec_from_file_location("hedge_cli_isolated", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
register_hedge_subcommands = MODULE.register_hedge_subcommands


class DummyManager:
    def _build_args(self, optionlist, parser):
        return None


class HedgeCliRegistrationTest(unittest.TestCase):
    def test_hedge_research_optimizer_parser_and_overrides_are_registered(self) -> None:
        root = ArgumentParser()
        subparsers = root.add_subparsers(dest="command")
        common = ArgumentParser(add_help=False)
        strategy = ArgumentParser(add_help=False)
        register_hedge_subcommands(
            DummyManager(),
            subparsers,
            common,
            strategy,
            trade_options=[],
            backtest_options=[],
        )
        args = vars(
            root.parse_args(
                [
                    "hedge-research-optimize",
                    "--hedge-study-name",
                    "demo",
                    "--hedge-trials",
                    "12",
                    "--hedge-workers",
                    "2",
                    "--hedge-sampler",
                    "grid",
                ]
            )
        )
        self.assertEqual(args["command"], "hedge-research-optimize")
        self.assertEqual(args["hedge_study_name"], "demo")
        self.assertEqual(args["hedge_trials"], 12)
        self.assertEqual(args["hedge_workers"], 2)
        self.assertEqual(args["hedge_sampler"], "grid")
        self.assertTrue(callable(args["func"]))

    def test_registration_is_idempotent(self) -> None:
        root = ArgumentParser()
        subparsers = root.add_subparsers(dest="command")
        common = ArgumentParser(add_help=False)
        strategy = ArgumentParser(add_help=False)
        kwargs = {"trade_options": [], "backtest_options": []}
        register_hedge_subcommands(DummyManager(), subparsers, common, strategy, **kwargs)
        register_hedge_subcommands(DummyManager(), subparsers, common, strategy, **kwargs)
        self.assertIn("hedge-research-optimize", subparsers.choices)
        self.assertIn("hedge-hyperopt", subparsers.choices)


if __name__ == "__main__":
    unittest.main()
