import ast
import unittest
from pathlib import Path


HEDGE_METHOD_NAMES = {
    "get_trades_proxy_by_position",
    "get_open_trades_for_pair_side",
    "get_open_trade_for_pair_side",
    "assert_single_open_trade_per_side",
}

UPSTREAM_LOCALTRADE_METHODS = {
    "get_open_trades",
    "get_open_trade_count",
    "from_json",
}


class TestHedgePatchIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(
            "freqtrade/persistence/trade_model.py"
        ).read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _local_trade(self):
        return next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LocalTrade"
        )

    def test_order_restore_keywords_are_unique(self):
        expected_counts = {
            "position_side=order.get(": 1,
            'position_action=order.get("position_action")': 1,
            'action_group_id=order.get("action_group_id")': 1,
        }
        for marker, expected in expected_counts.items():
            with self.subTest(marker=marker):
                self.assertEqual(self.source.count(marker), expected)

    def test_hedge_methods_belong_to_local_trade(self):
        method_names = {
            node.name
            for node in self._local_trade().body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(HEDGE_METHOD_NAMES.issubset(method_names))

    def test_original_local_trade_methods_remain_in_class(self):
        method_names = {
            node.name
            for node in self._local_trade().body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(UPSTREAM_LOCALTRADE_METHODS.issubset(method_names))

    def test_no_hedge_methods_exist_at_module_scope(self):
        module_functions = {
            node.name
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(HEDGE_METHOD_NAMES.isdisjoint(module_functions))

    def test_local_trade_precedes_trade_model(self):
        local_trade = self._local_trade()
        trade = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trade"
        )
        self.assertLess(local_trade.end_lineno, trade.lineno)


if __name__ == "__main__":
    unittest.main()
