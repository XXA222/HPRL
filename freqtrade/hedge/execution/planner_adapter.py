"""Compatibility import for the canonical cross-direction adapter boundary."""

from freqtrade.hedge.contracts.adapters import (
    adapt_planner_intent,
    adapt_planner_intents,
    assert_internal_contract_compatibility,
)

__all__ = [
    "adapt_planner_intent",
    "adapt_planner_intents",
    "assert_internal_contract_compatibility",
]
