"""Dual-leg machine-learning and reinforcement-learning components for Hedge mode."""
# Public re-export grouping is intentional.
# ruff: noqa: I001

from .config import HedgeRLConfig, RewardWeights
from .contracts import ConfigSchemaVersion, SeedLedger
from .features import FeatureSchema

HEDGE_MLRL_SOURCE_VERSION = "clean-mainline"
HEDGE_MLRL_COMPLETED_ROUNDS = 80
HEDGE_MLRL_ADVANCED_ROUNDS = 80

__all__ = [
    "ConfigSchemaVersion",
    "FeatureSchema",
    "HEDGE_MLRL_ADVANCED_ROUNDS",
    "HEDGE_MLRL_COMPLETED_ROUNDS",
    "HEDGE_MLRL_SOURCE_VERSION",
    "HedgeRLConfig",
    "RewardWeights",
    "SeedLedger",
]
