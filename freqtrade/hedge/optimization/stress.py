"""Deterministic execution-cost and liquidity stress scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from freqtrade.hedge.optimization.types import exact_decimal


DEFAULT_STRESS_ONE = Decimal(1)
DEFAULT_STRESS_ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class StressScenario:
    name: str
    maker_fee_multiplier: Decimal = DEFAULT_STRESS_ONE
    taker_fee_multiplier: Decimal = DEFAULT_STRESS_ONE
    slippage_bps_add: Decimal = DEFAULT_STRESS_ZERO
    volume_participation_multiplier: Decimal = DEFAULT_STRESS_ONE
    funding_rate_multiplier: Decimal = DEFAULT_STRESS_ONE

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stress scenario name cannot be empty")
        for field_name in (
            "maker_fee_multiplier",
            "taker_fee_multiplier",
            "slippage_bps_add",
            "volume_participation_multiplier",
            "funding_rate_multiplier",
        ):
            value = exact_decimal(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, value)
        if self.maker_fee_multiplier < 0 or self.taker_fee_multiplier < 0:
            raise ValueError("fee stress multipliers cannot be negative")
        if self.slippage_bps_add < 0:
            raise ValueError("slippage stress cannot be negative")
        if self.funding_rate_multiplier < 0:
            raise ValueError("funding stress multiplier cannot be negative")
        if not Decimal(0) < self.volume_participation_multiplier <= Decimal(1):
            raise ValueError("volume participation multiplier must be in (0, 1]")


BASELINE_SCENARIO = StressScenario("baseline")


def _paper(config: dict[str, Any]) -> dict[str, Any]:
    hedge = config.setdefault("hedge", {})
    if not isinstance(hedge, dict):
        raise TypeError("hedge configuration must be an object")
    paper = hedge.setdefault("paper", {})
    if not isinstance(paper, dict):
        raise TypeError("hedge.paper configuration must be an object")
    return paper


def _configured_decimal(
    mapping: Mapping[str, object], name: str, default: str
) -> Decimal:
    return exact_decimal(mapping.get(name, default), field_name=f"hedge.paper.{name}")


def apply_stress_to_config(
    base_config: Mapping[str, Any], scenario: StressScenario
) -> dict[str, Any]:
    """Apply only simulation-cost stresses; live gates and credentials are untouched."""

    config = deepcopy(dict(base_config))
    paper = _paper(config)
    maker = _configured_decimal(paper, "maker_fee_rate", "0.0004")
    taker = _configured_decimal(paper, "taker_fee_rate", "0.0004")
    slippage = _configured_decimal(paper, "market_slippage_bps", "0")
    participation = _configured_decimal(paper, "volume_participation", "0.10")
    paper["maker_fee_rate"] = maker * scenario.maker_fee_multiplier
    paper["taker_fee_rate"] = taker * scenario.taker_fee_multiplier
    paper["market_slippage_bps"] = slippage + scenario.slippage_bps_add
    paper["volume_participation"] = max(
        Decimal("0.00000001"),
        participation * scenario.volume_participation_multiplier,
    )
    # The evaluator consumes this backtest-only metadata and scales FundingEvent
    # rates without changing the account/runtime configuration contract.
    optimization = config.setdefault("hedge_optimization_runtime", {})
    if not isinstance(optimization, dict):
        raise TypeError("hedge_optimization_runtime must be an object")
    optimization["stress_name"] = scenario.name
    optimization["maker_fee_multiplier"] = scenario.maker_fee_multiplier
    optimization["taker_fee_multiplier"] = scenario.taker_fee_multiplier
    optimization["funding_rate_multiplier"] = scenario.funding_rate_multiplier
    return config


def default_stress_scenarios() -> tuple[StressScenario, ...]:
    return (
        BASELINE_SCENARIO,
        StressScenario(
            "cost_2x",
            maker_fee_multiplier=Decimal(2),
            taker_fee_multiplier=Decimal(2),
            slippage_bps_add=Decimal(2),
        ),
        StressScenario(
            "thin_liquidity",
            volume_participation_multiplier=Decimal("0.5"),
            slippage_bps_add=Decimal(5),
        ),
        StressScenario("funding_2x", funding_rate_multiplier=Decimal(2)),
    )
