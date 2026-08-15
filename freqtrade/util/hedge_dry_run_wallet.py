from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from freqtrade.hedge.planning.context import PlannerConfig, StrategyPlanningPort
from freqtrade.hedge.simulation.exchange import (
    MarketRules,
    SimulationInputEvent,
    SimulationResult,
)
from freqtrade.hedge.simulation.matcher import MatchConfig
from freqtrade.hedge.simulation.replay import EventReplayEngine, ReplayCheckpoint


class HedgeDryRunWallet:
    """Dry-run facade sharing planner, matcher, wallet and reports with backtesting."""

    def __init__(
        self,
        *,
        initial_balance: Decimal,
        planner_config: PlannerConfig | None = None,
        leverage: Decimal = Decimal("3"),
        fee_rate: Decimal = Decimal("0.0004"),
        long_signal: Decimal = Decimal("1"),
        short_signal: Decimal = Decimal("1"),
        target_net_quantity: Decimal | None = None,
        market_rules: MarketRules | None = None,
        planner: StrategyPlanningPort | None = None,
        match_config: MatchConfig | None = None,
    ) -> None:
        self.engine = EventReplayEngine(
            initial_balance=initial_balance,
            planner_config=planner_config,
            leverage=leverage,
            fee_rate=fee_rate,
            long_signal=long_signal,
            short_signal=short_signal,
            target_net_quantity=target_net_quantity,
            market_rules=market_rules,
            planner=planner,
            match_config=match_config,
        )

    def replay(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        """Replay a complete scenario from the configured initial balance."""
        return self.engine.replay(events)

    def advance(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        """Process the next chronological dry-run event batch in place."""
        return self.engine.advance(events)

    def checkpoint(self) -> ReplayCheckpoint:
        """Return an isolated in-memory checkpoint for restart or rollback control."""
        return self.engine.checkpoint()

    def restore(self, checkpoint: ReplayCheckpoint) -> None:
        """Restore a checkpoint created by a configuration-compatible dry-run wallet."""
        self.engine.restore(checkpoint)
