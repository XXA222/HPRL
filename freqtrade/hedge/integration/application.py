"""High-level integrated application joining planner, risk and fake execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from freqtrade.hedge.execution.integrated_fake import (
    IntegratedFakeRuntime,
    build_integrated_fake_runtime,
)
from freqtrade.hedge.execution.planner_adapter import adapt_planner_intents
from freqtrade.hedge.execution.service import ExecutionResult
from freqtrade.hedge.planning.context import PlanningContext, PlanningResult
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner


@dataclass(frozen=True, slots=True)
class PlanningExecutionResult:
    planning: PlanningResult
    executions: tuple[ExecutionResult, ...]
    cancellation_results: tuple[ExecutionResult, ...]


class IntegratedFakeHedgeApplication:
    """Executable main path: facts -> planner -> intents -> fake exchange -> ledger."""

    def __init__(
        self,
        *,
        account_id: str = "hedge-main",
        planner: PureHedgePlanner | None = None,
        execution: IntegratedFakeRuntime | None = None,
    ) -> None:
        self.account_id = account_id
        self.planner = planner or PureHedgePlanner()
        self.execution = execution or build_integrated_fake_runtime()
        self._planner_order_to_client: dict[str, str] = {}

    def run_cycle(self, context: PlanningContext) -> PlanningExecutionResult:
        planning = self.planner.plan(context)
        cancellations: list[ExecutionResult] = []
        for planner_order_id in planning.cancel_order_ids:
            client_id = self._planner_order_to_client.get(planner_order_id)
            if client_id is None:
                continue
            cancellations.append(self.execution.engine.cancel(client_id))

        execution_intents = adapt_planner_intents(
            planning.submit_orders,
            account_id=self.account_id,
            strategy_id="pure-hedge-planner",
            cycle_id=context.market.timestamp.isoformat(),
        )
        executions: list[ExecutionResult] = []
        for planner_intent, execution_intent in zip(planning.submit_orders, execution_intents, strict=True):
            result = self.execution.engine.submit(execution_intent)
            executions.append(result)
            self._planner_order_to_client[planner_intent.intent_id] = result.order.client_order_id
        return PlanningExecutionResult(planning, tuple(executions), tuple(cancellations))

    def apply_full_fills(self, results: Iterable[ExecutionResult]) -> tuple[ExecutionResult, ...]:
        updated: list[ExecutionResult] = []
        for result in results:
            order = result.order
            price = order.intent.limit_price
            if price is None:
                raw = order.intent.metadata.get("reference_price")
                if raw is None:
                    raise ValueError("market fake fill requires metadata.reference_price")
                price = raw
            snapshot = self.execution.exchange.fill_order(
                order.client_order_id,
                quantity=order.approved_quantity,
                price=price,
            )
            updated.append(self.execution.engine.apply_exchange_event(snapshot))
        return tuple(updated)
