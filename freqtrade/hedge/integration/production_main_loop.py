"""Production-equivalent planner -> risk -> execution main loop.

The loop owns orchestration only. Exchange writes remain inside HedgeExecutionEngine and
its production gate. External orders are excluded from planner ownership and are never
canceled by this component.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from typing import Callable, Iterable

from freqtrade.hedge.execution.kill_switch import KillSwitch, KillSwitchSnapshot
from freqtrade.hedge.execution.orchestrator import HedgeExecutionEngine
from freqtrade.hedge.execution.ownership import (
    ExecutionOrderOwnershipRegistry,
    OrderOwnership,
)
from freqtrade.hedge.execution.service import ExecutionResult
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.planning.context import ActiveOrder, PlanningContext, PlanningResult
from freqtrade.hedge.native.admission import (
    CompositeAdmissionPolicy,
    planner_intent_to_native,
)
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner
from freqtrade.hedge.symbols import raw_symbol

from .strategy_state import StrategyStateStorePort

_SUPPORTED_SETTLE_SUFFIXES = ("USDT", "USDC", "FDUSD")
_ACTIVE_STATES = frozenset(
    {
        OrderState.PREPARED,
        OrderState.SUBMITTING,
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIAL,
        OrderState.UNKNOWN,
    }
)


def _valid_perpetual_symbol(value: str) -> bool:
    return (
        value.isascii()
        and value.isalnum()
        and any(value.endswith(suffix) and len(value) > len(suffix) for suffix in _SUPPORTED_SETTLE_SUFFIXES)
    )


class HedgeExecutionMode(StrEnum):
    HEDGE_DISABLED = "HEDGE_DISABLED"
    HEDGE_SIMULATED = "HEDGE_SIMULATED"
    HEDGE_PRODUCTION_LOCKED = "HEDGE_PRODUCTION_LOCKED"
    HEDGE_PRODUCTION_ARMED = "HEDGE_PRODUCTION_ARMED"


class ExecutionEngineKind(StrEnum):
    SIMULATED = "SIMULATED"
    PRODUCTION = "PRODUCTION"


@dataclass(frozen=True, slots=True)
class LoopActionError:
    operation: str
    reference: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class HedgeMainLoopCycle:
    mode: HedgeExecutionMode
    symbol: str
    cycle_id: str
    planning: PlanningResult | None
    submissions: tuple[ExecutionResult, ...] = ()
    cancellations: tuple[ExecutionResult, ...] = ()
    blocked_submit_intent_ids: tuple[str, ...] = ()
    blocked_cancel_order_ids: tuple[str, ...] = ()
    deferred_submit_intent_ids: tuple[str, ...] = ()
    deferred_cancel_order_ids: tuple[str, ...] = ()
    external_order_ids: tuple[str, ...] = ()
    orphan_order_ids: tuple[str, ...] = ()
    errors: tuple[LoopActionError, ...] = ()
    strategy_state_committed: bool = False
    writes_attempted: int = 0

    @property
    def successful(self) -> bool:
        return (
            not self.errors
            and not self.blocked_submit_intent_ids
            and not self.deferred_submit_intent_ids
            and not self.deferred_cancel_order_ids
        )


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    inspected: int
    refreshed: tuple[ExecutionResult, ...]
    resolved_unknown: tuple[ExecutionResult, ...]
    errors: tuple[LoopActionError, ...]


@dataclass(frozen=True, slots=True)
class EmergencyStopReport:
    kill_switch: KillSwitchSnapshot
    canceled: tuple[ExecutionResult, ...]
    errors: tuple[LoopActionError, ...]


@dataclass(frozen=True, slots=True)
class ManagedCancelReport:
    canceled: tuple[ExecutionResult, ...]
    errors: tuple[LoopActionError, ...]


@dataclass(frozen=True, slots=True)
class NewRiskControlSnapshot:
    enabled: bool
    reason: str | None
    actor: str | None


class ProductionEquivalentHedgeMainLoop:
    """One authoritative loop for simulated and production execution modes."""

    def __init__(
        self,
        *,
        account_id: str,
        engine: HedgeExecutionEngine,
        ownership: ExecutionOrderOwnershipRegistry,
        kill_switch: KillSwitch,
        mode: HedgeExecutionMode,
        engine_kind: ExecutionEngineKind,
        planner: PureHedgePlanner | None = None,
        strategy_id: str = "pure-hedge-planner",
        allowed_symbols: Iterable[str] = ("BTCUSDT", "ETHUSDT"),
        state_store: StrategyStateStorePort | None = None,
        max_submissions_per_cycle: int = 32,
        max_cancellations_per_cycle: int = 64,
        block_new_risk_on_external_side: bool = True,
        admission_provider: Callable[[object], object] | None = None,
    ) -> None:
        self.account_id = str(account_id).strip()
        if not self.account_id:
            raise ValueError("account_id is required")
        if not isinstance(engine, HedgeExecutionEngine):
            raise TypeError("engine must be HedgeExecutionEngine")
        self.engine = engine
        self.ownership = ownership
        if not isinstance(kill_switch, KillSwitch):
            raise TypeError("kill_switch must be KillSwitch")
        self.kill_switch = kill_switch
        self.mode = HedgeExecutionMode(mode)
        self.engine_kind = ExecutionEngineKind(engine_kind)
        self.planner = planner or PureHedgePlanner()
        self.strategy_id = str(strategy_id).strip()
        if not self.strategy_id:
            raise ValueError("strategy_id is required")
        normalized = tuple(dict.fromkeys(raw_symbol(item) for item in allowed_symbols))
        if not normalized or not all(_valid_perpetual_symbol(item) for item in normalized):
            raise ValueError(
                "allowed_symbols must contain valid USDT/USDC/FDUSD perpetual symbols"
            )
        self.allowed_symbols = normalized
        self.state_store = state_store
        self.max_submissions_per_cycle = _positive_int(
            max_submissions_per_cycle,
            "max_submissions_per_cycle",
        )
        self.max_cancellations_per_cycle = _positive_int(
            max_cancellations_per_cycle,
            "max_cancellations_per_cycle",
        )
        if not isinstance(block_new_risk_on_external_side, bool):
            raise TypeError("block_new_risk_on_external_side must be boolean")
        self.block_new_risk_on_external_side = block_new_risk_on_external_side
        self._new_risk_control = NewRiskControlSnapshot(True, None, None)
        self._admission_policy = CompositeAdmissionPolicy()
        self._intent_transformers: list[Callable[[object], object]] = []
        if admission_provider is not None:
            self.bind_order_admission_provider(admission_provider)
        self._validate_mode_kind()

    def _validate_mode_kind(self) -> None:
        if (
            self.mode is HedgeExecutionMode.HEDGE_SIMULATED
            and self.engine_kind is not ExecutionEngineKind.SIMULATED
        ):
            raise ValueError("HEDGE_SIMULATED requires a simulated engine")
        if self.mode in {
            HedgeExecutionMode.HEDGE_PRODUCTION_LOCKED,
            HedgeExecutionMode.HEDGE_PRODUCTION_ARMED,
        } and self.engine_kind is not ExecutionEngineKind.PRODUCTION:
            raise ValueError("production modes require a production engine")

    def set_mode(self, mode: HedgeExecutionMode) -> None:
        previous = self.mode
        self.mode = HedgeExecutionMode(mode)
        try:
            self._validate_mode_kind()
        except Exception:
            self.mode = previous
            raise

    @property
    def new_risk_enabled(self) -> bool:
        return self._new_risk_control.enabled

    @property
    def new_risk_control(self) -> NewRiskControlSnapshot:
        return self._new_risk_control

    def stop_new_orders(self, *, reason: str, actor: str) -> NewRiskControlSnapshot:
        reason_text = _control_text(reason, "reason", 1024)
        actor_text = _control_text(actor, "actor", 128)
        self._new_risk_control = NewRiskControlSnapshot(False, reason_text, actor_text)
        return self._new_risk_control

    def resume_new_orders(self, *, actor: str, confirmed: bool) -> NewRiskControlSnapshot:
        if not isinstance(confirmed, bool):
            raise TypeError("confirmed must be boolean")
        if not confirmed:
            raise PermissionError("resuming new risk requires secondary confirmation")
        actor_text = _control_text(actor, "actor", 128)
        self._new_risk_control = NewRiskControlSnapshot(True, None, actor_text)
        return self._new_risk_control

    def bind_order_admission_provider(self, provider: Callable[[object], object]) -> None:
        """Compose a side-aware native admission policy into the authoritative loop."""

        self._admission_policy.add(provider)

    @property
    def order_admission_provider_count(self) -> int:
        return self._admission_policy.provider_count

    def bind_order_transformer(self, transformer: Callable[[object], object]) -> None:
        if not callable(transformer):
            raise TypeError("order transformer must be callable")
        self._intent_transformers.append(transformer)

    def _transform_intent(self, intent: object) -> object:
        current = intent
        for transformer in tuple(self._intent_transformers):
            current = transformer(current)
        return current

    def run_cycle(
        self, context: PlanningContext, *, strategy_allows_new_risk: bool = True
    ) -> HedgeMainLoopCycle:
        symbol = raw_symbol(context.market.symbol)
        if symbol not in self.allowed_symbols:
            raise ValueError(f"symbol is not in the production perpetual allowlist: {symbol}")
        cycle_id = _cycle_id(context, self.strategy_id)
        if self.mode is HedgeExecutionMode.HEDGE_DISABLED:
            return HedgeMainLoopCycle(self.mode, symbol, cycle_id, None)

        context = self._restore_strategy_state(context)
        managed_active, external, orphan, cancel_reference_map = self._partition_orders(
            context.wallet.active_orders
        )
        planning_context = replace(
            context,
            wallet=replace(context.wallet, active_orders=managed_active),
        )
        planning = self.planner.plan(planning_context)

        cancel_candidates = tuple(
            dict.fromkeys(
                (
                    *planning.risk_cancel_order_ids,
                    *planning.cancel_order_ids,
                    *planning.modify_order_ids,
                    *planning.delete_order_ids,
                )
            )
        )
        if self.mode is HedgeExecutionMode.HEDGE_PRODUCTION_LOCKED:
            return HedgeMainLoopCycle(
                mode=self.mode,
                symbol=symbol,
                cycle_id=cycle_id,
                planning=planning,
                blocked_submit_intent_ids=tuple(item.intent_id for item in planning.submit_orders),
                blocked_cancel_order_ids=cancel_candidates,
                external_order_ids=tuple(item.order_id for item in external),
                orphan_order_ids=tuple(item.order_id for item in orphan),
                writes_attempted=0,
            )

        errors: list[LoopActionError] = []
        cancellations: list[ExecutionResult] = []
        submissions: list[ExecutionResult] = []
        blocked: list[str] = []

        cancel_ids = cancel_candidates[: self.max_cancellations_per_cycle]
        deferred_cancels = cancel_candidates[self.max_cancellations_per_cycle :]
        for reference in cancel_ids:
            client_id = cancel_reference_map.get(reference, reference)
            order = self.ownership.resolve_managed_reference(client_id)
            if order is None or order.lifecycle.terminal:
                continue
            try:
                result = self.engine.cancel(order.client_order_id)
                cancellations.append(result)
                if not result.order.lifecycle.terminal:
                    errors.append(
                        LoopActionError(
                            "cancel",
                            reference,
                            "CancellationUnresolved",
                            result.order.lifecycle.status.value,
                        )
                    )
            except Exception as exc:
                errors.append(_action_error("cancel", reference, exc))

        cancel_phase_incomplete = bool(errors or deferred_cancels)
        external_sides = {item.position_side for item in (*external, *orphan)}
        submit_orders = planning.submit_orders[: self.max_submissions_per_cycle]
        deferred_submits = tuple(
            item.intent_id for item in planning.submit_orders[self.max_submissions_per_cycle :]
        )
        for original_planner_intent in submit_orders:
            planner_intent = self._transform_intent(original_planner_intent)
            if cancel_phase_incomplete and not planner_intent.reduce_only:
                blocked.append(planner_intent.intent_id)
                continue
            if (not self.new_risk_enabled or not strategy_allows_new_risk) and not planner_intent.reduce_only:
                blocked.append(planner_intent.intent_id)
                continue
            if (
                self.block_new_risk_on_external_side
                and not planner_intent.reduce_only
                and planner_intent.position_side in external_sides
            ):
                blocked.append(planner_intent.intent_id)
                continue
            native_decision = self._admission_policy.evaluate(
                planner_intent_to_native(planner_intent)
            )
            if not native_decision.allowed:
                blocked.append(planner_intent.intent_id)
                continue
            intent = _adapt_stable(
                planner_intent,
                account_id=self.account_id,
                strategy_id=self.strategy_id,
                cycle_id=cycle_id,
            )
            try:
                submissions.append(self.engine.submit(intent))
            except Exception as exc:
                errors.append(_action_error("submit", planner_intent.intent_id, exc))

        committed = False
        if not errors and not blocked and not deferred_cancels and not deferred_submits:
            committed = self._commit_strategy_state(planning, cycle_id)
        return HedgeMainLoopCycle(
            mode=self.mode,
            symbol=symbol,
            cycle_id=cycle_id,
            planning=planning,
            submissions=tuple(submissions),
            cancellations=tuple(cancellations),
            blocked_submit_intent_ids=tuple(blocked),
            deferred_submit_intent_ids=deferred_submits,
            deferred_cancel_order_ids=tuple(deferred_cancels),
            external_order_ids=tuple(item.order_id for item in external),
            orphan_order_ids=tuple(item.order_id for item in orphan),
            errors=tuple(errors),
            strategy_state_committed=committed,
            writes_attempted=len(cancellations) + len(submissions),
        )

    def recover_pending(self, *, symbol: str | None = None) -> RecoveryReport:
        normalized = None if symbol is None else raw_symbol(symbol)
        if normalized is not None and normalized not in self.allowed_symbols:
            raise ValueError("recovery symbol is not allowlisted")
        orders = self.engine.core.list_orders(
            account_id=self.account_id,
            symbol=normalized,
            include_terminal=False,
        )
        refreshed: list[ExecutionResult] = []
        resolved: list[ExecutionResult] = []
        errors: list[LoopActionError] = []
        for order in orders:
            if order.lifecycle.status is OrderState.UNKNOWN:
                try:
                    resolved.append(self.engine.resolve_unknown(order.client_order_id))
                except Exception as exc:
                    errors.append(_action_error("resolve_unknown", order.client_order_id, exc))
            elif order.lifecycle.status in _ACTIVE_STATES:
                try:
                    refreshed.append(self.engine.refresh_order(order.client_order_id))
                except Exception as exc:
                    errors.append(_action_error("refresh", order.client_order_id, exc))
        return RecoveryReport(len(orders), tuple(refreshed), tuple(resolved), tuple(errors))

    def cancel_managed_orders(self, *, symbol: str | None = None) -> ManagedCancelReport:
        normalized = None if symbol is None else raw_symbol(symbol)
        if normalized is not None and normalized not in self.allowed_symbols:
            raise ValueError("cancel symbol is not allowlisted")
        canceled: list[ExecutionResult] = []
        errors: list[LoopActionError] = []
        for order in self.ownership.managed_open_orders(account_id=self.account_id):
            if order.intent.symbol not in self.allowed_symbols:
                continue
            if normalized is not None and order.intent.symbol != normalized:
                continue
            try:
                canceled.append(self.engine.cancel(order.client_order_id))
            except Exception as exc:
                errors.append(_action_error("managed_cancel", order.client_order_id, exc))
        return ManagedCancelReport(tuple(canceled), tuple(errors))

    def emergency_stop(self, *, reason: str, actor: str) -> EmergencyStopReport:
        snapshot = self.kill_switch.activate(reason=reason, actor=actor)
        self.stop_new_orders(reason=reason, actor=actor)
        report = self.cancel_managed_orders()
        return EmergencyStopReport(snapshot, report.canceled, report.errors)

    def resume_after_emergency(self, *, actor: str, confirmed: bool) -> KillSwitchSnapshot:
        return self.kill_switch.deactivate(actor=actor, confirmed=confirmed)

    def _partition_orders(
        self,
        active_orders: tuple[ActiveOrder, ...],
    ) -> tuple[
        tuple[ActiveOrder, ...],
        tuple[ActiveOrder, ...],
        tuple[ActiveOrder, ...],
        dict[str, str],
    ]:
        managed: list[ActiveOrder] = []
        external: list[ActiveOrder] = []
        orphan: list[ActiveOrder] = []
        references: dict[str, str] = {}
        for active in active_orders:
            client_id = active.client_order_id.strip() or active.order_id
            decision = self.ownership.classify(client_id)
            if decision.ownership is OrderOwnership.MANAGED:
                managed.append(active)
                references[active.order_id] = client_id
            elif decision.ownership is OrderOwnership.ORPHAN_PREFIX:
                orphan.append(active)
            else:
                external.append(active)
        return tuple(managed), tuple(external), tuple(orphan), references

    def _restore_strategy_state(self, context: PlanningContext) -> PlanningContext:
        if self.state_store is None:
            return context
        long_state = self.state_store.load(context.long_state.side) or context.long_state
        short_state = self.state_store.load(context.short_state.side) or context.short_state
        return replace(context, long_state=long_state, short_state=short_state)

    def _commit_strategy_state(self, planning: PlanningResult, decision_id: str) -> bool:
        if self.state_store is None:
            return False
        self.state_store.save(planning.long_state, decision_id=decision_id)
        self.state_store.save(planning.short_state, decision_id=decision_id)
        return True


def _adapt_stable(planner_intent: object, *, account_id: str, strategy_id: str, cycle_id: str):
    from freqtrade.hedge.execution.planner_adapter import adapt_planner_intent

    planner_id = str(getattr(planner_intent, "intent_id"))
    digest = sha256(f"{account_id}|{strategy_id}|{planner_id}".encode("utf-8")).hexdigest()[:32]
    return adapt_planner_intent(
        planner_intent,
        account_id=account_id,
        strategy_id=strategy_id,
        cycle_id=cycle_id,
        idempotency_key=f"r31:{digest}",
    )


def _cycle_id(context: PlanningContext, strategy_id: str) -> str:
    raw = raw_symbol(context.market.symbol)
    material = (
        f"{strategy_id}|{raw}|{context.market.timestamp.isoformat()}|"
        f"{context.market.mark}|{context.long_signal}|{context.short_signal}"
    )
    return "cycle-" + sha256(material.encode("utf-8")).hexdigest()[:24]


def _action_error(operation: str, reference: str, exc: Exception) -> LoopActionError:
    return LoopActionError(operation, str(reference), type(exc).__name__, str(exc)[:1000])


def _positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _control_text(value: object, field_name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result or len(result) > limit or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{field_name} is invalid")
    return result
