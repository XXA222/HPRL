"""Runtime convergence coordinator for state, protections, capital and callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from threading import RLock
from typing import Any

from .admission import CompositeAdmissionPolicy
from .callbacks import CallbackCompatibilityMode, HedgeStrategyCallbackAdapter
from .capital import FreqtradeCapitalPolicyAdapter
from .models import (
    AdmissionDecision,
    BotStateSnapshot,
    CapitalSnapshot,
    NativeOrderIntent,
    utc_datetime,
)
from .protections import HedgeProtectionAdapter
from .state import HedgeBotStateAdapter


class NativeConvergenceCoordinator:
    """One decision surface shared by Paper and production-equivalent loops."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        strategy: Any,
        protections: Any,
        pairlocks: Any,
        state_provider: Callable[[], Any],
        capital_provider: Callable[[], tuple[object, object, object]] | None = None,
    ) -> None:
        self.config = config
        self.state_provider = state_provider
        self.capital_provider = capital_provider
        self.state = HedgeBotStateAdapter(
            cancel_open_orders_on_exit=bool(config.get("cancel_open_orders_on_exit", False))
        )
        self.protections = HedgeProtectionAdapter(protections, pairlocks)
        self.capital = FreqtradeCapitalPolicyAdapter(config)
        hedge_raw = config.get("hedge", {})
        hedge = dict(hedge_raw) if isinstance(hedge_raw, Mapping) else {}
        convergence_raw = hedge.get("native_convergence", {})
        convergence = (
            dict(convergence_raw) if isinstance(convergence_raw, Mapping) else {}
        )
        self.callbacks = HedgeStrategyCallbackAdapter(
            strategy,
            mode=CallbackCompatibilityMode(
                str(convergence.get("callback_mode", "native_only"))
            ),
            fail_closed_confirmations=bool(
                convergence.get("fail_closed_confirmations", True)
            ),
        )
        self.model_gate = None
        self.producer_gate = None
        self.universe_gate = None
        self.policy = CompositeAdmissionPolicy()
        self.policy.add(self._admit_state)
        self.policy.add(self._admit_protections)
        self.policy.add(self._admit_capital)
        self.policy.add(self._admit_callbacks)
        self._lock = RLock()
        self._last_state: BotStateSnapshot | None = None
        self._last_capital: CapitalSnapshot | None = None
        self._last_decision: AdmissionDecision | None = None

    def bind_model_gate(self, gate: object) -> None:
        admit = getattr(gate, "admit", None)
        if not callable(admit):
            raise TypeError("model gate must provide admit(intent)")
        self.model_gate = gate
        self.policy.add(admit)

    def bind_producer_gate(self, gate: object) -> None:
        admit = getattr(gate, "admit", None)
        if not callable(admit):
            raise TypeError("producer gate must provide admit(intent)")
        self.producer_gate = gate
        self.policy.add(admit)

    def bind_universe_gate(self, gate: object) -> None:
        admit = getattr(gate, "admit", None)
        if not callable(admit):
            raise TypeError("universe gate must provide admit(intent)")
        self.universe_gate = gate
        self.policy.add(admit)

    def _starting_balance(self) -> float:
        wallet = self.config.get("dry_run_wallet", 0)
        try:
            return float(wallet)
        except (TypeError, ValueError):
            return 0.0

    def state_snapshot(self, *, at: datetime | None = None) -> BotStateSnapshot:
        snapshot = self.state.snapshot(self.state_provider(), at=at)
        with self._lock:
            self._last_state = snapshot
        return snapshot

    def new_risk_enabled(self) -> bool:
        return self.state_snapshot().allow_new_risk

    def planner_enabled(self) -> bool:
        return self.state_snapshot().allow_planner

    def _admit_state(self, intent: NativeOrderIntent) -> AdmissionDecision:
        return self.state.admit(self.state_provider(), intent)

    def _admit_protections(self, intent: NativeOrderIntent) -> AdmissionDecision:
        return self.protections.admit(
            intent,
            now=utc_datetime(),
            starting_balance=self._starting_balance(),
        )

    def capital_snapshot(self) -> CapitalSnapshot | None:
        if self.capital_provider is None:
            return None
        equity, available, gross = self.capital_provider()
        snapshot = self.capital.snapshot(
            equity=equity,
            available_balance=available,
            current_gross_notional=gross,
        )
        with self._lock:
            self._last_capital = snapshot
        return snapshot

    def _admit_capital(self, intent: NativeOrderIntent) -> AdmissionDecision:
        snapshot = self.capital_snapshot()
        if snapshot is None:
            return AdmissionDecision.allow(reason="CAPITAL_PROVIDER_NOT_BOUND")
        return self.capital.admit(intent, snapshot)

    def _admit_callbacks(self, intent: NativeOrderIntent) -> AdmissionDecision:
        return self.callbacks.confirm_intent(intent)

    def transform_planner_intent(self, planner_intent: object) -> object:
        """Apply Hedge-native custom stake/leverage callbacks conservatively."""

        from dataclasses import replace
        from .admission import planner_intent_to_native

        native = planner_intent_to_native(planner_intent)
        max_distance = self.config.get("custom_price_max_distance_ratio", 0.02)
        adjusted_price = self.callbacks.custom_order_price(
            native,
            max_distance_ratio=max_distance,
        )
        callback_intent = replace(native, price=adjusted_price)
        transformed = (
            planner_intent
            if adjusted_price == native.price
            else replace(planner_intent, price=adjusted_price)
        )
        if native.reduce_only:
            return transformed
        configured_leverage = (
            self.config.get("hedge", {}).get("target_leverage", "1")
            if isinstance(self.config.get("hedge", {}), Mapping)
            else "1"
        )
        leverage = self.callbacks.leverage(
            pair=native.pair,
            side=native.side,
            proposed_leverage=configured_leverage,
            max_leverage=configured_leverage,
            current_rate=adjusted_price,
            entry_tag=str(native.metadata.get("tag", "")) or None,
        )
        capital = self.capital_snapshot()
        proposed_stake = callback_intent.notional / leverage
        max_stake = (
            proposed_stake
            if capital is None
            else max(capital.remaining_notional / leverage, Decimal("0"))
        )
        adjusted_stake = self.callbacks.custom_stake_amount(
            callback_intent,
            proposed_stake=proposed_stake,
            max_stake=max_stake,
            leverage=leverage,
        )
        quantity = adjusted_stake * leverage / adjusted_price
        # Native convergence is conservative: callbacks may reduce planner risk but
        # cannot silently enlarge a planner intent. Hyperopt/planner own enlargement.
        quantity = min(quantity, native.quantity)
        if quantity <= Decimal("0"):
            return replace(transformed, quantity=Decimal("0.000000000000000001"))
        if quantity == native.quantity:
            return transformed
        return replace(transformed, quantity=quantity)

    def notify_order_filled(
        self, planner_intent: object, price: object, quantity: object, at: object
    ) -> None:
        from .admission import planner_intent_to_native

        self.callbacks.order_filled(
            intent=planner_intent_to_native(planner_intent),
            fill_price=price,
            fill_quantity=quantity,
            current_time=at,
        )

    def admit(self, intent: NativeOrderIntent) -> AdmissionDecision:
        decision = self.policy.evaluate(intent)
        with self._lock:
            self._last_decision = decision
        return decision

    @staticmethod
    def _gate_status(gate: object | None) -> object:
        if gate is None:
            return None
        status = getattr(gate, "status", None)
        if callable(status):
            return status()
        snapshot = getattr(gate, "snapshot", None)
        if callable(snapshot):
            value = snapshot()
            fields = getattr(type(value), "__dataclass_fields__", {})
            if fields:
                return {
                    name: (
                        getattr(getattr(value, name), "value", getattr(value, name))
                        if not isinstance(getattr(value, name), datetime)
                        else getattr(value, name).isoformat()
                    )
                    for name in fields
                }
        return {"bound": True}

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._last_state
            capital = self._last_capital
            decision = self._last_decision
        return {
            "state": None
            if state is None
            else {
                "source": state.source_state,
                "mode": state.mode.value,
                "allow_planner": state.allow_planner,
                "allow_new_risk": state.allow_new_risk,
                "allow_reduce_only": state.allow_reduce_only,
                "cancel_managed_orders": state.cancel_managed_orders,
            },
            "capital": None
            if capital is None
            else {
                "effective_limit": str(capital.effective_capital_limit),
                "remaining": str(capital.remaining_notional),
                "single_order_limit": str(capital.max_single_order_notional),
            },
            "last_decision": None
            if decision is None
            else {
                "allowed": decision.allowed,
                "code": decision.code.value,
                "reason": decision.reason,
            },
            "admission_provider_count": self.policy.provider_count,
            "model_gate": self._gate_status(self.model_gate),
            "producer_gate": self._gate_status(self.producer_gate),
            "universe_gate": self._gate_status(self.universe_gate),
        }



def build_native_convergence(
    *,
    config: Mapping[str, Any],
    strategy: Any,
    protections: Any,
    pairlocks: Any,
    state_provider: Callable[[], Any],
    capital_provider: Callable[[], tuple[object, object, object]] | None = None,
) -> NativeConvergenceCoordinator:
    return NativeConvergenceCoordinator(
        config=config,
        strategy=strategy,
        protections=protections,
        pairlocks=pairlocks,
        state_provider=state_provider,
        capital_provider=capital_provider,
    )
