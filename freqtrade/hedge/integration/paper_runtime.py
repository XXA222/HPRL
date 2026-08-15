"""Runnable hedge paper loop joining planning, risk and fake execution.

This module is the operational composition missing from the five independent
feature directions.  It keeps planner state between cycles, rebuilds the
planner wallet from actual execution fills and uses the direction-three risk
engine before every direction-five submission.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID
from threading import RLock
from typing import Any, Callable, Mapping

from freqtrade.enums.hedge import PositionAction as RiskAction
from freqtrade.enums.hedge import PositionSide as RiskPositionSide
from freqtrade.hedge.execution.action_group_store import ActionGroupRepository
from freqtrade.hedge.execution.event_publisher import InMemoryEventPublisher
from freqtrade.hedge.execution.idempotency import IdempotencyPort, ReservationState
from freqtrade.hedge.execution.integrated_fake import (
    IntegratedFakeRuntime,
    build_integrated_fake_runtime,
)
from freqtrade.hedge.execution.ledger import InMemoryExecutionLedger
from freqtrade.hedge.execution.planner_adapter import adapt_planner_intents
from freqtrade.hedge.execution.service import (
    ApprovedOrderIntent,
    ExecutionBlockedError,
    ExecutionOrder,
    ExecutionResult,
    ExecutionStorePort,
    InMemoryExecutionStore,
    ExternalOrderSnapshot,
    IntentAction as ExecutionAction,
    OrderIntent as ExecutionOrderIntent,
    OrderType as ExecutionOrderType,
    PositionSide as ExecutionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.hedge.integration.risk_adapter import PortfolioRiskApprovalAdapter
from freqtrade.hedge.paper_config import PaperOhlcvSource, PaperSimulationConfig
from freqtrade.hedge.integration.paper_events import (
    NullPaperAccountEventSink,
    NullPaperExecutionRecovery,
    PaperAccountEventSink,
    PaperExecutionRecoveryPort,
    fee_account_event,
    funding_account_event,
)
from freqtrade.hedge.planning.context import (
    ActiveOrder,
    LegPosition,
    IntentAction,
    MarketSnapshot,
    OrderIntent as PlannerOrderIntent,
    OrderSide,
    OrderType as PlannerOrderType,
    PlannerConfig,
    PlanningContext,
    PlanningResult,
    PositionBucket,
    PositionSide,
    StrategyLegState,
    TimeInForce,
    WalletSnapshot,
)
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner
from freqtrade.hedge.risk.engine import HedgeRiskEngine
from freqtrade.hedge.contracts.ports import (
    MarketRules as ExecutionMarketRules,
    MarketRulesPort,
    PositionKey as ContractPositionKey,
    PositionLockPort,
    ReadinessGatePort,
    SingleWriterPort,
)
from freqtrade.hedge.contracts.types import PositionSide as ContractPositionSide
from freqtrade.hedge.integration.paper_state import NullPaperStateStore, PaperStateStore
from freqtrade.hedge.integration.candle_cursor import bar_fingerprint
from freqtrade.hedge.integration.paper_risk_gate import apply_new_risk_gate
from freqtrade.hedge.native.admission import (
    CompositeAdmissionPolicy,
    apply_planning_admission_gate,
)
from freqtrade.hedge.native.callbacks import HedgeStrategyCallbackAdapter
from freqtrade.hedge.native.exit_overlay import NativeExitOverlay, policies_from_config
from freqtrade.hedge.native.exits import HedgeExitPolicyEngine
from freqtrade.hedge.control.dryrun import DryRunControlState
from freqtrade.hedge.operations.config import operations_config
from freqtrade.hedge.strategies.contract import (
    StrategyDirective,
    planner_config_for_directive,
    target_net_quantity_for_directive,
)
from freqtrade.hedge.telemetry.dryrun import (
    DryRunCycleTelemetry,
    DryRunTelemetryStore,
    JsonlDryRunTelemetryStore,
    StrategyTelemetry,
)
from freqtrade.hedge.operations.runtime import DryRunOperationsRuntime, OperationsCycleInput
from freqtrade.hedge.runtime import HedgeProjectionSource
from freqtrade.hedge.simulation.cross_wallet import CrossWallet
from freqtrade.hedge.simulation.exchange import AccountEvent, BarEvent, FundingEvent
from freqtrade.hedge.simulation.matcher import ConservativeMatcher, MatchConfig
from freqtrade.hedge.risk.limits import RiskLimits
from freqtrade.hedge.risk.models import PendingOrderRisk
from freqtrade.hedge.symbols import raw_symbol
from freqtrade.hedge.risk.portfolio import (
    PositionRiskLeg,
    RiskPortfolioSnapshot,
    build_risk_portfolio,
)

ZERO = Decimal("0")
ONE = Decimal("1")


def _decimal(value: object, default: str) -> Decimal:
    if value is None:
        return Decimal(default)
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("paper runtime decimal configuration must be finite")
    return result


def planner_config_from_mapping(values: Mapping[str, Any] | None) -> PlannerConfig:
    raw = dict(values or {})
    aliases = {
        "qty_scale": "grid_qty_growth",
        "grid_initial_distance": "trailing_trigger_distance",
    }
    for old_name, new_name in aliases.items():
        if old_name not in raw:
            continue
        if new_name in raw and raw[new_name] != raw[old_name]:
            raise ValueError(
                f"hedge.planner.{old_name} conflicts with hedge.planner.{new_name}"
            )
        raw[new_name] = raw.pop(old_name)
    fields = PlannerConfig.__dataclass_fields__
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            "unknown hedge.planner option(s): " + ", ".join(unknown)
        )
    converted: dict[str, object] = {}
    for name, value in raw.items():
        default = fields[name].default
        if isinstance(default, Decimal):
            if isinstance(value, bool):
                raise ValueError(f"hedge.planner.{name} must be an exact decimal")
            converted[name] = Decimal(str(value))
        elif isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"hedge.planner.{name} must be an integer")
            converted[name] = value
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"hedge.planner.{name} must be a boolean")
            converted[name] = value
        else:
            converted[name] = value
    return PlannerConfig(**converted)


def _risk_limits(values: Mapping[str, Any], initial_balance: Decimal) -> RiskLimits:
    max_gross = values.get("max_gross_notional")
    max_ratio = values.get("max_gross_exposure_ratio", "0.80")
    max_single = values.get("max_single_order_notional")
    if max_single is None:
        max_single = initial_balance * Decimal("0.25")
    return RiskLimits(
        max_margin_utilization=_decimal(values.get("max_margin_utilization"), "0.80"),
        min_liquidation_buffer_ratio=_decimal(
            values.get("min_liquidation_buffer_ratio"), "0.05"
        ),
        max_gross_notional=(None if max_gross is None else _decimal(max_gross, "0")),
        max_gross_exposure_ratio=(
            None if max_ratio is None else _decimal(max_ratio, "0.80")
        ),
        max_single_order_notional=_decimal(max_single, "0"),
    )


@dataclass(frozen=True, slots=True)
class PaperCycleResult:
    planning: PlanningResult
    executions: tuple[ExecutionResult, ...]
    fills: tuple[ExecutionResult, ...]
    cancellations: tuple[ExecutionResult, ...]
    wallet: WalletSnapshot
    account_events: tuple[AccountEvent, ...] = ()


@dataclass(slots=True)
class _BucketState:
    core_quantity: Decimal = ZERO
    core_average: Decimal = ZERO
    core_opened_at: datetime | None = None
    tactical_quantity: Decimal = ZERO
    tactical_average: Decimal = ZERO
    tactical_opened_at: datetime | None = None

    def increase(
        self,
        bucket: PositionBucket,
        quantity: Decimal,
        price: Decimal,
        opened_at: datetime | None = None,
    ) -> None:
        opened_at = datetime.now(UTC) if opened_at is None else opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=UTC)
        else:
            opened_at = opened_at.astimezone(UTC)
        if bucket is PositionBucket.CORE:
            total = self.core_quantity + quantity
            self.core_average = (
                (self.core_quantity * self.core_average + quantity * price) / total
            )
            if self.core_quantity == ZERO:
                self.core_opened_at = opened_at
            self.core_quantity = total
        else:
            total = self.tactical_quantity + quantity
            self.tactical_average = (
                (self.tactical_quantity * self.tactical_average + quantity * price) / total
            )
            if self.tactical_quantity == ZERO:
                self.tactical_opened_at = opened_at
            self.tactical_quantity = total

    def reduce(self, bucket: PositionBucket, quantity: Decimal) -> None:
        remaining = quantity
        if bucket is PositionBucket.CORE:
            used = min(remaining, self.core_quantity)
            self.core_quantity -= used
            remaining -= used
            if self.core_quantity == 0:
                self.core_average = ZERO
                self.core_opened_at = None
        else:
            used = min(remaining, self.tactical_quantity)
            self.tactical_quantity -= used
            remaining -= used
            if self.tactical_quantity == 0:
                self.tactical_average = ZERO
                self.tactical_opened_at = None
        # A planner reduction may intentionally span the tactical/core boundary.
        if remaining > 0:
            if bucket is PositionBucket.CORE:
                self.tactical_quantity = max(self.tactical_quantity - remaining, ZERO)
                if self.tactical_quantity == 0:
                    self.tactical_average = ZERO
                    self.tactical_opened_at = None
            else:
                self.core_quantity = max(self.core_quantity - remaining, ZERO)
                if self.core_quantity == 0:
                    self.core_average = ZERO
                    self.core_opened_at = None


class IntegratedPaperHedgeApplication:
    """Stateful paper application suitable for the real Freqtrade process loop."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        account_id: str,
        symbol: str,
        publisher: object | None = None,
        execution_runtime: IntegratedFakeRuntime | None = None,
        build_execution: bool = True,
        state_store: PaperStateStore | None = None,
        account_event_sink: PaperAccountEventSink | None = None,
        execution_recovery: PaperExecutionRecoveryPort | None = None,
    ) -> None:
        hedge_raw = config.get("hedge", {})
        hedge = dict(hedge_raw) if isinstance(hedge_raw, Mapping) else {}
        planner_values = hedge.get("planner", {})
        if not isinstance(planner_values, Mapping):
            planner_values = {}
        self.paper_config = PaperSimulationConfig.from_hedge_mapping(hedge)
        self.account_id = account_id
        self.symbol = symbol
        self.execution_symbol = raw_symbol(symbol)
        self.initial_balance = self.paper_config.initial_balance
        self.leverage = self.paper_config.leverage
        self.auto_fill = self.paper_config.auto_fill
        self.fill_model = self.paper_config.fill_model
        self.default_long_signal = self.paper_config.default_long_signal
        self.default_short_signal = self.paper_config.default_short_signal
        self.planner_config = planner_config_from_mapping(planner_values)
        dashboard_raw = hedge.get("dashboard", {})
        dashboard = dict(dashboard_raw) if isinstance(dashboard_raw, Mapping) else {}
        self.dashboard_enabled = bool(dashboard.get("enabled", False))
        telemetry_capacity = int(dashboard.get("telemetry_capacity", 2000))
        telemetry_backend = str(dashboard.get("telemetry_backend", "memory")).lower()
        if telemetry_backend == "jsonl":
            telemetry_path = str(dashboard.get("telemetry_path", "user_data/hedge/telemetry/dryrun_cycles.jsonl"))
            self.telemetry = JsonlDryRunTelemetryStore(telemetry_path, capacity=telemetry_capacity)
        else:
            self.telemetry = DryRunTelemetryStore(telemetry_capacity)
        control_path = dashboard.get("control_state_path")
        self.dryrun_control = DryRunControlState(None if control_path in {None, ""} else str(control_path))
        operations_values = dict(operations_config(config))
        self.operations_error: str | None = None
        self.operations: DryRunOperationsRuntime | None = None
        if bool(operations_values.get("enabled", False)):
            state_path = operations_values.get(
                "state_path",
                "user_data/hedge/operations/runtime-state.json",
            )
            self.operations = DryRunOperationsRuntime(
                account_id=account_id,
                symbols=(symbol,),
                config={"hedge": {"operations": operations_values}},
                state_path=None if state_path in {None, ""} else str(state_path),
            )
        self._new_risk_enabled_providers: list[Callable[[], bool]] = []
        self._order_admission_policy = CompositeAdmissionPolicy()
        self._intent_transformers: list[Callable[[object], object]] = []
        self._fill_observers: list[Callable[[object, object, object, object], None]] = []
        self.add_new_risk_provider(
            lambda: self.dryrun_control.snapshot().new_risk_enabled
            and (
                self.operations is None
                or (
                    self.operations_error is None
                    and self.operations.latest is not None
                    and self.operations.latest.new_risk_enabled
                )
            )
        )
        self.planner = PureHedgePlanner()
        self._exit_overlay = NativeExitOverlay(
            HedgeExitPolicyEngine(policies_from_config(config))
        )
        self.long_state = StrategyLegState(PositionSide.LONG)
        self.short_state = StrategyLegState(PositionSide.SHORT)
        self._bucket = {
            PositionSide.LONG: _BucketState(),
            PositionSide.SHORT: _BucketState(),
        }
        self._planner_order_to_client: dict[str, str] = {}
        self._simulation_intents: dict[str, PlannerOrderIntent] = {}
        self._last_market: MarketSnapshot | None = None
        self._last_bar: BarEvent | None = None
        self._cycle_market: MarketSnapshot | None = None
        self._lock = RLock()
        self._state_store = state_store or NullPaperStateStore()
        self._state_loaded = False
        self._requires_restart = False
        self._state_durable = not isinstance(self._state_store, NullPaperStateStore)
        self._account_event_sink = account_event_sink or NullPaperAccountEventSink()
        self._execution_recovery = execution_recovery or NullPaperExecutionRecovery()
        self._applied_account_event_ids: set[str] = set()
        self._funding_balance_delta = ZERO
        self._last_funding_event_time: datetime | None = None
        self._paper_fee_rate = self.paper_config.taker_fee_rate
        self._bar_volume = self.paper_config.bar_volume
        self.matcher = ConservativeMatcher(
            MatchConfig(
                maker_fee_rate=self.paper_config.maker_fee_rate,
                taker_fee_rate=self.paper_config.taker_fee_rate,
                volume_participation=self.paper_config.volume_participation,
                market_slippage_bps=self.paper_config.market_slippage_bps,
                price_tick=self.paper_config.tick_size,
                qty_step=self.paper_config.qty_step,
                min_fill_qty=self.paper_config.min_qty,
                min_fill_notional=self.paper_config.min_notional,
                max_entry_layers_per_bar=self.paper_config.max_entry_layers_per_bar,
                max_reduce_layers_per_bar=self.paper_config.max_reduce_layers_per_bar,
                max_fill_ratio_per_order=self.paper_config.max_fill_ratio_per_order,
                max_fills_per_bar=self.paper_config.max_fills_per_bar,
            )
        )
        self.execution: IntegratedFakeRuntime | None = execution_runtime
        if self.execution is None and build_execution:
            risk_engine = HedgeRiskEngine(_risk_limits(hedge, self.initial_balance))
            risk = PortfolioRiskApprovalAdapter(
                engine=risk_engine,
                portfolio_provider=self.risk_portfolio,
            )
            self.execution = build_integrated_fake_runtime(
                risk=risk,
                publisher=publisher,
                fee_rate=self._paper_fee_rate,
            )
        if self.execution is not None:
            self._restore_state()

    def bind_execution(
        self,
        *,
        risk: object,
        readiness: ReadinessGatePort,
        single_writer: SingleWriterPort,
        position_lock: PositionLockPort,
        market_rules: MarketRulesPort,
        publisher: object | None = None,
        action_groups: ActionGroupRepository | None = None,
        transaction: object | None = None,
        store: ExecutionStorePort | None = None,
        idempotency: IdempotencyPort[ExecutionResult] | None = None,
        account_event_sink: PaperAccountEventSink | None = None,
    ) -> None:
        """Bind the authoritative direction-three/direction-five graph exactly once."""

        with self._lock:
            if self.execution is not None:
                raise RuntimeError("paper execution runtime is already bound")
            transaction_port = transaction or InMemoryExecutionLedger()
            event_publisher = publisher or InMemoryEventPublisher()
            self.execution = build_integrated_fake_runtime(
                risk=risk,  # type: ignore[arg-type]
                publisher=event_publisher,  # type: ignore[arg-type]
                readiness=readiness,
                single_writer=single_writer,
                position_lock=position_lock,
                market_rules=market_rules,
                transaction=transaction_port,  # type: ignore[arg-type]
                action_groups=action_groups,
                store=store,
                idempotency=idempotency,
                fee_rate=self._paper_fee_rate,
                strict_dependencies=True,
            )
            if account_event_sink is not None:
                self._account_event_sink = account_event_sink
            self._restore_state()

    def _execution(self) -> IntegratedFakeRuntime:
        if self.execution is None:
            raise RuntimeError("paper execution runtime has not been bound")
        return self.execution

    def bind_new_risk_provider(self, provider: Callable[[], bool]) -> None:
        """Add a control-plane gate to Paper order submission.

        The runtime composes the built-in Dry-run operations gate with external
        control-plane providers instead of overwriting either authority.  Providers are now composed with logical AND so official
        bot state, the Hedge control plane, readiness and the production-equivalent loop
        must all agree before new risk is allowed.
        """

        self.add_new_risk_provider(provider)

    def add_new_risk_provider(self, provider: Callable[[], bool]) -> None:
        if not callable(provider):
            raise TypeError("new-risk provider must be callable")
        self._new_risk_enabled_providers.append(provider)

    @property
    def new_risk_provider_count(self) -> int:
        return len(self._new_risk_enabled_providers)

    def bind_order_admission_provider(self, provider: Callable[[object], object]) -> None:
        """Add a side-aware admission provider for every planned submission."""

        self._order_admission_policy.add(provider)

    @property
    def order_admission_provider_count(self) -> int:
        return self._order_admission_policy.provider_count

    def bind_order_transformer(self, transformer: Callable[[object], object]) -> None:
        if not callable(transformer):
            raise TypeError("order transformer must be callable")
        self._intent_transformers.append(transformer)

    def add_fill_observer(
        self, observer: Callable[[object, object, object, object], None]
    ) -> None:
        if not callable(observer):
            raise TypeError("fill observer must be callable")
        self._fill_observers.append(observer)

    def _transform_planning_intents(self, planning: PlanningResult) -> PlanningResult:
        if not self._intent_transformers:
            return planning
        transformed: list[PlannerOrderIntent] = []
        for item in planning.submit_orders:
            current: object = item
            for transformer in tuple(self._intent_transformers):
                current = transformer(current)
            if not isinstance(current, PlannerOrderIntent):
                raise TypeError("Paper order transformer must return PlannerOrderIntent")
            transformed.append(current)
        return replace(planning, submit_orders=tuple(transformed))

    def _notify_fill_observers(
        self, planner_intent: object, price: object, quantity: object, at: object
    ) -> None:
        for observer in tuple(self._fill_observers):
            try:
                observer(planner_intent, price, quantity, at)
            except Exception:
                # Execution/fill authority must never be rolled back by notifications.
                continue

    def new_risk_enabled(self) -> bool:
        for provider in tuple(self._new_risk_enabled_providers):
            try:
                value = provider()
            except Exception as exc:
                raise RuntimeError("Paper new-risk provider failed closed") from exc
            if not isinstance(value, bool):
                raise TypeError("Paper new-risk provider must return bool")
            if not value:
                return False
        return True

    def cancel_managed_orders(self) -> tuple[tuple[ExecutionResult, ...], tuple[str, ...]]:
        """Cancel every Paper order owned by the Hedge planner.

        This is the Hedge equivalent of ``cancel_open_orders_on_exit``.  It never
        touches external exchange orders because the mapping contains only order IDs
        created by this Paper application's planner/execution graph.
        """

        canceled: list[ExecutionResult] = []
        errors: list[str] = []
        with self._lock:
            execution = self._execution()
            for planner_id, client_id in tuple(self._planner_order_to_client.items()):
                terminal = False
                try:
                    canceled.append(execution.engine.cancel(client_id))
                    terminal = True
                except (KeyError, ValueError):
                    # Already terminal or absent is idempotent success for shutdown.
                    terminal = True
                except Exception as exc:
                    errors.append(f"{client_id}:{type(exc).__name__}:{exc}")
                if terminal:
                    self._simulation_intents.pop(client_id, None)
                    self._planner_order_to_client.pop(planner_id, None)
        return tuple(canceled), tuple(errors)

    @property
    def last_funding_event_time(self) -> datetime | None:
        return self._last_funding_event_time

    @property
    def last_market(self) -> MarketSnapshot | None:
        with self._lock:
            return self._last_market

    @property
    def last_bar(self) -> BarEvent | None:
        with self._lock:
            return self._last_bar

    @property
    def last_bar_fingerprint(self) -> str | None:
        with self._lock:
            return None if self._last_bar is None else bar_fingerprint(self._last_bar)

    def _fake_leg(self, side: PositionSide):
        return self._execution().account.leg(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            position_side=ExecutionSide(side.value),
        )

    def _leg(self, side: PositionSide) -> LegPosition:
        fake = self._fake_leg(side)
        bucket = self._bucket[side]
        bucket_total = bucket.core_quantity + bucket.tactical_quantity
        # Existing seeded positions have no strategy label. Treat the unlabelled
        # remainder as tactical so planning can continue instead of crashing.
        if bucket_total != fake.quantity:
            delta = fake.quantity - bucket_total
            if delta > 0:
                combined = bucket.tactical_quantity + delta
                bucket.tactical_average = (
                    fake.average_price
                    if combined > 0
                    else ZERO
                )
                bucket.tactical_quantity = combined
            elif delta < 0:
                bucket.reduce(PositionBucket.TACTICAL, -delta)
        return LegPosition(
            side=side,
            quantity=fake.quantity,
            average_price=fake.average_price,
            core_quantity=bucket.core_quantity,
            core_average_price=bucket.core_average,
            tactical_quantity=bucket.tactical_quantity,
            tactical_average_price=bucket.tactical_average,
            realized_pnl=fake.realized_pnl,
            tactical_realized_pnl=fake.realized_pnl,
        )

    def _active_execution_orders(self) -> tuple[ExecutionOrder, ...]:
        """Return non-terminal execution orders without projecting planner fields.

        Recovery assertions and execution diagnostics need the durable lifecycle
        and approved quantity.  Planner consumers must continue to use
        :meth:`_active_orders`, which intentionally exposes only the planning
        projection and remaining quantity.
        """
        return self._execution().core.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            include_terminal=False,
        )

    def _active_orders(self) -> tuple[ActiveOrder, ...]:
        rows: list[ActiveOrder] = []
        for order in self._active_execution_orders():
            price = order.intent.limit_price
            if price is None:
                raw = order.intent.metadata.get("reference_price")
                if raw is None:
                    continue
                price = Decimal(str(raw))
            bucket = PositionBucket(str(order.intent.metadata.get("bucket", "TACTICAL")))
            rows.append(
                ActiveOrder(
                    order_id=str(
                        order.intent.metadata.get(
                            "planner_intent_id",
                            order.client_order_id,
                        )
                    ),
                    client_order_id=order.client_order_id,
                    symbol=self.symbol,
                    position_side=PositionSide(order.intent.position_side.value),
                    order_side=(
                        OrderSide.BUY
                        if (
                            order.intent.position_side is ExecutionSide.LONG
                            and not order.intent.reduces_risk
                        )
                        or (
                            order.intent.position_side is ExecutionSide.SHORT
                            and order.intent.reduces_risk
                        )
                        else OrderSide.SELL
                    ),
                    quantity=order.approved_quantity - order.lifecycle.filled_quantity,
                    price=price,
                    reduce_only=order.intent.reduce_only,
                    bucket=bucket,
                    action=IntentAction(
                        str(order.intent.metadata.get("strategy_action", order.intent.action.value))
                    ),
                    created_at=order.created_at,
                    layer=int(order.intent.metadata.get("layer", 0)),
                )
            )
        return tuple(rows)

    def wallet(self, market: MarketSnapshot | None = None) -> WalletSnapshot:
        current_market = market or self.last_market
        mark = Decimal("1") if current_market is None else current_market.mark
        long = self._leg(PositionSide.LONG)
        short = self._leg(PositionSide.SHORT)
        realized = long.realized_pnl + short.realized_pnl
        fees = self._fake_leg(PositionSide.LONG).fees + self._fake_leg(PositionSide.SHORT).fees
        unrealized = long.unrealized_pnl(mark) + short.unrealized_pnl(mark)
        balance = self.initial_balance + realized - fees + self._funding_balance_delta
        equity = balance + unrealized
        initial_margin = (long.quantity + short.quantity) * mark / self.leverage
        pending_margin = sum(
            (
                item.notional / self.leverage
                for item in self._active_orders()
                if not item.reduce_only
            ),
            ZERO,
        )
        available = max(equity - initial_margin - pending_margin, ZERO)
        return WalletSnapshot(
            balance=balance,
            equity=max(equity, Decimal("0.00000001")),
            available_balance=available,
            long=long,
            short=short,
            active_orders=self._active_orders(),
            leverage=self.leverage,
        )

    def risk_portfolio(self) -> RiskPortfolioSnapshot:
        market = self._cycle_market or self.last_market
        mark = Decimal("1") if market is None else market.mark
        wallet = self.wallet(market)
        positions: list[PositionRiskLeg] = []
        for side, leg in (
            (RiskPositionSide.LONG, wallet.long),
            (RiskPositionSide.SHORT, wallet.short),
        ):
            if leg.quantity <= 0:
                continue
            maintenance = leg.quantity * mark * self.planner_config.maintenance_margin_rate
            if side is RiskPositionSide.LONG:
                liquidation = max(
                    leg.average_price
                    * (
                        ONE
                        - ONE / self.leverage
                        + self.planner_config.maintenance_margin_rate
                    ),
                    Decimal("0.00000001"),
                )
            else:
                liquidation = leg.average_price * (
                    ONE + ONE / self.leverage - self.planner_config.maintenance_margin_rate
                )
            positions.append(
                PositionRiskLeg(
                    account_id=self.account_id,
                    symbol=self.symbol,
                    position_side=side,
                    quantity=leg.quantity,
                    mark_price=mark,
                    leverage=self.leverage,
                    maintenance_margin=maintenance,
                    liquidation_price=liquidation,
                )
            )
        pending: list[PendingOrderRisk] = []
        for order in self._execution().core.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            include_terminal=False,
        ):
            price = order.intent.limit_price or mark
            pending.append(
                PendingOrderRisk(
                    account_id=self.account_id,
                    symbol=self.symbol,
                    position_side=RiskPositionSide(order.intent.position_side.value),
                    action=RiskAction(order.intent.action.value),
                    remaining_quantity=order.approved_quantity - order.lifecycle.filled_quantity,
                    reference_price=price,
                    leverage=self.leverage,
                    maintenance_margin_rate=self.planner_config.maintenance_margin_rate,
                )
            )
        return build_risk_portfolio(
            account_id=self.account_id,
            equity=max(wallet.equity, Decimal("0.00000001")),
            wallet_balance=max(wallet.balance, ZERO),
            available_balance=wallet.available_balance,
            positions=positions,
            pending_orders=pending,
            strict_completeness=True,
        )

    @staticmethod
    def _encode_leg_state(state: StrategyLegState) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(StrategyLegState):
            value = getattr(state, item.name)
            if isinstance(value, Decimal):
                payload[item.name] = str(value)
            elif isinstance(value, datetime):
                payload[item.name] = value.isoformat()
            elif hasattr(value, "value"):
                payload[item.name] = value.value
            else:
                payload[item.name] = value
        return payload

    @staticmethod
    def _decode_leg_state(payload: object, side: PositionSide) -> StrategyLegState:
        if not isinstance(payload, Mapping):
            return StrategyLegState(side)
        values = dict(payload)
        values["side"] = side
        for name in (
            "trailing_extreme",
            "trailing_trigger_price",
            "unstuck_daily_loss",
            "unstuck_weekly_loss",
        ):
            if values.get(name) is not None:
                values[name] = Decimal(str(values[name]))
        for name in (
            "last_entry_at",
            "last_reduce_at",
            "trailing_started_at",
            "trailing_confirmed_at",
            "trailing_cooldown_until",
            "last_unstuck_at",
        ):
            if values.get(name):
                values[name] = datetime.fromisoformat(str(values[name]))
        allowed = {item.name for item in fields(StrategyLegState)}
        return StrategyLegState(**{key: value for key, value in values.items() if key in allowed})

    @staticmethod
    def _encode_market(market: MarketSnapshot | None) -> dict[str, object] | None:
        if market is None:
            return None
        return {
            "symbol": market.symbol,
            "timestamp": market.timestamp.isoformat(),
            "bid": str(market.bid),
            "ask": str(market.ask),
            "mark": str(market.mark),
            "tick_size": str(market.tick_size),
            "qty_step": str(market.qty_step),
            "min_qty": str(market.min_qty),
            "min_notional": str(market.min_notional),
        }

    @staticmethod
    def _decode_market(payload: object) -> MarketSnapshot | None:
        if not isinstance(payload, Mapping):
            return None
        return MarketSnapshot(
            symbol=str(payload["symbol"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            bid=Decimal(str(payload["bid"])),
            ask=Decimal(str(payload["ask"])),
            mark=Decimal(str(payload["mark"])),
            tick_size=Decimal(str(payload.get("tick_size", "0.01"))),
            qty_step=Decimal(str(payload.get("qty_step", "0.001"))),
            min_qty=Decimal(str(payload.get("min_qty", "0"))),
            min_notional=Decimal(str(payload.get("min_notional", "0"))),
        )

    @staticmethod
    def _encode_bar(bar: BarEvent | None) -> dict[str, object] | None:
        if bar is None:
            return None
        return {
            "timestamp": bar.timestamp.isoformat(),
            "symbol": bar.symbol,
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": None if bar.volume is None else str(bar.volume),
            "fingerprint": bar_fingerprint(bar),
        }

    @staticmethod
    def _decode_bar(payload: object) -> BarEvent | None:
        if not isinstance(payload, Mapping):
            return None
        bar = BarEvent(
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            symbol=str(payload["symbol"]),
            open=Decimal(str(payload["open"])),
            high=Decimal(str(payload["high"])),
            low=Decimal(str(payload["low"])),
            close=Decimal(str(payload["close"])),
            volume=(None if payload.get("volume") is None else Decimal(str(payload["volume"]))),
        )
        expected = payload.get("fingerprint")
        if expected is not None and str(expected) != bar_fingerprint(bar):
            raise ValueError("paper checkpoint bar fingerprint is invalid")
        return bar

    @staticmethod
    def _json_compatible(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(key): IntegratedPaperHedgeApplication._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list, set, frozenset)):
            return [
                IntegratedPaperHedgeApplication._json_compatible(item)
                for item in value
            ]
        raise TypeError(f"paper state cannot serialize {type(value).__name__}")

    @staticmethod
    def _encode_planner_intent(intent: PlannerOrderIntent | None) -> dict[str, object] | None:
        if intent is None:
            return None
        return {
            "intent_id": intent.intent_id,
            "symbol": intent.symbol,
            "position_side": intent.position_side.value,
            "order_side": intent.order_side.value,
            "action": intent.action.value,
            "bucket": intent.bucket.value,
            "quantity": str(intent.quantity),
            "price": str(intent.price),
            "reduce_only": intent.reduce_only,
            "order_type": intent.order_type.value,
            "time_in_force": intent.time_in_force.value,
            "layer": intent.layer,
            "reason": intent.reason,
            "tactical_lot_id": intent.tactical_lot_id,
        }

    @staticmethod
    def _decode_planner_intent(payload: object) -> PlannerOrderIntent | None:
        if not isinstance(payload, Mapping):
            return None
        return PlannerOrderIntent(
            intent_id=str(payload["intent_id"]),
            symbol=str(payload["symbol"]),
            position_side=PositionSide(str(payload["position_side"])),
            order_side=OrderSide(str(payload["order_side"])),
            action=IntentAction(str(payload["action"])),
            bucket=PositionBucket(str(payload["bucket"])),
            quantity=Decimal(str(payload["quantity"])),
            price=Decimal(str(payload["price"])),
            reduce_only=bool(payload["reduce_only"]),
            order_type=PlannerOrderType(str(payload.get("order_type", "LIMIT"))),
            time_in_force=TimeInForce(str(payload.get("time_in_force", "GTC"))),
            layer=int(payload.get("layer", 0)),
            reason=str(payload.get("reason", "")),
            tactical_lot_id=(
                None
                if payload.get("tactical_lot_id") is None
                else str(payload.get("tactical_lot_id"))
            ),
        )

    def _encode_active_execution_orders(self) -> list[dict[str, object]]:
        execution = self._execution()
        snapshots = {
            item.client_order_id: item for item in execution.exchange.list_orders()
        }
        rows: list[dict[str, object]] = []
        for order in execution.store.list_orders():
            if order.lifecycle.status not in {
                OrderState.ACKNOWLEDGED,
                OrderState.PARTIAL,
                OrderState.UNKNOWN,
            }:
                continue
            intent = order.intent
            snapshot = snapshots.get(order.client_order_id)
            rows.append(
                {
                    "client_order_id": order.client_order_id,
                    "approved_quantity": str(order.approved_quantity),
                    "created_at": order.created_at.isoformat(),
                    "intent": {
                        "account_id": intent.account_id,
                        "symbol": intent.symbol,
                        "position_side": intent.position_side.value,
                        "action": intent.action.value,
                        "quantity": str(intent.quantity),
                        "idempotency_key": intent.idempotency_key,
                        "order_type": intent.order_type.value,
                        "limit_price": (
                            None if intent.limit_price is None else str(intent.limit_price)
                        ),
                        "reduce_only": intent.reduce_only,
                        "intent_id": str(intent.intent_id),
                        "action_group_id": (
                            None
                            if intent.action_group_id is None
                            else str(intent.action_group_id)
                        ),
                        "metadata": self._json_compatible(intent.metadata),
                    },
                    "lifecycle": {
                        "status": order.lifecycle.status.value,
                        "filled_quantity": str(order.lifecycle.filled_quantity),
                        "average_price": (
                            None
                            if order.lifecycle.average_price is None
                            else str(order.lifecycle.average_price)
                        ),
                        "exchange_order_id": order.lifecycle.exchange_order_id,
                        "version": order.lifecycle.version,
                        "updated_at": order.lifecycle.updated_at.isoformat(),
                        "reason": order.lifecycle.reason,
                    },
                    "external": (
                        None
                        if snapshot is None
                        else {
                            "status": snapshot.status.value,
                            "filled_quantity": str(snapshot.filled_quantity),
                            "average_price": (
                                None
                                if snapshot.average_price is None
                                else str(snapshot.average_price)
                            ),
                            "exchange_order_id": snapshot.exchange_order_id,
                            "exchange_trade_id": snapshot.exchange_trade_id,
                            "last_fill_fee": str(snapshot.last_fill_fee),
                            "fee_currency": snapshot.fee_currency,
                            "reason": snapshot.reason,
                            "observed_at": snapshot.observed_at.isoformat(),
                        }
                    ),
                    "planner_intent": self._encode_planner_intent(
                        self._simulation_intents.get(order.client_order_id)
                    ),
                }
            )
        return rows

    def _restore_active_execution_orders(self, payload: object) -> None:
        if payload is None:
            return
        if not isinstance(payload, list):
            raise ValueError("paper active_orders must be a list")
        execution = self._execution()
        for raw in payload:
            if not isinstance(raw, Mapping):
                raise ValueError("paper active order row must be a mapping")
            intent_payload = raw.get("intent")
            lifecycle_payload = raw.get("lifecycle")
            if not isinstance(intent_payload, Mapping) or not isinstance(
                lifecycle_payload, Mapping
            ):
                raise ValueError("paper active order is missing intent/lifecycle")
            intent = ExecutionOrderIntent(
                account_id=str(intent_payload["account_id"]),
                symbol=str(intent_payload["symbol"]),
                position_side=ExecutionSide(str(intent_payload["position_side"])),
                action=ExecutionAction(str(intent_payload["action"])),
                quantity=Decimal(str(intent_payload["quantity"])),
                idempotency_key=str(intent_payload["idempotency_key"]),
                order_type=ExecutionOrderType(str(intent_payload.get("order_type", "LIMIT"))),
                limit_price=(
                    None
                    if intent_payload.get("limit_price") is None
                    else Decimal(str(intent_payload["limit_price"]))
                ),
                reduce_only=bool(intent_payload.get("reduce_only", False)),
                intent_id=UUID(str(intent_payload["intent_id"])),
                action_group_id=(
                    None
                    if intent_payload.get("action_group_id") is None
                    else UUID(str(intent_payload["action_group_id"]))
                ),
                metadata=(
                    dict(intent_payload.get("metadata", {}))
                    if isinstance(intent_payload.get("metadata", {}), Mapping)
                    else {}
                ),
            )
            lifecycle = OrderLifecycle(
                status=OrderState(str(lifecycle_payload["status"])),
                filled_quantity=Decimal(str(lifecycle_payload.get("filled_quantity", "0"))),
                average_price=(
                    None
                    if lifecycle_payload.get("average_price") is None
                    else Decimal(str(lifecycle_payload["average_price"]))
                ),
                exchange_order_id=(
                    None
                    if lifecycle_payload.get("exchange_order_id") is None
                    else str(lifecycle_payload["exchange_order_id"])
                ),
                version=int(lifecycle_payload.get("version", 0)),
                updated_at=datetime.fromisoformat(str(lifecycle_payload["updated_at"])),
                reason=(
                    None
                    if lifecycle_payload.get("reason") is None
                    else str(lifecycle_payload["reason"])
                ),
            )
            client_order_id = str(raw["client_order_id"])
            created_at = datetime.fromisoformat(str(raw["created_at"]))
            order = ExecutionOrder(
                intent=intent,
                client_order_id=client_order_id,
                approved_quantity=Decimal(str(raw["approved_quantity"])),
                lifecycle=lifecycle,
                created_at=created_at,
            )
            approved = ApprovedOrderIntent(
                intent=intent,
                approved_quantity=order.approved_quantity,
                client_order_id=client_order_id,
                approved_at=created_at,
                risk_reason_codes=("RECOVERED_DURABLE_PAPER",),
            )
            external_payload = raw.get("external")
            if isinstance(external_payload, Mapping):
                snapshot = ExternalOrderSnapshot(
                    client_order_id=client_order_id,
                    status=OrderState(str(external_payload["status"])),
                    filled_quantity=Decimal(
                        str(external_payload.get("filled_quantity", "0"))
                    ),
                    average_price=(
                        None
                        if external_payload.get("average_price") is None
                        else Decimal(str(external_payload["average_price"]))
                    ),
                    exchange_order_id=(
                        None
                        if external_payload.get("exchange_order_id") is None
                        else str(external_payload["exchange_order_id"])
                    ),
                    exchange_trade_id=(
                        None
                        if external_payload.get("exchange_trade_id") is None
                        else str(external_payload["exchange_trade_id"])
                    ),
                    last_fill_fee=Decimal(
                        str(external_payload.get("last_fill_fee", "0"))
                    ),
                    fee_currency=(
                        None
                        if external_payload.get("fee_currency") is None
                        else str(external_payload["fee_currency"])
                    ),
                    reason=(
                        None
                        if external_payload.get("reason") is None
                        else str(external_payload["reason"])
                    ),
                    observed_at=datetime.fromisoformat(
                        str(external_payload["observed_at"])
                    ),
                )
            else:
                snapshot = ExternalOrderSnapshot(
                    client_order_id=client_order_id,
                    status=order.lifecycle.status,
                    filled_quantity=order.lifecycle.filled_quantity,
                    average_price=order.lifecycle.average_price,
                    exchange_order_id=order.lifecycle.exchange_order_id,
                    reason=order.lifecycle.reason,
                    observed_at=order.lifecycle.updated_at,
                )
            execution.store.put(order)
            execution.exchange.restore_order(approved, snapshot)
            reservation = execution.idempotency.reserve(intent.idempotency_key)
            if reservation.value is None:
                execution.idempotency.complete(
                    intent.idempotency_key,
                    ExecutionResult(order=order, message="RECOVERED_DURABLE_PAPER"),
                )
            execution.ledger.record(
                order=order,
                event_type="ORDER_RECOVERED",
                payload={
                    "account_id": self.account_id,
                    "symbol": self.symbol,
                    "client_order_id": client_order_id,
                    "status": order.lifecycle.status.value,
                },
            )
            planner_intent = self._decode_planner_intent(raw.get("planner_intent"))
            if planner_intent is not None:
                self._simulation_intents[client_order_id] = planner_intent
                self._planner_order_to_client[planner_intent.intent_id] = client_order_id

    @staticmethod
    def _planner_intent_from_execution(order: ExecutionOrder) -> PlannerOrderIntent | None:
        metadata = order.intent.metadata
        planner_id = metadata.get("planner_intent_id")
        if planner_id is None or order.intent.limit_price is None:
            return None
        side = PositionSide(order.intent.position_side.value)
        action = IntentAction(order.intent.action.value)
        increases = action in {IntentAction.OPEN, IntentAction.INCREASE}
        order_side = (
            OrderSide.BUY
            if (side is PositionSide.LONG) == increases
            else OrderSide.SELL
        )
        try:
            bucket = PositionBucket(str(metadata.get("bucket", "TACTICAL")))
            time_in_force = TimeInForce(str(metadata.get("time_in_force", "GTC")))
        except ValueError:
            return None
        return PlannerOrderIntent(
            intent_id=str(planner_id),
            symbol=order.intent.symbol,
            position_side=side,
            order_side=order_side,
            action=action,
            bucket=bucket,
            quantity=order.approved_quantity,
            price=order.intent.limit_price,
            reduce_only=order.intent.reduce_only,
            order_type=PlannerOrderType(order.intent.order_type.value),
            time_in_force=time_in_force,
            layer=int(metadata.get("layer", 0)),
            reason=str(metadata.get("reason", "recovered from SQL execution state")),
            tactical_lot_id=(
                None
                if metadata.get("tactical_lot_id") is None
                else str(metadata.get("tactical_lot_id"))
            ),
        )

    def _restore_authoritative_execution_orders(self) -> bool:
        execution = self._execution()
        restored_any = False
        for order in execution.store.list_orders():
            if (
                order.intent.account_id != self.account_id
                or raw_symbol(order.intent.symbol) != self.execution_symbol
                or str(order.intent.metadata.get("exchange", "")).lower() != "paper"
            ):
                continue
            if order.lifecycle.status not in {
                OrderState.ACKNOWLEDGED,
                OrderState.PARTIAL,
                OrderState.UNKNOWN,
            }:
                continue
            approved = ApprovedOrderIntent(
                intent=order.intent,
                approved_quantity=order.approved_quantity,
                client_order_id=order.client_order_id,
                approved_at=order.created_at,
                risk_reason_codes=("RECOVERED_SQL_EXECUTION_STATE",),
            )
            snapshot = execution.exchange.query_order(
                client_order_id=order.client_order_id
            )
            if snapshot is None:
                snapshot = ExternalOrderSnapshot(
                    client_order_id=order.client_order_id,
                    status=order.lifecycle.status,
                    filled_quantity=order.lifecycle.filled_quantity,
                    average_price=order.lifecycle.average_price,
                    exchange_order_id=order.lifecycle.exchange_order_id,
                    reason=order.lifecycle.reason,
                    observed_at=order.lifecycle.updated_at,
                )
                execution.exchange.restore_order(approved, snapshot)
            else:
                if (
                    snapshot.status is not order.lifecycle.status
                    or snapshot.filled_quantity != order.lifecycle.filled_quantity
                    or snapshot.average_price != order.lifecycle.average_price
                ):
                    raise RuntimeError(
                        "existing Paper exchange snapshot conflicts with authoritative order state"
                    )
            result = ExecutionResult(
                order=order,
                message="RECOVERED_SQL_EXECUTION_STATE",
            )
            recover_completed = getattr(
                execution.idempotency,
                "recover_completed",
                None,
            )
            if callable(recover_completed):
                recover_completed(order.intent.idempotency_key, result)
            else:
                reservation = execution.idempotency.reserve(order.intent.idempotency_key)
                if reservation.state is ReservationState.NEW:
                    execution.idempotency.complete(
                        order.intent.idempotency_key,
                        result,
                    )
                elif reservation.state is ReservationState.IN_FLIGHT:
                    raise RuntimeError(
                        "cannot recover SQL order while idempotency is still in flight"
                    )
            planner_intent = self._planner_intent_from_execution(order)
            if planner_intent is not None:
                self._simulation_intents[order.client_order_id] = planner_intent
                self._planner_order_to_client[planner_intent.intent_id] = (
                    order.client_order_id
                )
            restored_any = True
        return restored_any

    def _restore_authoritative_fills(self) -> bool:
        recover = getattr(self._execution_recovery, "recover_fills", None)
        if not callable(recover):
            return False
        fills = recover()
        if fills is None:
            return False
        execution = self._execution()
        execution.account.restore(())
        self._bucket = {
            PositionSide.LONG: _BucketState(),
            PositionSide.SHORT: _BucketState(),
        }
        remember = getattr(execution.exchange, "remember_fill_identity", None)
        for fill in fills:
            side = ExecutionSide(fill.position_side)
            action = ExecutionAction(fill.action)
            execution.account.apply_fill(
                trade_id=fill.trade_id,
                account_id=self.account_id,
                symbol=self.execution_symbol,
                position_side=side,
                action=action,
                quantity=fill.quantity,
                price=fill.price,
                fee_amount=fill.fee,
            )
            bucket = PositionBucket(fill.bucket)
            bucket_state = self._bucket[PositionSide(fill.position_side)]
            if action in {ExecutionAction.OPEN, ExecutionAction.INCREASE}:
                bucket_state.increase(bucket, fill.quantity, fill.price, fill.event_time)
            else:
                bucket_state.reduce(bucket, fill.quantity)
            if callable(remember):
                remember(
                    trade_id=fill.trade_id,
                    client_order_id=fill.client_order_id,
                    quantity=fill.quantity,
                    price=fill.price,
                    fee=fill.fee,
                    fee_currency=fill.fee_currency,
                )
            # A process may terminate after the immutable SQL fill commits but
            # before its fee account-event/outbox transaction. Reconcile the
            # deterministic fee event from the authoritative fill fact during
            # recovery without applying the cash effect twice.
            if fill.fee > ZERO:
                self._record_account_event(
                    fee_account_event(
                        fill_event_id=fill.trade_id,
                        timestamp=fill.event_time,
                        symbol=self.symbol,
                        amount=fill.fee,
                        position_side=PositionSide(fill.position_side),
                    )
                )
        return True

    def _restore_account_events(self) -> None:
        if not self.paper_config.account_events_enabled:
            return
        recover = getattr(self._account_event_sink, "recover", None)
        if not callable(recover):
            return
        recovered = recover()
        if recovered is None:
            return
        self._applied_account_event_ids.update(recovered.event_ids)
        self._funding_balance_delta = recovered.funding_balance_delta
        self._last_funding_event_time = recovered.last_funding_event_time

    def _restore_state(self) -> None:
        if self._state_loaded:
            return
        payload = self._state_store.load()
        self._state_loaded = True
        if payload is None:
            self._restore_account_events()
            self._restore_authoritative_execution_orders()
            self._restore_authoritative_fills()
            return
        if payload.get("account_id") != self.account_id or payload.get("symbol") != self.symbol:
            raise ValueError("paper state identity does not match configured account/symbol")
        execution = self._execution()
        execution.account.restore(payload.get("account_legs", ()))
        buckets = payload.get("buckets", {})
        if not isinstance(buckets, Mapping):
            raise ValueError("paper bucket state must be a mapping")
        for side in (PositionSide.LONG, PositionSide.SHORT):
            row = buckets.get(side.value, {})
            if not isinstance(row, Mapping):
                raise ValueError("paper bucket row must be a mapping")
            self._bucket[side] = _BucketState(
                core_quantity=Decimal(str(row.get("core_quantity", "0"))),
                core_average=Decimal(str(row.get("core_average", "0"))),
                core_opened_at=(
                    None
                    if row.get("core_opened_at") in (None, "")
                    else datetime.fromisoformat(str(row["core_opened_at"]))
                ),
                tactical_quantity=Decimal(str(row.get("tactical_quantity", "0"))),
                tactical_average=Decimal(str(row.get("tactical_average", "0"))),
                tactical_opened_at=(
                    None
                    if row.get("tactical_opened_at") in (None, "")
                    else datetime.fromisoformat(str(row["tactical_opened_at"]))
                ),
            )
        self.long_state = self._decode_leg_state(payload.get("long_state"), PositionSide.LONG)
        self.short_state = self._decode_leg_state(payload.get("short_state"), PositionSide.SHORT)
        self._last_market = self._decode_market(payload.get("last_market"))
        self._last_bar = self._decode_bar(payload.get("last_bar"))
        if self._last_bar is not None:
            if self._last_market is None or self._last_bar.timestamp != self._last_market.timestamp:
                raise ValueError("paper checkpoint market/bar cursor mismatch")
            if self._last_bar.symbol != self._last_market.symbol or self._last_bar.close != self._last_market.mark:
                raise ValueError("paper checkpoint market/bar values mismatch")
        self._funding_balance_delta = Decimal(str(payload.get("funding_balance_delta", "0")))
        raw_funding_time = payload.get("last_funding_event_time")
        self._last_funding_event_time = (
            None
            if raw_funding_time in (None, "")
            else datetime.fromisoformat(str(raw_funding_time))
        )
        if (
            self._last_funding_event_time is not None
            and self._last_funding_event_time.tzinfo is None
        ):
            self._last_funding_event_time = self._last_funding_event_time.replace(tzinfo=UTC)
        event_ids = payload.get("applied_account_event_ids", ())
        if isinstance(event_ids, (str, bytes)):
            raise ValueError("paper account event ids must be a sequence")
        self._applied_account_event_ids = {str(item) for item in event_ids}
        self._restore_account_events()
        self._planner_order_to_client.clear()
        self._simulation_intents.clear()
        if not self._restore_authoritative_execution_orders():
            self._restore_active_execution_orders(payload.get("active_orders"))
        self._restore_authoritative_fills()

    def _persist_state(
        self,
        *,
        committed_market: MarketSnapshot | None = None,
        committed_bar: BarEvent | None = None,
    ) -> None:
        execution = self._execution()
        payload = {
            "account_id": self.account_id,
            "symbol": self.symbol,
            "account_legs": list(execution.account.snapshot()),
            "buckets": {
                side.value: {
                    "core_quantity": str(state.core_quantity),
                    "core_average": str(state.core_average),
                    "core_opened_at": (
                        None if state.core_opened_at is None else state.core_opened_at.isoformat()
                    ),
                    "tactical_quantity": str(state.tactical_quantity),
                    "tactical_average": str(state.tactical_average),
                    "tactical_opened_at": (
                        None
                        if state.tactical_opened_at is None
                        else state.tactical_opened_at.isoformat()
                    ),
                }
                for side, state in self._bucket.items()
            },
            "long_state": self._encode_leg_state(self.long_state),
            "short_state": self._encode_leg_state(self.short_state),
            "last_market": self._encode_market(committed_market or self._last_market),
            "last_bar": self._encode_bar(committed_bar or self._last_bar),
            "funding_balance_delta": str(self._funding_balance_delta),
            "last_funding_event_time": (
                None
                if self._last_funding_event_time is None
                else self._last_funding_event_time.isoformat()
            ),
            "applied_account_event_ids": sorted(self._applied_account_event_ids),
            "execution_state_authority": (
                "checkpoint"
                if isinstance(execution.store, InMemoryExecutionStore)
                else "sql"
            ),
            "active_orders": (
                self._encode_active_execution_orders()
                if isinstance(execution.store, InMemoryExecutionStore)
                else []
            ),
            "pending_orders_recovered": True,
        }
        try:
            self._state_store.save(payload)
        except Exception:
            # Business facts may already be committed to SQL, while planner
            # temporal state still relies on the checkpoint. Continuing in the
            # same process would mix pre- and post-failure memory. Force a clean
            # restart so recovery can converge from durable facts.
            self._requires_restart = True
            raise

    def _record_account_event(self, event: AccountEvent) -> bool:
        if event.event_id in self._applied_account_event_ids:
            return False
        created = True
        if self.paper_config.account_events_enabled:
            created = self._account_event_sink.record(event)
        self._applied_account_event_ids.add(event.event_id)
        return created

    def _apply_funding_events(
        self,
        events: tuple[FundingEvent, ...],
    ) -> tuple[AccountEvent, ...]:
        applied: list[AccountEvent] = []
        for funding in sorted(events, key=lambda item: item.timestamp):
            long_leg = self._leg(PositionSide.LONG)
            short_leg = self._leg(PositionSide.SHORT)
            long_amount = -(long_leg.quantity * funding.mark_price * funding.rate)
            short_amount = short_leg.quantity * funding.mark_price * funding.rate
            total = long_amount + short_amount
            event = funding_account_event(funding=funding, amount=total)
            if self._record_account_event(event):
                self._funding_balance_delta += total
                if (
                    self._last_funding_event_time is None
                    or funding.timestamp > self._last_funding_event_time
                ):
                    self._last_funding_event_time = funding.timestamp
                applied.append(event)
        return tuple(applied)

    def _update_market_rules(
        self,
        market: MarketSnapshot,
        *,
        maker_fee_rate: Decimal | None = None,
        taker_fee_rate: Decimal | None = None,
    ) -> None:
        execution = self._execution()
        setter = getattr(execution.market_rules, "set_rules", None)
        rules = ExecutionMarketRules(
            quantity_step=market.qty_step,
            price_tick=market.tick_size,
            minimum_quantity=max(market.min_qty, Decimal("0.00000001")),
            minimum_notional=max(market.min_notional, Decimal("0.00000001")),
        )
        if callable(setter):
            for side in (ContractPositionSide.LONG, ContractPositionSide.SHORT):
                setter(
                    ContractPositionKey(
                        exchange="paper",
                        account_id=self.account_id,
                        symbol=self.symbol,
                        position_side=side,
                    ),
                    rules,
                )

        maker = self.matcher.config.maker_fee_rate if maker_fee_rate is None else maker_fee_rate
        taker = self.matcher.config.taker_fee_rate if taker_fee_rate is None else taker_fee_rate
        for field, value in (("maker_fee_rate", maker), ("taker_fee_rate", taker)):
            if not value.is_finite() or value < ZERO or value > ONE:
                raise ValueError(f"{field} must be finite and within [0, 1]")
        # DataProvider OHLCV, exchange precision and current account/market fee
        # rates become one immutable matching snapshot for this cycle.
        self.matcher = ConservativeMatcher(
            replace(
                self.matcher.config,
                fee_rate=None,
                maker_fee_rate=maker,
                taker_fee_rate=taker,
                price_tick=market.tick_size,
                qty_step=market.qty_step,
                min_fill_qty=market.min_qty,
                min_fill_notional=market.min_notional,
            )
        )
        self._paper_fee_rate = taker

    def _planner_intent_for_execution_order(
        self,
        order: ExecutionOrder,
    ) -> PlannerOrderIntent | None:
        existing = self._simulation_intents.get(order.client_order_id)
        if existing is not None:
            return existing
        price = order.intent.limit_price
        if price is None:
            reference = order.intent.metadata.get("reference_price")
            if reference is None:
                return None
            price = Decimal(str(reference))
        side = PositionSide(order.intent.position_side.value)
        reduce_only = order.intent.reduces_risk
        order_side = (
            OrderSide.SELL
            if side is PositionSide.LONG and reduce_only
            else OrderSide.BUY
            if side is PositionSide.LONG
            else OrderSide.BUY
            if reduce_only
            else OrderSide.SELL
        )
        try:
            bucket = PositionBucket(
                str(order.intent.metadata.get("bucket", "TACTICAL")).upper()
            )
        except ValueError:
            bucket = PositionBucket.TACTICAL
        planner_intent = PlannerOrderIntent(
            intent_id=str(
                order.intent.metadata.get("planner_intent_id", order.intent.intent_id)
            ),
            symbol=self.symbol,
            position_side=side,
            order_side=order_side,
            action=IntentAction(order.intent.action.value),
            bucket=bucket,
            quantity=order.approved_quantity - order.lifecycle.filled_quantity,
            price=price,
            reduce_only=reduce_only,
            order_type=PlannerOrderType(order.intent.order_type.value),
            time_in_force=TimeInForce(
                str(order.intent.metadata.get("time_in_force", "GTC")).upper()
            ),
            layer=int(order.intent.metadata.get("layer", 0)),
            reason=str(
                order.intent.metadata.get(
                    "control_action",
                    order.intent.metadata.get("reason", "managed_execution_order"),
                )
            )[:256],
            tactical_lot_id=(
                None
                if order.intent.metadata.get("tactical_lot_id") is None
                else str(order.intent.metadata.get("tactical_lot_id"))
            ),
        )
        self._simulation_intents[order.client_order_id] = planner_intent
        return planner_intent

    def _matcher_wallet(self, market: MarketSnapshot) -> CrossWallet:
        execution = self._execution()
        wallet = CrossWallet(
            initial_balance=self.initial_balance,
            leverage=self.leverage,
            fee_rate=self._paper_fee_rate,
            maintenance_margin_rate=self.planner_config.maintenance_margin_rate,
            liquidation_fee_rate=self.planner_config.liquidation_fee_rate,
            liquidation_buffer_warning_ratio=(
                self.planner_config.liquidation_buffer_warning_ratio
            ),
        )
        now = market.timestamp
        for side in (PositionSide.LONG, PositionSide.SHORT):
            state = self._bucket[side]
            leg = wallet.leg(side)
            if state.core_quantity > 0:
                leg.increase(
                    state.core_quantity,
                    state.core_average,
                    PositionBucket.CORE,
                    tactical_lot_id=None,
                    opened_at=now,
                    layer=0,
                    fee=ZERO,
                )
            if state.tactical_quantity > 0:
                leg.increase(
                    state.tactical_quantity,
                    state.tactical_average,
                    PositionBucket.TACTICAL,
                    tactical_lot_id=f"paper-recovered-{side.value.lower()}",
                    opened_at=now,
                    layer=0,
                    fee=ZERO,
                )
            fake = self._fake_leg(side)
            leg.realized_pnl = fake.realized_pnl
            leg.tactical_realized_pnl = fake.realized_pnl
        fees = self._fake_leg(PositionSide.LONG).fees + self._fake_leg(PositionSide.SHORT).fees
        realized = (
            self._fake_leg(PositionSide.LONG).realized_pnl
            + self._fake_leg(PositionSide.SHORT).realized_pnl
        )
        wallet.balance = self.initial_balance + realized - fees + self._funding_balance_delta
        for order in execution.core.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            include_terminal=False,
        ):
            planner_intent = self._planner_intent_for_execution_order(order)
            if planner_intent is None:
                continue
            remaining = order.approved_quantity - order.lifecycle.filled_quantity
            if remaining <= 0:
                continue
            wallet.accept_order(
                order.client_order_id,
                replace(planner_intent, quantity=remaining),
                accepted_at=order.created_at,
            )
        return wallet

    def _match_active_orders(
        self,
        market: MarketSnapshot,
        bar: BarEvent,
    ) -> tuple[list[ExecutionResult], list[ExecutionResult], list[AccountEvent]]:
        execution = self._execution()
        if bar.symbol != market.symbol or bar.timestamp != market.timestamp:
            raise ValueError("Paper BarEvent must match the planning MarketSnapshot")
        outcome = self.matcher.match_outcome(bar, self._matcher_wallet(market))
        fills: list[ExecutionResult] = []
        expirations: list[ExecutionResult] = []
        account_events: list[AccountEvent] = []
        for fill in outcome.fills:
            try:
                snapshot = execution.exchange.fill_order(
                    fill.order_id,
                    quantity=fill.quantity,
                    price=fill.price,
                    exchange_trade_id=fill.event_id,
                    fee=fill.fee,
                )
                result = execution.engine.apply_exchange_event(snapshot)
            except (KeyError, ValueError):
                continue
            fills.append(result)
            planner_for_callback = self._simulation_intents.get(fill.order_id)
            if planner_for_callback is not None:
                self._notify_fill_observers(
                    planner_for_callback, fill.price, fill.quantity, fill.timestamp
                )
            fee_event = fee_account_event(
                fill_event_id=fill.event_id,
                timestamp=fill.timestamp,
                symbol=fill.symbol,
                amount=fill.fee,
                position_side=fill.position_side,
            )
            if self._record_account_event(fee_event):
                account_events.append(fee_event)
            state = self._bucket[fill.position_side]
            if fill.action in {IntentAction.OPEN, IntentAction.INCREASE}:
                state.increase(fill.bucket, fill.quantity, fill.price, fill.timestamp)
            else:
                state.reduce(fill.bucket, fill.quantity)
            if result.order.lifecycle.filled_quantity >= result.order.approved_quantity:
                self._simulation_intents.pop(fill.order_id, None)
        for client_id in outcome.expired_order_ids:
            try:
                expirations.append(execution.engine.cancel(client_id))
            except Exception:
                continue
            self._simulation_intents.pop(client_id, None)
        return fills, expirations, account_events

    def run_market_cycle(
        self,
        market: MarketSnapshot,
        *,
        bar: BarEvent | None = None,
        funding_events: tuple[FundingEvent, ...] = (),
        long_signal: Decimal | None = None,
        short_signal: Decimal | None = None,
        target_net_quantity: Decimal | None = None,
        target_net_ratio: Decimal | None = None,
        confidence: Decimal = ONE,
        risk_scale: Decimal = ONE,
        long_exposure_scale: Decimal = ONE,
        short_exposure_scale: Decimal = ONE,
        allow_new_risk: bool = True,
        regime: str = "UNSPECIFIED",
        strategy_reason: str = "",
        model_version: str = "strategy",
        maker_fee_rate: Decimal | None = None,
        taker_fee_rate: Decimal | None = None,
    ) -> PaperCycleResult:
        with self._lock:
            if self._requires_restart:
                raise RuntimeError(
                    "Paper runtime requires restart after a failed durable checkpoint"
                )
            if bar is None:
                if self.paper_config.ohlcv_source is not PaperOhlcvSource.TICKER_COMPAT:
                    raise ValueError(
                        "production Paper cycles require the closed DataProvider BarEvent"
                    )
                previous = self._last_market
                open_price = market.mark if previous is None else previous.mark
                bar = BarEvent(
                    timestamp=market.timestamp,
                    symbol=market.symbol,
                    open=open_price,
                    high=max(open_price, market.bid, market.ask, market.mark),
                    low=min(open_price, market.bid, market.ask, market.mark),
                    close=market.mark,
                    volume=self._bar_volume,
                )
            if bar.symbol != market.symbol or bar.timestamp != market.timestamp:
                raise ValueError("Paper market and OHLCV bar identity must match")
            if bar.close != market.mark:
                raise ValueError("Paper planning mark must equal the analyzed candle close")
            previous_market = self._last_market
            if previous_market is not None and market.timestamp <= previous_market.timestamp:
                relation = "duplicate" if market.timestamp == previous_market.timestamp else "out-of-order"
                raise ValueError(
                    f"Paper refused {relation} candle {market.timestamp.isoformat()} "
                    f"after {previous_market.timestamp.isoformat()}"
                )

            try:
                self._cycle_market = market
                self._update_market_rules(
                    market,
                    maker_fee_rate=maker_fee_rate,
                    taker_fee_rate=taker_fee_rate,
                )
                account_events: list[AccountEvent] = list(
                    self._apply_funding_events(funding_events)
                )
                # Match only orders accepted before this candle.  New orders are
                # created at the analyzed candle close and become eligible on the
                # next bar, eliminating same-candle look-ahead in live Paper.
                fills: list[ExecutionResult] = []
                cancellations: list[ExecutionResult] = []
                if self.auto_fill and self.fill_model != "instant":
                    matched, expired, matched_events = self._match_active_orders(market, bar)
                    fills.extend(matched)
                    cancellations.extend(expired)
                    account_events.extend(matched_events)

                wallet_before = self.wallet(market)
                directive = StrategyDirective(
                    long_score=self.default_long_signal if long_signal is None else long_signal,
                    short_score=self.default_short_signal if short_signal is None else short_signal,
                    target_net_quantity=(None if target_net_ratio is not None else target_net_quantity),
                    target_net_ratio=target_net_ratio,
                    confidence=confidence,
                    risk_scale=risk_scale,
                    long_exposure_scale=long_exposure_scale,
                    short_exposure_scale=short_exposure_scale,
                    allow_new_risk=allow_new_risk,
                    regime=regime,
                    reason=strategy_reason,
                    model_version=model_version,
                )
                effective_config = planner_config_for_directive(self.planner_config, directive)
                effective_target = target_net_quantity_for_directive(
                    directive=directive,
                    base=self.planner_config,
                    equity=wallet_before.equity,
                    mark_price=market.mark,
                )
                context = PlanningContext(
                    market=market,
                    wallet=wallet_before,
                    config=effective_config,
                    long_state=self.long_state,
                    short_state=self.short_state,
                    long_signal=directive.long_score,
                    short_signal=directive.short_score,
                    target_net_quantity=effective_target,
                )
                planning = self.planner.plan(context)
                planning, _native_exit_diagnostics = self._exit_overlay.apply(
                    planning, app=self, market=market
                )
                planning = self._transform_planning_intents(planning)
                planning, _blocked_new_risk = apply_new_risk_gate(
                    planning,
                    enabled=self.new_risk_enabled() and directive.allow_new_risk,
                    current_long_state=self.long_state,
                    current_short_state=self.short_state,
                )
                planning, native_admission_blocks = apply_planning_admission_gate(
                    planning,
                    evaluate=self._order_admission_policy.evaluate,
                    current_long_state=self.long_state,
                    current_short_state=self.short_state,
                )
                _blocked_new_risk += len(native_admission_blocks)
                for planner_order_id in planning.cancel_order_ids:
                    client_id = self._planner_order_to_client.pop(planner_order_id, None)
                    if client_id is None:
                        continue
                    try:
                        cancellations.append(self._execution().engine.cancel(client_id))
                    finally:
                        self._simulation_intents.pop(client_id, None)

                execution_intents = adapt_planner_intents(
                    planning.submit_orders,
                    account_id=self.account_id,
                    exchange="paper",
                    strategy_id="pure-hedge-planner",
                    cycle_id=market.timestamp.isoformat(),
                )
                executions: list[ExecutionResult] = []
                for planner_intent, execution_intent in zip(
                    planning.submit_orders,
                    execution_intents,
                    strict=True,
                ):
                    try:
                        result = self._execution().engine.submit(execution_intent)
                    except ExecutionBlockedError:
                        continue
                    executions.append(result)
                    client_id = result.order.client_order_id
                    self._planner_order_to_client[planner_intent.intent_id] = client_id
                    self._simulation_intents[client_id] = planner_intent

                if self.auto_fill:
                    if self.fill_model == "instant":
                        for result in executions:
                            fill_price = result.order.intent.limit_price or market.mark
                            snapshot = self._execution().exchange.fill_order(
                                result.order.client_order_id,
                                quantity=result.order.approved_quantity,
                                price=fill_price,
                            )
                            applied = self._execution().engine.apply_exchange_event(snapshot)
                            fills.append(applied)
                            trade_id = snapshot.exchange_trade_id or result.order.client_order_id
                            fee = result.order.approved_quantity * fill_price * self._paper_fee_rate
                            fee_event = fee_account_event(
                                fill_event_id=trade_id,
                                timestamp=bar.timestamp,
                                symbol=market.symbol,
                                amount=fee,
                                position_side=PositionSide(result.order.intent.position_side.value),
                            )
                            if self._record_account_event(fee_event):
                                account_events.append(fee_event)
                            planner_intent = self._simulation_intents.pop(
                                result.order.client_order_id,
                                None,
                            )
                            if planner_intent is not None:
                                self._notify_fill_observers(
                                    planner_intent,
                                    fill_price,
                                    result.order.approved_quantity,
                                    bar.timestamp,
                                )
                                state = self._bucket[planner_intent.position_side]
                                if planner_intent.action in {IntentAction.OPEN, IntentAction.INCREASE}:
                                    state.increase(
                                        planner_intent.bucket,
                                        result.order.approved_quantity,
                                        fill_price,
                                        bar.timestamp,
                                    )
                                else:
                                    state.reduce(
                                        planner_intent.bucket,
                                        result.order.approved_quantity,
                                    )

                self.long_state = planning.long_state
                self.short_state = planning.short_state
                result = PaperCycleResult(
                    planning=planning,
                    executions=tuple(executions),
                    fills=tuple(fills),
                    cancellations=tuple(cancellations),
                    wallet=self.wallet(market),
                    account_events=tuple(account_events),
                )
                # The durable candle cursor advances only after every business fact
                # and the auxiliary checkpoint have committed. A failed save leaves
                # the prior cursor intact so the same candle can be retried and
                # converged from SQL facts instead of being silently skipped.
                self._persist_state(committed_market=market, committed_bar=bar)
                self._last_market = market
                self._last_bar = bar
                wallet_after = result.wallet
                realized = wallet_after.long.realized_pnl + wallet_after.short.realized_pnl
                fees = self._fake_leg(PositionSide.LONG).fees + self._fake_leg(PositionSide.SHORT).fees
                cycle_telemetry = DryRunCycleTelemetry(
                        cycle_id=market.timestamp.isoformat(),
                        account_id=self.account_id,
                        symbol=self.symbol,
                        timestamp=market.timestamp,
                        mark_price=market.mark,
                        equity=wallet_after.equity,
                        available_balance=wallet_after.available_balance,
                        gross_notional=wallet_after.gross_notional(market.mark),
                        net_quantity=wallet_after.long.quantity-wallet_after.short.quantity,
                        target_net_quantity=planning.target_net_quantity,
                        net_gap_quantity=planning.net_gap_quantity,
                        long_quantity=wallet_after.long.quantity,
                        short_quantity=wallet_after.short.quantity,
                        long_target_quantity=planning.long_target_quantity,
                        short_target_quantity=planning.short_target_quantity,
                        long_average_price=wallet_after.long.average_price,
                        short_average_price=wallet_after.short.average_price,
                        unrealized_pnl=wallet_after.long.unrealized_pnl(market.mark)+wallet_after.short.unrealized_pnl(market.mark),
                        realized_pnl=realized,
                        funding_pnl=self._funding_balance_delta,
                        fees=fees,
                        ideal_order_count=len(planning.ideal_orders),
                        submit_order_count=len(planning.submit_orders),
                        cancel_order_count=len(planning.cancel_order_ids),
                        fill_count=len(fills),
                        active_order_count=len(wallet_after.active_orders),
                        risk_blocked=bool(_blocked_new_risk),
                        diagnostics=planning.diagnostics,
                        strategy=StrategyTelemetry(
                            long_score=directive.long_score,short_score=directive.short_score,
                            target_net_quantity=directive.target_net_quantity,target_net_ratio=directive.target_net_ratio,
                            confidence=directive.confidence,risk_scale=directive.risk_scale,
                            long_exposure_scale=directive.long_exposure_scale,short_exposure_scale=directive.short_exposure_scale,
                            allow_new_risk=directive.allow_new_risk,regime=directive.regime,reason=directive.reason,model_version=directive.model_version,
                        ),
                    )
                self.telemetry.append(cycle_telemetry)
                if self.operations is not None:
                    try:
                        self.operations.observe(
                            OperationsCycleInput(
                                timestamp=market.timestamp,
                                symbol=self.symbol,
                                timeframe_seconds=60,
                                mark_price=market.mark,
                                index_price=market.mark,
                                equity=wallet_after.equity,
                                initial_equity=self.initial_balance,
                                long_notional=wallet_after.long.quantity * market.mark,
                                short_notional=wallet_after.short.quantity * market.mark,
                                margin_used=(wallet_after.gross_notional(market.mark) / max(self.leverage, ONE)),
                                realized_pnl=realized,
                                unrealized_pnl=cycle_telemetry.unrealized_pnl,
                                funding_pnl=self._funding_balance_delta,
                                fees=fees,
                                slippage_cost=ZERO,
                                base_candles=self.operations.session.cycle_sequence + 1,
                                informative_candles={},
                                observed_at=datetime.now(UTC),
                                order_count=len(executions),
                                fill_count=len(fills),
                                active_order_count=len(wallet_after.active_orders),
                                reconciliation_fresh=(
                                    self._state_durable and not self._requires_restart
                                ),
                                api_healthy=self.dashboard_enabled,
                                dashboard_healthy=self.dashboard_enabled,
                            )
                        )
                        self.operations_error = None
                    except Exception as exc:
                        self.operations_error = f"{type(exc).__name__}: {exc}"[:512]
                self._cycle_market = None
                return result
            finally:
                self._cycle_market = None

    def publish_runtime(
        self,
        runtime: object,
        *,
        market_data_fresh: bool = True,
        funding_source_healthy: bool = True,
        market_source: str = "UNKNOWN",
    ) -> None:
        """Publish PAPER state without impersonating Binance health checks."""
        from freqtrade.hedge.position_book import PositionRecord

        market = self.last_market
        if market is None:
            return
        wallet = self.wallet(market)
        positions = []
        for side, leg in ((PositionSide.LONG, wallet.long), (PositionSide.SHORT, wallet.short)):
            if leg.quantity <= 0:
                continue
            positions.append(
                PositionRecord(
                    symbol=self.symbol,
                    position_side=side.value,
                    amount=leg.quantity,
                    entry_price=leg.average_price,
                    mark_price=market.mark,
                    unrealized_pnl=leg.unrealized_pnl(market.mark),
                    leverage=self.leverage,
                    collateral=leg.quantity * market.mark / self.leverage,
                    source="PAPER_EXECUTION",
                    exchange="paper",
                    account_id=self.account_id,
                )
            )
        risk = self.risk_portfolio().account
        runtime.publish(
            source=HedgeProjectionSource.PAPER,
            positions=tuple(positions),
            risk=risk,
            reconciliation_status="NOT_APPLICABLE",
            reconciliation_at=None,
            reconciliation_details=("source=paper-execution-ledger",),
            stream_state="NOT_APPLICABLE",
            stream_last_event_at=None,
            stream_reconnect_count=0,
            checks={
                "common.persistence_healthy": self._state_durable,
                "paper.market_data_fresh": market_data_fresh,
                "paper.funding_source_healthy": funding_source_healthy,
                "paper.account_events_durable": (
                    self._state_durable and self.paper_config.account_events_enabled
                )
                or not self.paper_config.account_events_enabled,
                "paper.simulation_engine_healthy": True,
                "paper.ledger_durable": self._state_durable,
                "paper.risk_snapshot_valid": risk.effective_risk_data_valid,
            },
            reasons=(
                ()
                if self._state_durable and risk.effective_risk_data_valid
                else tuple(
                    dict.fromkeys(
                        (
                            *(
                                ()
                                if self._state_durable
                                else ("PAPER_LEDGER_NOT_DURABLE",)
                            ),
                            *risk.risk_data_errors,
                        )
                    )
                )
            ),
            source_version=f"{market_source}:{market.timestamp.isoformat()}",
            source_event_time=market.timestamp,
            stale=False,
        )
