"""Integrated execution entrypoint that composes directions one, three, four and five."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Mapping, Protocol

from freqtrade.hedge.contracts.events import FillEvent, OutboxEvent
from freqtrade.hedge.contracts.ports import (
    AlwaysReadyGate,
    ClockPort,
    EventPublisherPort,
    ExecutionTransactionPort,
    InMemoryPositionLock,
    InMemorySingleWriter,
    MarketRulesPort,
    NullEventPublisher,
    NullExecutionTransaction,
    PositionLockPort,
    ReadinessGatePort,
    SingleWriterPort,
    StaticMarketRules,
    SystemClock,
)
from freqtrade.hedge.contracts.types import (
    IntentAction as ContractIntentAction,
    OrderSide,
    PositionKey,
    PositionSide as ContractPositionSide,
    expected_order_side,
)

from .service import (
    ExecutionBlockedError,
    ExecutionOrder,
    ExecutionResult,
    ExecutionService,
    ExternalOrderSnapshot,
    OrderIntent,
    OrderType,
)
from .state_machine import OrderState
from .unknown_resolver import UserStreamOrderCacheSinkPort


class UnknownOrderSupervisorPort(Protocol):
    """Minimal query-only UNKNOWN recovery coordinator owned by the runtime."""

    def register(self, client_order_id: str) -> object: ...

    def run_due(self) -> tuple[object, ...]: ...


class HedgeExecutionEngine:
    """The production-facing execution entrypoint.

    ``ExecutionService`` remains the deterministic order lifecycle core.  This engine adds
    readiness, single-writer, cross-process lock ports, market filters, transactional facts
    and event publication without depending on concrete database or Binance classes.
    """

    def __init__(
        self,
        core: ExecutionService,
        *,
        readiness: ReadinessGatePort | None = None,
        single_writer: SingleWriterPort | None = None,
        position_lock: PositionLockPort | None = None,
        market_rules: MarketRulesPort | None = None,
        clock: ClockPort | None = None,
        transaction: ExecutionTransactionPort | None = None,
        publisher: EventPublisherPort | None = None,
        exchange: str = "binance",
        user_stream_cache: UserStreamOrderCacheSinkPort | None = None,
        unknown_supervisor: UnknownOrderSupervisorPort | None = None,
        strict_dependencies: bool = False,
    ) -> None:
        if not isinstance(core, ExecutionService):
            raise TypeError("core must be an ExecutionService")
        if strict_dependencies:
            required = {
                "readiness": readiness,
                "single_writer": single_writer,
                "position_lock": position_lock,
                "market_rules": market_rules,
                "transaction": transaction,
                "publisher": publisher,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "strict HedgeExecutionEngine requires explicit dependencies: "
                    + ", ".join(missing)
                )
        self._core = core
        self._readiness = readiness or AlwaysReadyGate()
        self._single_writer = single_writer or InMemorySingleWriter()
        self._position_lock = position_lock or InMemoryPositionLock()
        self._market_rules = market_rules or StaticMarketRules()
        self._clock = clock or SystemClock()
        self._transaction = transaction or NullExecutionTransaction()
        self._publisher = publisher or NullEventPublisher()
        self._exchange = exchange.strip().lower()
        self._user_stream_cache = user_stream_cache
        self._unknown_supervisor = unknown_supervisor
        if not self._exchange:
            raise ValueError("exchange is required")

    @property
    def core(self) -> ExecutionService:
        return self._core

    @property
    def unknown_supervisor(self) -> UnknownOrderSupervisorPort | None:
        return self._unknown_supervisor

    def bind_unknown_supervisor(self, supervisor: UnknownOrderSupervisorPort) -> None:
        if supervisor is None:
            raise TypeError("unknown supervisor is required")
        register = getattr(supervisor, "register", None)
        run_due = getattr(supervisor, "run_due", None)
        if not callable(register) or not callable(run_due):
            raise TypeError("unknown supervisor must expose register/run_due")
        if self._unknown_supervisor is not None and self._unknown_supervisor is not supervisor:
            raise RuntimeError("unknown supervisor is already bound")
        self._unknown_supervisor = supervisor

    def run_unknown_recovery(self) -> tuple[object, ...]:
        """Run only query-based UNKNOWN recovery attempts; never resubmit an order."""
        if self._unknown_supervisor is None:
            return ()
        return tuple(self._unknown_supervisor.run_due())

    def _register_unknown(self, result: ExecutionResult) -> None:
        if (
            self._unknown_supervisor is not None
            and result.order.lifecycle.status is OrderState.UNKNOWN
        ):
            self._unknown_supervisor.register(result.order.client_order_id)

    def position_key(self, intent: OrderIntent) -> PositionKey:
        exchange = str(intent.metadata.get("exchange", self._exchange))
        return PositionKey(
            exchange=exchange,
            account_id=intent.account_id,
            symbol=intent.symbol,
            position_side=ContractPositionSide(intent.position_side.value),
        )

    def submit(self, intent: OrderIntent) -> ExecutionResult:
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        now = self._clock.now()
        expires_at = intent.metadata.get("expires_at")
        if expires_at is not None:
            if not isinstance(expires_at, datetime):
                raise ExecutionBlockedError("INTENT_EXPIRY_INVALID")
            if expires_at.tzinfo is None or expires_at.utcoffset() is None:
                raise ExecutionBlockedError("INTENT_EXPIRY_INVALID")
            if now >= expires_at:
                raise ExecutionBlockedError("INTENT_EXPIRED")
        key = self.position_key(intent)
        decision = self._readiness.evaluate(key)
        if intent.reduces_risk:
            if not decision.allow_reduce:
                raise ExecutionBlockedError(
                    ",".join(decision.reason_codes) or "READINESS_REDUCE_BLOCKED"
                )
        elif not decision.allow_increase:
            raise ExecutionBlockedError(
                ",".join(decision.reason_codes) or "READINESS_NOT_READY"
            )
        self._single_writer.assert_leader(account_id=intent.account_id, now=now)
        normalized = self._normalize(intent, key)
        with self._position_lock.acquire(key):
            result = self._core.submit(normalized)
        self._record_result(result, previous=None, event_type="INTENT_EXECUTED")
        self._register_unknown(result)
        return result

    def apply_exchange_event(self, snapshot: ExternalOrderSnapshot) -> ExecutionResult:
        if self._user_stream_cache is not None:
            self._user_stream_cache.put(snapshot)
        before = self._core.get_order(snapshot.client_order_id)
        key = self.position_key(before.intent)
        with self._position_lock.acquire(key):
            result = self._core.apply_exchange_event(snapshot)
        self._record_result(result, previous=before, event_type="ORDER_EVENT_APPLIED", snapshot=snapshot)
        return result

    def refresh_order(self, client_order_id: str) -> ExecutionResult:
        before = self._core.get_order(client_order_id)
        key = self.position_key(before.intent)
        with self._position_lock.acquire(key):
            result = self._core.refresh_order(client_order_id)
        self._record_result(result, previous=before, event_type="ORDER_REFRESHED")
        return result

    def resolve_unknown(self, client_order_id: str) -> ExecutionResult:
        before = self._core.get_order(client_order_id)
        key = self.position_key(before.intent)
        with self._position_lock.acquire(key):
            result = self._core.resolve_unknown(client_order_id)
        self._record_result(result, previous=before, event_type="UNKNOWN_RESOLUTION_ATTEMPTED")
        return result

    def cancel(self, client_order_id: str) -> ExecutionResult:
        before = self._core.get_order(client_order_id)
        key = self.position_key(before.intent)
        with self._position_lock.acquire(key):
            result = self._core.cancel(client_order_id)
        self._record_result(result, previous=before, event_type="ORDER_CANCEL_REQUESTED")
        return result

    def _normalize(self, intent: OrderIntent, key: PositionKey) -> OrderIntent:
        rules = self._market_rules.rules_for(key)
        quantity = rules.normalize_quantity(intent.quantity)
        if quantity < rules.minimum_quantity:
            raise ExecutionBlockedError("PRECISION_ZERO_OR_MIN_QTY")
        limit_price = intent.limit_price
        if intent.order_type is OrderType.LIMIT:
            if limit_price is None:
                raise ExecutionBlockedError("LIMIT_PRICE_REQUIRED")
            increases = intent.action.value in {"OPEN", "INCREASE"}
            is_long = intent.position_side.value == "LONG"
            order_side = "BUY" if increases == is_long else "SELL"
            limit_price = rules.normalize_price(
                limit_price,
                order_side=order_side,
                post_only=str(intent.metadata.get("time_in_force", "")).upper() == "GTX",
            )
            if limit_price <= 0:
                raise ExecutionBlockedError("PRICE_PRECISION_ZERO")
        reference = limit_price
        if reference is None:
            metadata_reference = intent.metadata.get("reference_price")
            if metadata_reference is not None:
                reference = Decimal(metadata_reference)
        if reference is not None and quantity * reference < rules.minimum_notional:
            raise ExecutionBlockedError("MIN_NOTIONAL")
        return replace(intent, quantity=quantity, limit_price=limit_price)

    def _record_result(
        self,
        result: ExecutionResult,
        *,
        previous: ExecutionOrder | None,
        event_type: str,
        snapshot: ExternalOrderSnapshot | None = None,
    ) -> None:
        order = result.order
        fill = self._derive_fill(previous, order, snapshot=snapshot)
        payload = self._payload(order)
        outbox = OutboxEvent(
            event_type=self._outbox_type(order, fill),
            payload=payload,
            correlation_id=(
                str(order.intent.action_group_id)
                if order.intent.action_group_id is not None
                else str(order.intent.intent_id)
            ),
            occurred_at=order.lifecycle.updated_at,
        )
        self._transaction.record(
            order=order,
            event_type=event_type,
            fill=fill,
            outbox=outbox,
            payload=payload,
        )
        # Event delivery is deliberately best-effort here. The transaction already contains
        # the outbox fact; a publisher outage must not make a successfully submitted order
        # look failed to the caller. OutboxDispatcher retries unpublished events.
        try:
            self._publisher.publish(outbox)
        except Exception:
            marker = getattr(self._transaction, "mark_publish_attempt", None)
            if callable(marker):
                marker(str(outbox.event_id))
        else:
            marker = getattr(self._transaction, "mark_published", None)
            if callable(marker):
                marker(str(outbox.event_id), published_at=self._clock.now())

    def _derive_fill(
        self,
        previous: ExecutionOrder | None,
        current: ExecutionOrder,
        *,
        snapshot: ExternalOrderSnapshot | None,
    ) -> FillEvent | None:
        before_qty = Decimal("0") if previous is None else previous.lifecycle.filled_quantity
        after_qty = current.lifecycle.filled_quantity
        delta = after_qty - before_qty
        if delta <= 0:
            return None
        after_avg = current.lifecycle.average_price
        if after_avg is None:
            return None
        before_avg = None if previous is None else previous.lifecycle.average_price
        before_notional = before_qty * (before_avg or Decimal("0"))
        delta_price = ((after_qty * after_avg) - before_notional) / delta
        raw_trade_id = snapshot.exchange_trade_id if snapshot is not None else None
        trade_id = raw_trade_id or sha256(
            f"{current.client_order_id}|{after_qty}|{after_avg}|{current.lifecycle.version}".encode()
        ).hexdigest()[:32]
        contract_side = ContractPositionSide(current.intent.position_side.value)
        contract_action = ContractIntentAction(current.intent.action.value)
        return FillEvent(
            position_key=self.position_key(current.intent),
            trade_id=trade_id,
            client_order_id=current.client_order_id,
            action=contract_action,
            order_side=expected_order_side(contract_side, contract_action),
            quantity=delta,
            price=delta_price,
            exchange_time=(snapshot.observed_at if snapshot is not None else current.lifecycle.updated_at),
            observed_time=self._clock.now(),
            fee=(Decimal("0") if snapshot is None else snapshot.last_fill_fee),
            fee_currency=("USDT" if snapshot is None else snapshot.fee_currency or "USDT"),
        )

    @staticmethod
    def _outbox_type(order: ExecutionOrder, fill: FillEvent | None) -> str:
        if fill is not None:
            return "FILL_RECORDED"
        return f"ORDER_{order.lifecycle.status.value}"

    def _payload(self, order: ExecutionOrder) -> Mapping[str, object]:
        key = self.position_key(order.intent)
        return {
            "exchange": key.exchange,
            "account_id": key.account_id,
            "symbol": key.symbol,
            "position_side": key.position_side.value,
            "intent_id": str(order.intent.intent_id),
            "client_order_id": order.client_order_id,
            "action": order.intent.action.value,
            "status": order.lifecycle.status.value,
            "approved_quantity": str(order.approved_quantity),
            "filled_quantity": str(order.lifecycle.filled_quantity),
            "average_price": (
                None if order.lifecycle.average_price is None else str(order.lifecycle.average_price)
            ),
            "version": order.lifecycle.version,
        }
