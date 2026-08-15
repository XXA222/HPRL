from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType

import pytest

from freqtrade.hedge.execution.cancel_replace import CancelReplaceCoordinator
from freqtrade.hedge.execution.client_order_id import build_client_order_id
from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ApprovedOrderIntent,
    DefinitiveCancellationError,
    ExecutionBlockedError,
    ExecutionOrder,
    ExecutionResult,
    ExecutionService,
    ExternalOrderSnapshot,
    InMemoryAuditLog,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    RiskApproval,
)
from freqtrade.hedge.execution.state_machine import (
    InvalidOrderTransition,
    OrderLifecycle,
    OrderState,
)
from freqtrade.hedge.execution.unknown_resolver import UnknownOrderResolver
from freqtrade.hedge.telemetry.metrics import HedgeMetrics


def make_intent(
    key: str,
    *,
    quantity: object = Decimal("1"),
    action: IntentAction = IntentAction.OPEN,
    metadata: object = None,
    group=None,
) -> OrderIntent:
    return OrderIntent(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=action,
        quantity=quantity,
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("3000"),
        action_group_id=group,
        metadata={} if metadata is None else metadata,
    )


def make_service(exchange=None, *, kill_switch=None, idempotency=None):
    exchange = exchange or FakeExchangeExecutionPort()
    store = InMemoryExecutionStore()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=idempotency or InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=kill_switch or KillSwitch(),
    )
    return service, store, exchange


@pytest.mark.parametrize("value", [True, False, 0.1, float("inf"), float("nan")])
def test_order_intent_rejects_inexact_or_boolean_quantity(value) -> None:
    with pytest.raises(ValueError):
        make_intent("bad-quantity", quantity=value)


def test_market_intent_rejects_limit_price() -> None:
    with pytest.raises(ValueError, match="MARKET"):
        OrderIntent(
            account_id="main",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key="market-limit",
            order_type=OrderType.MARKET,
            limit_price=Decimal("3000"),
        )


@pytest.mark.parametrize(
    "symbol",
    ["BTC/USDT:USDC", "ETH USDT", "ETH💰USDT", "ETH/USDT:USDT:USDT"],
)
def test_symbol_rejects_ambiguous_or_unsupported_forms(symbol: str) -> None:
    with pytest.raises(ValueError):
        make_intent("bad-symbol").__class__(
            account_id="main",
            symbol=symbol,
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key="bad-symbol",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("3000"),
        )


def test_metadata_is_deeply_immutable_and_deterministic() -> None:
    source = {"nested": {"items": [1, 2]}, "set": {"b", "a"}}
    intent = make_intent("metadata", metadata=source)
    source["nested"]["items"].append(3)
    assert isinstance(intent.metadata, MappingProxyType)
    assert intent.metadata["nested"]["items"] == (1, 2)
    assert intent.metadata["set"] == ("a", "b")
    with pytest.raises(TypeError):
        intent.metadata["new"] = 1


def test_metadata_rejects_key_collision_after_strip() -> None:
    with pytest.raises(ValueError, match="collide"):
        make_intent("metadata-collision", metadata={"a": 1, " a ": 2})


def test_metadata_is_part_of_idempotency_semantics() -> None:
    service, _, exchange = make_service()
    first = service.submit(make_intent("semantic", metadata={"strategy": "a"}))
    assert first.order.lifecycle.status is OrderState.ACKNOWLEDGED
    with pytest.raises(ExecutionBlockedError):
        service.submit(make_intent("semantic", metadata={"strategy": "b"}))
    assert len(exchange.submit_calls) == 1


def test_risk_reason_codes_reject_plain_string() -> None:
    with pytest.raises(TypeError, match="sequence"):
        RiskApproval(True, Decimal("1"), "ABC")


def test_approved_intent_validates_client_order_id() -> None:
    with pytest.raises(ValueError):
        ApprovedOrderIntent(
            intent=make_intent("approved"),
            approved_quantity=Decimal("1"),
            client_order_id="bad id",
            approved_at=datetime.now(UTC),
            risk_reason_codes=(),
        )


def test_external_snapshot_rejects_internal_state() -> None:
    with pytest.raises(ValueError, match="externally observable"):
        ExternalOrderSnapshot("client", OrderState.PREPARED)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": OrderState.ACKNOWLEDGED, "filled_quantity": Decimal("0.1")},
        {"status": OrderState.PARTIAL, "filled_quantity": Decimal("0")},
        {"status": OrderState.FILLED, "filled_quantity": Decimal("0")},
        {"status": OrderState.REJECTED, "filled_quantity": Decimal("0.1")},
    ],
)
def test_external_snapshot_enforces_status_fill_consistency(kwargs) -> None:
    with pytest.raises(ValueError):
        ExternalOrderSnapshot(client_order_id="client", **kwargs)


def test_lifecycle_direct_construction_rejects_impossible_filled_state() -> None:
    with pytest.raises(InvalidOrderTransition):
        OrderLifecycle(status=OrderState.FILLED, filled_quantity=Decimal("0"))


def test_store_rejects_same_version_conflict() -> None:
    service, store, _ = make_service()
    order = service.submit(make_intent("store-conflict")).order
    conflict = replace(
        order,
        lifecycle=replace(order.lifecycle, reason="conflicting fact"),
    )
    with pytest.raises(ValueError, match="same lifecycle version"):
        store.put(conflict)


def test_explicit_unknown_response_queries_all_sources_without_resubmit() -> None:
    exchange = FakeExchangeExecutionPort()
    exchange.queue_snapshot(OrderState.UNKNOWN, reason="exchange uncertain")
    service, _, _ = make_service(exchange)
    result = service.submit(make_intent("explicit-unknown"))
    assert result.order.lifecycle.status is OrderState.UNKNOWN
    assert len(exchange.submit_calls) == 1
    assert exchange.query_calls == [result.order.client_order_id]
    assert exchange.open_order_queries == 1
    assert exchange.recent_fill_queries == 1


class CancelErrorExchange(FakeExchangeExecutionPort):
    def cancel_order(self, *, client_order_id: str):
        raise ConnectionError("uncertain cancel")


def test_non_timeout_cancel_error_becomes_unknown_then_queries() -> None:
    exchange = CancelErrorExchange()
    service, _, _ = make_service(exchange)
    order = service.submit(make_intent("cancel-error")).order
    result = service.cancel(order.client_order_id)
    assert result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert exchange.query_calls[-1] == order.client_order_id


class DefinitiveCancelExchange(FakeExchangeExecutionPort):
    def cancel_order(self, *, client_order_id: str):
        raise DefinitiveCancellationError("known not submitted")


def test_definitive_cancel_error_is_not_converted_to_unknown() -> None:
    exchange = DefinitiveCancelExchange()
    service, store, _ = make_service(exchange)
    order = service.submit(make_intent("cancel-definitive")).order
    with pytest.raises(DefinitiveCancellationError):
        service.cancel(order.client_order_id)
    assert store.get_by_client_order_id(order.client_order_id).lifecycle.status is (
        OrderState.ACKNOWLEDGED
    )


def test_canceled_order_absorbs_late_fills_and_can_become_filled() -> None:
    service, _, exchange = make_service()
    exchange.queue_snapshot(
        OrderState.PARTIAL,
        filled_quantity="0.4",
        average_price="3000",
    )
    order = service.submit(make_intent("late-fill")).order
    canceled = service.cancel(order.client_order_id).order
    assert canceled.lifecycle.status is OrderState.CANCELED
    late_partial = ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=OrderState.CANCELED,
        filled_quantity=Decimal("0.6"),
        average_price=Decimal("3001"),
    )
    updated = service.apply_snapshot(canceled, late_partial)
    assert updated.lifecycle.filled_quantity == Decimal("0.6")
    late_full = ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=OrderState.CANCELED,
        filled_quantity=Decimal("1"),
        average_price=Decimal("3002"),
    )
    filled = service.apply_snapshot(updated, late_full)
    assert filled.lifecycle.status is OrderState.FILLED


def test_full_partial_snapshot_is_normalized_to_filled() -> None:
    service, _, exchange = make_service()
    exchange.queue_snapshot(
        OrderState.PARTIAL,
        filled_quantity="1",
        average_price="3000",
    )
    result = service.submit(make_intent("full-partial"))
    assert result.order.lifecycle.status is OrderState.FILLED


class FillDuringCancelExchange(FakeExchangeExecutionPort):
    def cancel_order(self, *, client_order_id: str):
        current = self.query_order(client_order_id=client_order_id)
        return ExternalOrderSnapshot(
            client_order_id=client_order_id,
            status=OrderState.CANCELED,
            filled_quantity=Decimal("0.8"),
            average_price=Decimal("3000"),
            exchange_order_id=current.exchange_order_id,
        )


def test_cancel_replace_rechecks_remaining_after_cancel() -> None:
    exchange = FillDuringCancelExchange()
    service, _, _ = make_service(exchange)
    original = service.submit(make_intent("replace-original")).order
    replacement = make_intent("replace-new", quantity=Decimal("0.5"))
    result = CancelReplaceCoordinator(service).execute(
        original_client_order_id=original.client_order_id,
        replacement_intent=replacement,
    )
    assert not result.completed
    assert result.replacement is None
    assert len(exchange.submit_calls) == 1


def test_cancel_replace_preserves_action_group() -> None:
    from uuid import uuid4

    group = uuid4()
    service, _, _ = make_service()
    original = service.submit(make_intent("group-original", group=group)).order
    with pytest.raises(ValueError, match="action_group_id"):
        CancelReplaceCoordinator(service).execute(
            original_client_order_id=original.client_order_id,
            replacement_intent=make_intent("group-new"),
        )


class FailingCompleteStore(InMemoryIdempotencyStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def complete(self, key, value) -> None:
        if self.fail:
            self.fail = False
            raise OSError("persistence unavailable")
        super().complete(key, value)


def test_post_submit_idempotency_failure_never_releases_for_resubmit() -> None:
    exchange = FakeExchangeExecutionPort()
    idempotency = FailingCompleteStore()
    service, _, _ = make_service(exchange, idempotency=idempotency)
    with pytest.raises(OSError):
        service.submit(make_intent("complete-failure"))
    with pytest.raises(ExecutionBlockedError, match="in flight"):
        service.submit(make_intent("complete-failure"))
    assert len(exchange.submit_calls) == 1


def test_many_concurrent_identical_submits_send_only_once() -> None:
    service, _, exchange = make_service()
    intent = make_intent("concurrent-256")

    def submit_once(_):
        return service.submit(intent)

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(submit_once, range(256)))
    assert len(exchange.submit_calls) == 1
    assert sum(result.idempotent_replay for result in results) == 255


def test_service_uses_fixed_lock_stripes() -> None:
    service, _, _ = make_service()
    for index in range(1000):
        intent = OrderIntent(
            account_id=f"account-{index}",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key=f"key-{index}",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("3000"),
        )
        with service.leg_guard(intent):
            pass
    assert len(service._leg_locks) == 257


def test_kill_switch_emits_metrics_and_audit() -> None:
    metrics = HedgeMetrics()
    audit = InMemoryAuditLog()
    switch = KillSwitch(metrics=metrics, audit=audit)
    switch.activate(reason="manual", actor="admin")
    switch.deactivate(actor="admin", confirmed=True)
    assert metrics.snapshot()["halt_total"]
    assert [record.event for record in audit.records] == [
        "KILL_SWITCH_CHANGED",
        "KILL_SWITCH_CHANGED",
    ]


def test_stale_higher_fill_fact_is_absorbed_without_time_regression() -> None:
    service, _, _ = make_service()
    order = service.submit(make_intent("stale-fill")).order
    stale = ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=OrderState.PARTIAL,
        filled_quantity=Decimal("0.2"),
        average_price=Decimal("3000"),
        observed_at=order.lifecycle.updated_at - timedelta(seconds=1),
    )
    updated = service.apply_snapshot(order, stale)
    assert updated.lifecycle.filled_quantity == Decimal("0.2")
    assert updated.lifecycle.updated_at == order.lifecycle.updated_at


class MismatchedSubmitAckExchange(FakeExchangeExecutionPort):
    def submit_order(self, approved: ApprovedOrderIntent) -> ExternalOrderSnapshot:
        super().submit_order(approved)
        return ExternalOrderSnapshot(
            client_order_id="wrong-client-order-id",
            status=OrderState.ACKNOWLEDGED,
        )


def test_mismatched_submit_ack_queries_and_completes_idempotency() -> None:
    exchange = MismatchedSubmitAckExchange()
    service, _, _ = make_service(exchange)
    intent = make_intent("mismatched-submit-ack")

    first = service.submit(intent)
    second = service.submit(intent)

    assert first.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert second.idempotent_replay
    assert len(exchange.submit_calls) == 1
    assert exchange.query_calls == [first.order.client_order_id]


class MismatchedCancelAckExchange(FakeExchangeExecutionPort):
    def cancel_order(self, *, client_order_id: str) -> ExternalOrderSnapshot:
        self.cancel_calls.append(client_order_id)
        return ExternalOrderSnapshot(
            client_order_id="wrong-client-order-id",
            status=OrderState.CANCELED,
        )


def test_mismatched_cancel_ack_enters_recovery_instead_of_raising() -> None:
    exchange = MismatchedCancelAckExchange()
    service, _, _ = make_service(exchange)
    order = service.submit(make_intent("mismatched-cancel-ack")).order

    result = service.cancel(order.client_order_id)

    assert result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert exchange.cancel_calls == [order.client_order_id]
    assert exchange.query_calls[-1] == order.client_order_id


class UnknownReturningResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, order: ExecutionOrder) -> ExternalOrderSnapshot:
        self.calls += 1
        return ExternalOrderSnapshot(
            client_order_id=order.client_order_id,
            status=OrderState.UNKNOWN,
            reason="still uncertain",
        )


def test_resolver_returning_unknown_is_treated_as_unresolved() -> None:
    exchange = FakeExchangeExecutionPort()
    exchange.queue_snapshot(OrderState.UNKNOWN)
    resolver = UnknownReturningResolver()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=InMemoryExecutionStore(),
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=resolver,
        kill_switch=KillSwitch(),
    )

    result = service.submit(make_intent("resolver-still-unknown"))

    assert result.order.lifecycle.status is OrderState.UNKNOWN
    assert resolver.calls == 1
    assert len(exchange.submit_calls) == 1


def test_canceled_order_accepts_late_partial_status_with_more_fill() -> None:
    service, _, exchange = make_service()
    exchange.queue_snapshot(
        OrderState.PARTIAL,
        filled_quantity="0.2",
        average_price="3000",
    )
    order = service.submit(make_intent("late-partial-status")).order
    canceled = service.cancel(order.client_order_id).order

    updated = service.apply_snapshot(
        canceled,
        ExternalOrderSnapshot(
            client_order_id=order.client_order_id,
            status=OrderState.PARTIAL,
            filled_quantity=Decimal("0.4"),
            average_price=Decimal("3001"),
        ),
    )

    assert updated.lifecycle.status is OrderState.CANCELED
    assert updated.lifecycle.filled_quantity == Decimal("0.4")


def make_service_with_store(exchange, store, idempotency):
    return ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=idempotency,
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
    )


def test_fresh_idempotency_store_replays_existing_order_without_resubmit() -> None:
    exchange = FakeExchangeExecutionPort()
    store = InMemoryExecutionStore()
    first_service = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )
    intent = make_intent("restart-existing-order")
    first = first_service.submit(intent)

    restarted = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )
    replay = restarted.submit(intent)

    assert first.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert replay.idempotent_replay
    assert replay.order == first.order
    assert len(exchange.submit_calls) == 1


def test_restart_recovers_existing_unknown_instead_of_resubmitting() -> None:
    exchange = FakeExchangeExecutionPort()
    exchange.queue_snapshot(OrderState.UNKNOWN)
    store = InMemoryExecutionStore()
    intent = make_intent("restart-existing-unknown")
    first_service = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )
    first = first_service.submit(intent)
    assert first.order.lifecycle.status is OrderState.UNKNOWN

    exchange.set_order(
        ExternalOrderSnapshot(
            client_order_id=first.order.client_order_id,
            status=OrderState.ACKNOWLEDGED,
        )
    )
    restarted = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )
    recovered = restarted.submit(intent)

    assert recovered.idempotent_replay
    assert recovered.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert len(exchange.submit_calls) == 1


def test_restart_detects_existing_client_id_semantic_conflict() -> None:
    exchange = FakeExchangeExecutionPort()
    store = InMemoryExecutionStore()
    first_service = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )
    first_service.submit(make_intent("restart-conflict", quantity=Decimal("1")))

    restarted = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )
    with pytest.raises(ExecutionBlockedError, match="different intent"):
        restarted.submit(
            make_intent("restart-conflict", quantity=Decimal("2"))
        )
    assert len(exchange.submit_calls) == 1


def test_kill_switch_does_not_block_pure_idempotent_replay() -> None:
    kill_switch = KillSwitch()
    service, _, exchange = make_service(kill_switch=kill_switch)
    intent = make_intent("halted-replay")
    first = service.submit(intent)
    kill_switch.activate(reason="operator halt", actor="admin")

    replay = service.submit(intent)

    assert replay.idempotent_replay
    assert replay.order == first.order
    assert len(exchange.submit_calls) == 1


def test_kill_switch_does_not_block_existing_record_recovery_after_restart() -> None:
    exchange = FakeExchangeExecutionPort()
    store = InMemoryExecutionStore()
    intent = make_intent("halted-restart-replay")
    first = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    ).submit(intent)
    halted = KillSwitch()
    halted.activate(reason="operator halt", actor="admin")
    restarted = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=halted,
    )

    replay = restarted.submit(intent)

    assert replay.idempotent_replay
    assert replay.order == first.order
    assert len(exchange.submit_calls) == 1



@pytest.mark.parametrize("initial_state", [OrderState.PREPARED, OrderState.SUBMITTING])
def test_restart_turns_ambiguous_pre_submit_state_unknown_before_query(
    initial_state: OrderState,
) -> None:
    exchange = FakeExchangeExecutionPort()
    store = InMemoryExecutionStore()
    intent = make_intent(f"restart-{initial_state.value.lower()}")
    client_id = build_client_order_id(
        account_id=intent.account_id,
        symbol=intent.symbol,
        position_side=intent.position_side.value,
        idempotency_key=intent.idempotency_key,
    )
    now = datetime.now(UTC)
    lifecycle = OrderLifecycle(updated_at=now)
    if initial_state is OrderState.SUBMITTING:
        lifecycle = lifecycle.transition(
            OrderState.SUBMITTING,
            ordered_quantity=intent.quantity,
            occurred_at=now,
        )
    store.put(
        ExecutionOrder(
            intent=intent,
            client_order_id=client_id,
            approved_quantity=intent.quantity,
            lifecycle=lifecycle,
            created_at=now,
        )
    )
    exchange.set_order(
        ExternalOrderSnapshot(
            client_order_id=client_id,
            status=OrderState.ACKNOWLEDGED,
        )
    )
    restarted = make_service_with_store(
        exchange,
        store,
        InMemoryIdempotencyStore(),
    )

    result = restarted.submit(intent)

    assert result.idempotent_replay
    assert result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert exchange.query_calls == [client_id]
    assert exchange.submit_calls == []
