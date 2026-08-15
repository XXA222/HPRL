from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from freqtrade.hedge.execution.client_order_id import build_client_order_id
from freqtrade.hedge.execution.idempotency import ReservationState
from freqtrade.hedge.execution.service import (
    ExecutionOrder,
    ExecutionResult,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.persistence.hedge_execution_adapters import (
    SqlExecutionIdempotencyStore,
    SqlExecutionStore,
)
from freqtrade.persistence.hedge_models import (
    ExecutionIdempotencyRow,
    ExecutionOrderStateRow,
    HedgeModelBase,
)


def _database(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'execution.db'}")
    ExecutionOrderStateRow.__table__.create(engine, checkfirst=True)
    ExecutionIdempotencyRow.__table__.create(engine, checkfirst=True)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _order(
    *,
    key: str = "paper-cycle-1",
    version: int = 2,
    exchange: str = "paper",
) -> ExecutionOrder:
    intent = OrderIntent(
        account_id="paper-main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.50"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("2000"),
        metadata={
            "exchange": exchange,
            "planner_intent_id": "planner-1",
            "bucket": "CORE",
            "layer": 1,
            "cycle_time": datetime(2026, 8, 1, tzinfo=UTC),
            "exact": Decimal("0.1"),
        },
    )
    client_id = build_client_order_id(
        account_id=intent.account_id,
        symbol=intent.symbol,
        position_side=intent.position_side.value,
        idempotency_key=key,
    )
    lifecycle = OrderLifecycle(
        status=OrderState.ACKNOWLEDGED,
        filled_quantity=Decimal("0"),
        version=version,
        updated_at=datetime(2026, 8, 1, 1, tzinfo=UTC),
    )
    return ExecutionOrder(
        intent=intent,
        client_order_id=client_id,
        approved_quantity=Decimal("0.50"),
        lifecycle=lifecycle,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def test_execution_store_survives_new_session_and_preserves_exact_metadata(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        first = SqlExecutionStore(sessions)
        order = _order()
        first.put(order)

        recovered = SqlExecutionStore(sessions).get_by_client_order_id(order.client_order_id)
        assert recovered == order
        assert recovered is not None
        assert recovered.intent.metadata["exact"] == Decimal("0.1")
        assert recovered.intent.metadata["cycle_time"] == datetime(2026, 8, 1, tzinfo=UTC)
        assert recovered.leg_key == order.leg_key
    finally:
        engine.dispose()


def test_execution_store_rejects_lifecycle_regression_and_same_version_conflict(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        current = _order(version=3)
        store.put(current)
        with pytest.raises(ValueError, match="newer durable"):
            store.put(_order(version=2))
        conflicting = ExecutionOrder(
            intent=current.intent,
            client_order_id=current.client_order_id,
            approved_quantity=current.approved_quantity,
            lifecycle=OrderLifecycle(
                status=OrderState.UNKNOWN,
                version=3,
                updated_at=current.lifecycle.updated_at,
                reason="conflict",
            ),
            created_at=current.created_at,
        )
        with pytest.raises(ValueError, match="same lifecycle version"):
            store.put(conflicting)
    finally:
        engine.dispose()


def test_sql_idempotency_replays_authoritative_order_and_recovers_expired_lease(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        idempotency = SqlExecutionIdempotencyStore(
            sessions,
            store,
            lease_seconds=60,
            owner_id="worker-a",
        )
        order = _order()
        assert idempotency.reserve(order.intent.idempotency_key).state is ReservationState.NEW
        assert (
            idempotency.reserve(order.intent.idempotency_key).state
            is ReservationState.IN_FLIGHT
        )
        idempotency.complete(order.intent.idempotency_key, ExecutionResult(order=order))

        replay = SqlExecutionIdempotencyStore(
            sessions,
            SqlExecutionStore(sessions),
            owner_id="worker-b",
        ).reserve(order.intent.idempotency_key)
        assert replay.state is ReservationState.COMPLETED
        assert replay.value is not None
        assert replay.value.idempotent_replay is True
        assert replay.value.order == order

        with sessions.begin() as session:
            session.add(
                ExecutionIdempotencyRow(
                    idempotency_key="abandoned",
                    state=ReservationState.IN_FLIGHT.value,
                    client_order_id=None,
                    lease_owner="dead-worker",
                    lease_expires_at=datetime.now(UTC).replace(tzinfo=None)
                    - timedelta(seconds=1),
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                    updated_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )
        recovered = idempotency.reserve("abandoned")
        assert recovered.state is ReservationState.NEW
        with sessions() as session:
            row = session.scalar(
                select(ExecutionIdempotencyRow).where(
                    ExecutionIdempotencyRow.idempotency_key == "abandoned"
                )
            )
            assert row is not None
            assert row.lease_owner == "worker-a"
    finally:
        engine.dispose()


def test_idempotency_completion_is_atomic_with_execution_projection(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        idempotency = SqlExecutionIdempotencyStore(sessions, store, owner_id="worker-a")
        order = _order(key="not-reserved")
        with pytest.raises(KeyError, match="was not reserved"):
            idempotency.complete(order.intent.idempotency_key, ExecutionResult(order=order))
        assert store.get_by_client_order_id(order.client_order_id) is None
    finally:
        engine.dispose()


def test_execution_store_canonicalizes_metadata_enums_for_idempotent_same_version(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        current = _order()
        enum_metadata_order = ExecutionOrder(
            intent=OrderIntent(
                account_id=current.intent.account_id,
                symbol=current.intent.symbol,
                position_side=current.intent.position_side,
                action=current.intent.action,
                quantity=current.intent.quantity,
                idempotency_key=current.intent.idempotency_key,
                order_type=current.intent.order_type,
                limit_price=current.intent.limit_price,
                reduce_only=current.intent.reduce_only,
                intent_id=current.intent.intent_id,
                action_group_id=current.intent.action_group_id,
                metadata={**current.intent.metadata, "side_enum": PositionSide.LONG},
            ),
            client_order_id=current.client_order_id,
            approved_quantity=current.approved_quantity,
            lifecycle=current.lifecycle,
            created_at=current.created_at,
        )
        store.put(enum_metadata_order)
        store.put(enum_metadata_order)
        recovered = store.get_by_client_order_id(current.client_order_id)
        assert recovered is not None
        assert recovered.intent.metadata["side_enum"] == "LONG"
    finally:
        engine.dispose()


def test_execution_store_rejects_identity_mutation_across_lifecycle_versions(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        current = _order(version=2)
        store.put(current)
        mutated_intent = OrderIntent(
            account_id=current.intent.account_id,
            symbol=current.intent.symbol,
            position_side=current.intent.position_side,
            action=current.intent.action,
            quantity=Decimal("0.75"),
            idempotency_key=current.intent.idempotency_key,
            order_type=current.intent.order_type,
            limit_price=current.intent.limit_price,
            reduce_only=current.intent.reduce_only,
            intent_id=current.intent.intent_id,
            action_group_id=current.intent.action_group_id,
            metadata=current.intent.metadata,
        )
        mutated = ExecutionOrder(
            intent=mutated_intent,
            client_order_id=current.client_order_id,
            approved_quantity=current.approved_quantity,
            lifecycle=OrderLifecycle(
                status=OrderState.PARTIAL,
                filled_quantity=Decimal("0.1"),
                version=3,
                updated_at=current.lifecycle.updated_at + timedelta(seconds=1),
            ),
            created_at=current.created_at,
        )
        with pytest.raises(ValueError, match="mutate order identity"):
            store.put(mutated)
    finally:
        engine.dispose()


def test_idempotency_completion_and_release_require_reservation_owner(tmp_path) -> None:
    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        owner_a = SqlExecutionIdempotencyStore(sessions, store, owner_id="worker-a")
        owner_b = SqlExecutionIdempotencyStore(sessions, store, owner_id="worker-b")
        order = _order(key="owned-key")
        assert owner_a.reserve(order.intent.idempotency_key).state is ReservationState.NEW
        with pytest.raises(PermissionError, match="another execution worker"):
            owner_b.complete(order.intent.idempotency_key, ExecutionResult(order=order))
        owner_b.release(order.intent.idempotency_key)
        assert owner_b.reserve(order.intent.idempotency_key).state is ReservationState.IN_FLIGHT
        owner_a.release(order.intent.idempotency_key)
        assert owner_b.reserve(order.intent.idempotency_key).state is ReservationState.NEW
    finally:
        engine.dispose()


def test_paper_restores_scoped_sql_orders_without_checkpoint_and_converges_idempotency(
    tmp_path,
) -> None:
    from dataclasses import replace

    from freqtrade.hedge.execution.integrated_fake import build_integrated_fake_runtime
    from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication

    engine, sessions = _database(tmp_path)
    try:
        store = SqlExecutionStore(sessions)
        wanted = _order(key="wanted")
        from uuid import uuid4

        other_intent = replace(
            wanted.intent,
            account_id="other-account",
            idempotency_key="other-key",
            intent_id=uuid4(),
        )
        other = replace(
            wanted,
            intent=other_intent,
            client_order_id=build_client_order_id(
                account_id=other_intent.account_id,
                symbol=other_intent.symbol,
                position_side=other_intent.position_side.value,
                idempotency_key=other_intent.idempotency_key,
            ),
        )
        store.put(wanted)
        store.put(other)
        old_owner = SqlExecutionIdempotencyStore(
            sessions,
            store,
            owner_id="dead-worker",
        )
        assert old_owner.reserve(wanted.intent.idempotency_key).state is ReservationState.NEW

        recovering_idempotency = SqlExecutionIdempotencyStore(
            sessions,
            store,
            owner_id="restart-worker",
        )
        runtime = build_integrated_fake_runtime(
            store=store,
            idempotency=recovering_idempotency,
        )
        app = IntegratedPaperHedgeApplication(
            config={"hedge": {"paper": {"funding_source": "none", "ephemeral": True}}},
            account_id=wanted.intent.account_id,
            symbol=wanted.intent.symbol,
            execution_runtime=runtime,
        )

        active = app._active_execution_orders()
        assert [item.client_order_id for item in active] == [wanted.client_order_id]
        replay = recovering_idempotency.reserve(wanted.intent.idempotency_key)
        assert replay.state is ReservationState.COMPLETED
        assert replay.value is not None and replay.value.order == wanted
    finally:
        engine.dispose()
