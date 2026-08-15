from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from freqtrade.hedge.exchange.base import (
    AccountEventFact,
    ExchangeFactBatch,
    OrderFact,
    PositionFact,
)
from freqtrade.hedge.integration.repository import PersistenceMirroringReadonlyRepository
from freqtrade.persistence.hedge_models import AccountEvent, OrderSnapshot, PositionSnapshot
from freqtrade.persistence.hedge_service import HedgePersistenceService

NOW = datetime(2026, 7, 27, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_direction2_batch_is_mirrored_into_direction1_ledger(session_factory):
    repository = PersistenceMirroringReadonlyRepository(
        HedgePersistenceService(session_factory)
    )
    batch = ExchangeFactBatch(
        account_id="hedge-main",
        source="BINANCE_REST",
        observed_at=NOW,
        positions=(
            PositionFact(
                account_id="hedge-main",
                symbol="ETHUSDT",
                position_side="LONG",
                quantity=Decimal("1"),
                entry_price=Decimal("2000"),
                mark_price=Decimal("2010"),
                unrealized_pnl=Decimal("10"),
                liquidation_price=Decimal("1000"),
                leverage=3,
                margin_mode="CROSSED",
                update_time_ms=1_722_000_000_000,
                observed_at=NOW,
                source="BINANCE_REST",
            ),
        ),
        orders=(
            OrderFact(
                account_id="hedge-main",
                symbol="ETHUSDT",
                position_side="LONG",
                exchange_order_id="1001",
                client_order_id="fthedge-test",
                side="BUY",
                order_type="LIMIT",
                status="NEW",
                original_quantity=Decimal("0.2"),
                cumulative_filled_quantity=Decimal("0"),
                average_price=Decimal("0"),
                reduce_only=False,
                update_time_ms=1_722_000_000_100,
                observed_at=NOW,
                source="BINANCE_REST",
            ),
        ),
        account_events=(
            AccountEventFact(
                account_id="hedge-main",
                event_type="FUNDING_FEE",
                event_key="funding-1",
                event_time_ms=1_722_000_000_200,
                transaction_time_ms=1_722_000_000_200,
                payload={},
                observed_at=NOW,
                source="BINANCE_USER_STREAM",
                currency="USDT",
                amount=Decimal("-0.25"),
            ),
        ),
    )
    await repository.append_exchange_fact_batch(batch)

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(PositionSnapshot)) >= 1
        assert session.scalar(select(func.count()).select_from(OrderSnapshot)) == 1
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 1
