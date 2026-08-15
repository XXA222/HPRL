from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.exchange.base import (
    AccountSnapshotFact,
    ExchangeFactBatch,
    FillFact,
    OrderFact,
    OrderOrigin,
    PositionFact,
)
from freqtrade.hedge.exchange.contracts_bridge import batch_to_contract_envelopes


def _observed_at() -> datetime:
    return datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _account_snapshot() -> AccountSnapshotFact:
    observed_at = _observed_at()
    return AccountSnapshotFact(
        account_id="acct",
        total_wallet_balance=Decimal("1000"),
        total_available_balance=Decimal("800"),
        total_margin_balance=Decimal("1010"),
        total_initial_margin=Decimal("100"),
        total_maintenance_margin=Decimal("10"),
        total_unrealized_pnl=Decimal("10"),
        observed_at=observed_at,
        collection_started_at=observed_at,
        collection_completed_at=observed_at,
    )


def _position() -> PositionFact:
    return PositionFact(
        account_id="acct",
        symbol="ETHUSDT",
        position_side="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("2000"),
        mark_price=Decimal("2100"),
        unrealized_pnl=Decimal("100"),
        liquidation_price=Decimal("1200"),
        leverage=3,
        margin_mode="CROSSED",
        update_time_ms=1_722_081_600_000,
        observed_at=_observed_at(),
        source="BINANCE_REST",
    )


def _order() -> OrderFact:
    return OrderFact(
        account_id="acct",
        symbol="ETHUSDT",
        position_side="LONG",
        exchange_order_id="order-1",
        client_order_id="fthedge-order-1",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        original_quantity=Decimal("1"),
        cumulative_filled_quantity=Decimal("0"),
        average_price=Decimal("0"),
        reduce_only=False,
        update_time_ms=1_722_081_600_001,
        observed_at=_observed_at(),
        source="BINANCE_REST",
        origin=OrderOrigin.SYSTEM,
    )


def _fill() -> FillFact:
    return FillFact(
        account_id="acct",
        symbol="ETHUSDT",
        position_side="LONG",
        exchange_trade_id="trade-1",
        exchange_order_id="order-1",
        side="BUY",
        quantity=Decimal("0.25"),
        price=Decimal("2000"),
        commission=Decimal("0.2"),
        commission_asset="USDT",
        realized_pnl=Decimal("0"),
        event_time_ms=1_722_081_600_002,
        observed_at=_observed_at(),
        source="BINANCE_REST",
    )


def test_atomic_batch_maps_to_versioned_common_contract_envelopes() -> None:
    batch = ExchangeFactBatch(
        account_id="acct",
        source="BINANCE_REST",
        observed_at=_observed_at(),
        reconciliation_run_id="run-1",
        account_snapshot=_account_snapshot(),
        positions=(_position(),),
        orders=(_order(),),
        fills=(_fill(),),
        correlation_id="corr-1",
    )

    envelopes = batch_to_contract_envelopes(batch)

    assert {item.event_type for item in envelopes} == {
        "AccountSnapshot",
        "PositionSnapshot",
        "OrderSnapshot",
        "FillEvent",
    }
    position_event = next(
        item for item in envelopes if item.event_type == "PositionSnapshot"
    )
    assert position_event.contracts_version == "2.0"
    assert position_event.payload_version == 1
    assert position_event.correlation_id == "corr-1"
    assert position_event.payload["canonical_symbol"] == "ETH/USDT:USDT"
    assert position_event.payload["position_key"] == {
        "exchange": "binance",
        "account_id": "acct",
        "symbol": "ETH/USDT:USDT",
        "position_side": "LONG",
    }
    assert Decimal(position_event.payload["quantity"]) == Decimal("1")


def test_direction2_public_contract_types_are_exported() -> None:
    from freqtrade.hedge.exchange import (
        AtomicReadonlyFactRepository,
        ReadonlyExchangePort,
        TransportTelemetry,
    )
    from freqtrade.hedge.readonly import HistoryBackfillRequired

    assert AtomicReadonlyFactRepository.__name__ == "AtomicReadonlyFactRepository"
    assert ReadonlyExchangePort.__name__ == "ReadonlyExchangePort"
    assert TransportTelemetry.__name__ == "TransportTelemetry"
    assert HistoryBackfillRequired.__name__ == "HistoryBackfillRequired"
