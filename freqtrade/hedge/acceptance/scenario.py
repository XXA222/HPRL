from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from freqtrade.hedge.exchange.base import (
    AccountConfigurationFact,
    AccountSnapshotFact,
    BalanceFact,
    FillFact,
    OrderFact,
    PositionFact,
)
from freqtrade.hedge.acceptance.acceptance import RuntimeAcceptanceEngine
from freqtrade.hedge.acceptance.clock import ClockSample, evaluate_clock
from freqtrade.hedge.acceptance.events import EventEnvelope
from freqtrade.hedge.acceptance.facts import build_fact_plane
from freqtrade.hedge.acceptance.models import AcceptancePolicy
from freqtrade.hedge.acceptance.persistence import RuntimeAcceptanceStore
from freqtrade.hedge.acceptance.reconciliation import clone_plane


def _now() -> datetime:
    return datetime.now(UTC)


def _configuration(now: datetime) -> AccountConfigurationFact:
    return AccountConfigurationFact(
        account_id="acct",
        hedge_mode=True,
        active_margin_modes=("cross",),
        leverage_by_symbol_side={"BTCUSDT:LONG": 3, "BTCUSDT:SHORT": 3},
        observed_at=now,
        raw={},
    )


def _snapshot(now: datetime) -> AccountSnapshotFact:
    return AccountSnapshotFact(
        account_id="acct",
        total_wallet_balance=Decimal(1000),
        total_available_balance=Decimal(900),
        total_margin_balance=Decimal(1005),
        total_initial_margin=Decimal(100),
        total_maintenance_margin=Decimal(10),
        total_unrealized_pnl=Decimal(5),
        observed_at=now,
        collection_started_at=now - timedelta(milliseconds=20),
        collection_completed_at=now,
        raw={},
    )


def _balances(now: datetime) -> tuple[BalanceFact, ...]:
    return (
        BalanceFact(
            account_id="acct",
            asset="USDT",
            wallet_balance=Decimal(1000),
            available_balance=Decimal(900),
            cross_wallet_balance=Decimal(1000),
            unrealized_pnl=Decimal(5),
            observed_at=now,
            source="REST",
            raw={},
        ),
    )


def _positions(now: datetime) -> tuple[PositionFact, ...]:
    return (
        PositionFact(
            account_id="acct",
            symbol="BTCUSDT",
            position_side="LONG",
            quantity=Decimal("0.01"),
            entry_price=Decimal(60000),
            mark_price=Decimal(60500),
            unrealized_pnl=Decimal(5),
            liquidation_price=Decimal(30000),
            leverage=3,
            margin_mode="cross",
            update_time_ms=1_000,
            observed_at=now,
            source="REST",
            raw={},
        ),
        PositionFact(
            account_id="acct",
            symbol="BTCUSDT",
            position_side="SHORT",
            quantity=Decimal(0),
            entry_price=Decimal(0),
            mark_price=Decimal(60500),
            unrealized_pnl=Decimal(0),
            liquidation_price=None,
            leverage=3,
            margin_mode="cross",
            update_time_ms=1_000,
            observed_at=now,
            source="REST",
            raw={},
        ),
    )


def _orders(now: datetime) -> tuple[OrderFact, ...]:
    return (
        OrderFact(
            account_id="acct",
            symbol="BTCUSDT",
            position_side="LONG",
            exchange_order_id="101",
            client_order_id="fthedge-101",
            side="BUY",
            order_type="LIMIT",
            status="NEW",
            original_quantity=Decimal("0.01"),
            cumulative_filled_quantity=Decimal(0),
            average_price=Decimal(0),
            reduce_only=False,
            update_time_ms=1_000,
            observed_at=now,
            source="REST",
            raw={},
        ),
    )


def _fills(now: datetime) -> tuple[FillFact, ...]:
    return (
        FillFact(
            account_id="acct",
            symbol="BTCUSDT",
            position_side="LONG",
            exchange_trade_id="501",
            exchange_order_id="100",
            side="BUY",
            quantity=Decimal("0.01"),
            price=Decimal(60000),
            commission=Decimal("0.1"),
            commission_asset="USDT",
            realized_pnl=Decimal(0),
            event_time_ms=1_000,
            observed_at=now,
            source="REST",
            raw={},
        ),
    )


def _income() -> tuple[dict[str, object], ...]:
    return (
        {
            "incomeType": "FUNDING_FEE",
            "tranId": "701",
            "time": 1_000,
            "symbol": "BTCUSDT",
            "asset": "USDT",
            "income": "0.25",
        },
        {
            "incomeType": "COMMISSION",
            "tranId": "702",
            "time": 1_001,
            "symbol": "BTCUSDT",
            "asset": "USDT",
            "income": "-0.1",
        },
    )


def _event(
    event_type: str, *, transaction_ms: int, event_ms: int, trade_id: str = ""
) -> EventEnvelope:
    return EventEnvelope(
        account_id="acct",
        event_type=event_type,
        event_time_ms=event_ms,
        transaction_time_ms=transaction_ms,
        symbol="BTCUSDT",
        position_side="LONG",
        order_id="101",
        trade_id=trade_id,
        payload={"e": event_type, "E": event_ms, "T": transaction_ms, "tranId": "fund-1"},
    )


def run_deterministic_acceptance(*, project_root: Path, output_db: Path | None = None):
    now = _now()
    config = {
        "exchange": {
            "name": "binance",
            "pair_whitelist": ["BTC/USDT:USDT"],
        },
        "hedge_mode_enabled": True,
        "db_url": "sqlite:///runtime.db",
        "hedge": {
            "read_only": True,
            "live_trading_enabled": False,
            "operation_mode": "readonly",
            "managed_symbols": ["BTCUSDT"],
        },
    }
    temporary = TemporaryDirectory() if output_db is None else None
    db_path = output_db or Path(temporary.name) / "acceptance.sqlite"
    store = RuntimeAcceptanceStore(db_path)
    try:
        engine = RuntimeAcceptanceEngine(
            config=config,
            project_root=project_root,
            account_id="acct",
            managed_symbols=("BTCUSDT",),
            store=store,
            policy=AcceptancePolicy(),
            live_evidence=False,
        )
        configuration = _configuration(now)
        snapshot = _snapshot(now)
        balances = _balances(now)
        positions = _positions(now)
        orders = _orders(now)
        fills = _fills(now)
        income = _income()
        engine.round01_baseline()
        engine.round02_clock(
            evaluate_clock(
                (
                    ClockSample(1_000, 1_006, 1_010),
                    ClockSample(2_000, 2_006, 2_010),
                    ClockSample(3_000, 3_006, 3_010),
                ),
                max_abs_skew_ms=1_000,
                max_rtt_ms=5_000,
            )
        )
        engine.round03_assets(snapshot, balances)
        engine.round04_configuration(configuration, target_leverage=3)
        engine.round05_identity(positions, configuration)
        engine.round06_orders(orders, orders)
        engine.round07_trades(fills)
        engine.round08_income(income)
        engine.round09_rest_snapshot(
            positions=positions,
            balances=balances,
            orders=orders,
            fills=fills,
            income=income,
            observed_at=now,
        )
        engine.round10_user_stream()
        engine.round11_account_update(
            _event("ACCOUNT_UPDATE", transaction_ms=2_000, event_ms=2_000)
        )
        engine.round12_order_trade_update(
            _event("ORDER_TRADE_UPDATE", transaction_ms=2_100, event_ms=2_100, trade_id="501")
        )
        fill_event = _event(
            "ORDER_TRADE_UPDATE", transaction_ms=2_200, event_ms=2_200, trade_id="502"
        )
        funding_event = _event("ACCOUNT_UPDATE", transaction_ms=2_300, event_ms=2_300)
        engine.round13_duplicates(fill_event, funding_event)
        engine.round14_out_of_order(
            _event("ORDER_TRADE_UPDATE", transaction_ms=4_000, event_ms=4_000, trade_id="900"),
            _event("ORDER_TRADE_UPDATE", transaction_ms=3_000, event_ms=3_000, trade_id="901"),
        )
        engine.round15_gap_recovery()
        plane = build_fact_plane(
            account_id="acct",
            observed_at=now,
            positions=positions,
            balances=balances,
            orders=orders,
            fills=fills,
            income=income,
        )
        memory = clone_plane(plane)
        database = clone_plane(plane)
        store.save_plane("MEMORY", memory)
        store.save_plane("DB", database)
        engine.round16_fast_reconciliation(memory, database)
        engine.round17_deep_reconciliation(memory, database)
        engine.round18_crash_recovery(state_hash=plane.fingerprint())
        engine.round19_fault_injection()
        engine.session.note(
            "Deterministic acceptance validates runtime acceptance logic only; "
            "it is not Binance live/soak evidence."
        )
        engine.round20_readiness(observed_duration=timedelta(hours=72), target_stage="72h")
        return engine.session.finalize()
    finally:
        store.close()
        if temporary is not None:
            temporary.cleanup()
