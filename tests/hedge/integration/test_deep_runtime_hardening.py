from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from freqtrade.commands.hedge_runtime_commands import _validate_paper_runtime_config
from freqtrade.exceptions import OperationalException
from freqtrade.hedge.integration.candle_cursor import bar_fingerprint
from freqtrade.hedge.integration.paper_runtime import (
    IntegratedPaperHedgeApplication,
    planner_config_from_mapping,
)
from freqtrade.hedge.integration.signal_provider import FreqtradeStrategySignalProvider
from freqtrade.hedge.planning.context import MarketSnapshot
from freqtrade.hedge.simulation.exchange import BarEvent
from freqtrade.optimize.hedge_backtesting import events_from_analyzed_dataframe

PAIR = "ETH/USDT:USDT"
START = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)


class _DataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        assert pair == PAIR
        assert timeframe == "5m"
        return self.frame, START + timedelta(hours=1)


class _Strategy:
    timeframe = "5m"

    def version(self) -> str:
        return "r22"


def _frame(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": START + timedelta(minutes=5 * index),
                "open": 100 + index,
                "high": 102 + index,
                "low": 99 + index,
                "close": 101 + index,
                "volume": 1000 + index,
                "hedge_long_score": "0.8",
                "hedge_short_score": "0.2",
            }
            for index in range(count)
        ]
    )


def _paper_config() -> dict[str, object]:
    return {
        "managed_pair": PAIR,
        "hedge": {
            "paper": {
                "ephemeral": True,
                "initial_balance": "1000",
                "leverage": "3",
                "auto_fill": True,
                "funding_source": "none",
                "account_events_enabled": False,
                "long_signal": "1",
                "short_signal": "0",
                "bar_volume": "1000",
            },
            "planner": {
                "short_enabled": False,
                "core_wallet_exposure_long": "0.20",
                "tactical_wallet_exposure_long": "0",
                "initial_entry_fraction": "1",
                "max_grid_layers": 1,
                "cooldown_seconds": 0,
                "trailing_rebound": "0",
            },
        },
    }


def _bar(ts: datetime, close: str = "100") -> BarEvent:
    value = Decimal(close)
    return BarEvent(
        timestamp=ts,
        symbol=PAIR,
        open=value,
        high=value + 2,
        low=value - 2,
        close=value,
        volume=Decimal("1000"),
    )


def _market(ts: datetime, close: str = "100") -> MarketSnapshot:
    value = Decimal(close)
    return MarketSnapshot(
        symbol=PAIR,
        timestamp=ts,
        bid=value - Decimal("0.1"),
        ask=value + Decimal("0.1"),
        mark=value,
        tick_size=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
    )


def test_signal_provider_returns_every_unseen_candle_after_durable_cursor() -> None:
    provider = FreqtradeStrategySignalProvider(_DataProvider(_frame()), _Strategy())
    all_rows = provider.signals_since(PAIR, "5m", after=None)
    assert len(all_rows) == 1
    assert all_rows[0].candle_close_time == START + timedelta(minutes=20)

    cursor_row = _frame().iloc[1]
    cursor_bar = BarEvent(
        timestamp=START + timedelta(minutes=10),
        symbol=PAIR,
        open=Decimal(str(cursor_row.open)),
        high=Decimal(str(cursor_row.high)),
        low=Decimal(str(cursor_row.low)),
        close=Decimal(str(cursor_row.close)),
        volume=Decimal(str(cursor_row.volume)),
    )
    caught_up = provider.signals_since(
        PAIR,
        "5m",
        after=cursor_bar.timestamp,
        cursor_fingerprint=bar_fingerprint(cursor_bar),
    )
    assert [item.candle_close_time for item in caught_up] == [
        START + timedelta(minutes=15),
        START + timedelta(minutes=20),
    ]


def test_signal_provider_rejects_revised_cursor_and_missing_candle_gap() -> None:
    frame = _frame()
    provider = FreqtradeStrategySignalProvider(_DataProvider(frame), _Strategy())
    cursor_close = START + timedelta(minutes=10)
    with pytest.raises(ValueError, match="revised"):
        provider.signals_since(
            PAIR,
            "5m",
            after=cursor_close,
            cursor_fingerprint="0" * 64,
        )

    gap = pd.concat([frame.iloc[:2], frame.iloc[3:]], ignore_index=True)
    provider = FreqtradeStrategySignalProvider(_DataProvider(gap), _Strategy())
    cursor = provider.signals_since(PAIR, "5m", after=None)[0]
    # Use a cursor before the retained rows so the missing 00:10 slot is observed.
    with pytest.raises(ValueError, match="missing candle"):
        provider.signals_since(
            PAIR,
            "5m",
            after=START + timedelta(minutes=5),
            max_missing_candles=0,
        )
    assert cursor.candle is not None


def test_paper_cursor_does_not_advance_when_checkpoint_commit_fails() -> None:
    class _FailingStore:
        def load(self):
            return None

        def save(self, payload):
            del payload
            raise OSError("simulated checkpoint failure")

    app = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="paper-r22",
        symbol=PAIR,
        state_store=_FailingStore(),
    )
    with pytest.raises(OSError, match="checkpoint"):
        app.run_market_cycle(_market(START), bar=_bar(START))
    assert app.last_market is None
    assert app.last_bar is None


def test_planner_config_rejects_unknown_and_conflicting_legacy_options() -> None:
    with pytest.raises(ValueError, match="unknown hedge.planner"):
        planner_config_from_mapping({"typo_grid_layers": 4})
    with pytest.raises(ValueError, match="conflicts"):
        planner_config_from_mapping(
            {"qty_scale": "1.1", "grid_qty_growth": "1.2"}
        )


def test_backtest_fingerprint_is_stable_and_gap_policy_is_fail_closed() -> None:
    first = events_from_analyzed_dataframe(
        pair=PAIR,
        timeframe="5m",
        frame=_frame(),
    )
    second = events_from_analyzed_dataframe(
        pair=PAIR,
        timeframe="5m",
        frame=_frame(),
    )
    assert first.data_fingerprint == second.data_fingerprint
    assert first.missing_candle_count == 0

    gap = pd.concat([_frame().iloc[:2], _frame().iloc[3:]], ignore_index=True)
    with pytest.raises(OperationalException, match="missing candle"):
        events_from_analyzed_dataframe(
            pair=PAIR,
            timeframe="5m",
            frame=gap,
            max_missing_candles=0,
        )


def _durable_runtime_config() -> dict[str, object]:
    return {
        "hedge_mode_enabled": True,
        "position_mode": "hedge",
        "managed_pair": PAIR,
        "trading_mode": "futures",
        "margin_mode": "cross",
        "dry_run": True,
        "dry_run_wallet": 1000,
        "max_open_trades": 1,
        "db_url": "sqlite:///user_data/hedge-paper.sqlite",
        "exchange": {"name": "binance", "pair_whitelist": [PAIR]},
        "hedge": {
            "operation_mode": "paper",
            "read_only": True,
            "live_trading_enabled": False,
            "target_leverage": "3",
            "paper": {
                "initial_balance": "1000",
                "leverage": "3",
                "state_backend": "sql",
                "ohlcv_source": "dataprovider",
                "funding_source": "exchange",
                "account_events_enabled": True,
            },
        },
    }


def test_paper_preflight_rejects_wallet_leverage_whitelist_and_capacity_drift() -> None:
    config = _durable_runtime_config()
    _validate_paper_runtime_config(config)

    broken = _durable_runtime_config()
    broken["dry_run_wallet"] = 999
    with pytest.raises(OperationalException, match="dry_run_wallet"):
        _validate_paper_runtime_config(broken)

    broken = _durable_runtime_config()
    broken["hedge"]["target_leverage"] = "5"  # type: ignore[index]
    with pytest.raises(OperationalException, match="target_leverage"):
        _validate_paper_runtime_config(broken)

    broken = _durable_runtime_config()
    broken["exchange"]["pair_whitelist"] = ["BTC/USDT:USDT"]  # type: ignore[index]
    with pytest.raises(OperationalException, match="whitelisted pair|pair_whitelist"):
        _validate_paper_runtime_config(broken)

    broken = _durable_runtime_config()
    broken["max_open_trades"] = 2
    with pytest.raises(OperationalException, match="max_open_trades"):
        _validate_paper_runtime_config(broken)


def test_fake_exchange_fill_id_is_exactly_once_and_fee_reaches_execution_ledger() -> None:
    from freqtrade.hedge.execution import (
        IntentAction,
        OrderIntent,
        OrderType,
        PositionSide,
        build_integrated_fake_runtime,
    )

    runtime = build_integrated_fake_runtime()
    submitted = runtime.engine.submit(
        OrderIntent(
            account_id="acct",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key="r22-fee",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
        )
    )
    first = runtime.exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.4",
        price="100",
        exchange_trade_id="r22-trade-1",
        fee="0.04",
    )
    replay = runtime.exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.4",
        price="100",
        exchange_trade_id="r22-trade-1",
        fee="0.04",
    )
    assert replay == first
    runtime.engine.apply_exchange_event(first)
    fill = runtime.ledger.fills()[0]
    assert fill.fee == Decimal("0.04")
    assert runtime.account.leg(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
    ).fees == Decimal("0.04")
    with pytest.raises(ValueError, match="conflicting fill"):
        runtime.exchange.fill_order(
            submitted.order.client_order_id,
            quantity="0.3",
            price="100",
            exchange_trade_id="r22-trade-1",
            fee="0.03",
        )


def test_sql_fill_facts_rebuild_account_bucket_and_fees_without_checkpoint(tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.execution import (
        IntentAction,
        OrderIntent,
        OrderType,
        PositionSide as ExecutionSide,
        build_integrated_fake_runtime,
    )
    from freqtrade.hedge.integration.paper_events import SqlPaperExecutionRecovery
    from freqtrade.hedge.planning.context import PositionSide
    from freqtrade.persistence.hedge_execution_adapters import (
        SqlExecutionIdempotencyStore,
        SqlExecutionLedger,
        SqlExecutionStore,
    )
    from freqtrade.persistence.hedge_models import HedgeModelBase

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'fills.db'}")
    HedgeModelBase.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        store = SqlExecutionStore(sessions)
        idempotency = SqlExecutionIdempotencyStore(
            sessions, store, owner_id="paper-first"
        )
        first = build_integrated_fake_runtime(
            store=store,
            idempotency=idempotency,
            transaction=SqlExecutionLedger(sessions),
        )
        submitted = first.engine.submit(
            OrderIntent(
                account_id="paper-main",
                symbol="ETHUSDT",
                position_side=ExecutionSide.LONG,
                action=IntentAction.OPEN,
                quantity=Decimal("0.5"),
                idempotency_key="sql-recovery-fill",
                order_type=OrderType.LIMIT,
                limit_price=Decimal("100"),
                metadata={"exchange": "paper", "bucket": "CORE", "layer": 0},
            )
        )
        snapshot = first.exchange.fill_order(
            submitted.order.client_order_id,
            quantity="0.5",
            price="100",
            exchange_trade_id="sql-fill-r22",
            fee="0.05",
        )
        first.engine.apply_exchange_event(snapshot)

        recovering_store = SqlExecutionStore(sessions)
        second = build_integrated_fake_runtime(
            store=recovering_store,
            idempotency=SqlExecutionIdempotencyStore(
                sessions, recovering_store, owner_id="paper-second"
            ),
            transaction=SqlExecutionLedger(sessions),
        )
        app = IntegratedPaperHedgeApplication(
            config=_paper_config(),
            account_id="paper-main",
            symbol=PAIR,
            execution_runtime=second,
            execution_recovery=SqlPaperExecutionRecovery(
                sessions, account_id="paper-main", symbol="ETHUSDT"
            ),
        )
        leg = app.wallet().long
        assert leg.quantity == Decimal("0.5")
        assert leg.core_quantity == Decimal("0.5")
        assert second.account.leg(
            account_id="paper-main",
            symbol="ETHUSDT",
            position_side=ExecutionSide.LONG,
        ).fees == Decimal("0.05")
    finally:
        engine.dispose()


def test_paper_preflight_normalizes_invalid_decimal_errors() -> None:
    broken = _durable_runtime_config()
    broken["dry_run_wallet"] = "not-a-number"
    with pytest.raises(OperationalException, match="finite decimal"):
        _validate_paper_runtime_config(broken)

    broken = _durable_runtime_config()
    broken["hedge"]["target_leverage"] = "Infinity"  # type: ignore[index]
    with pytest.raises(OperationalException, match="must be finite"):
        _validate_paper_runtime_config(broken)


def test_funding_provider_fails_when_declared_capability_is_not_implemented() -> None:
    from freqtrade.hedge.integration.paper_events import ExchangeFundingEventProvider

    class UnsupportedHistory:
        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            del symbol, since, limit
            raise NotImplementedError

    provider = ExchangeFundingEventProvider(
        UnsupportedHistory(),
        symbol=PAIR,
        poll_interval_seconds=0,
    )
    with pytest.raises(RuntimeError, match="unsupported at runtime"):
        provider.collect(_bar(START))
    assert provider.last_success_at is None


def test_checkpoint_active_order_preserves_last_fill_fee_metadata() -> None:
    from freqtrade.hedge.execution import (
        IntentAction,
        OrderIntent,
        OrderType,
        PositionSide as ExecutionSide,
        build_integrated_fake_runtime,
    )

    first_runtime = build_integrated_fake_runtime()
    submitted = first_runtime.engine.submit(
        OrderIntent(
            account_id="paper-fee-checkpoint",
            symbol=PAIR,
            position_side=ExecutionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key="checkpoint-fee-r22",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
            metadata={"exchange": "paper", "bucket": "CORE"},
        )
    )
    snapshot = first_runtime.exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.4",
        price="100",
        exchange_trade_id="checkpoint-fee-trade",
        fee="0.04",
        fee_currency="USDT",
    )
    first_runtime.engine.apply_exchange_event(snapshot)
    first_app = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="paper-fee-checkpoint",
        symbol=PAIR,
        execution_runtime=first_runtime,
    )
    rows = first_app._encode_active_execution_orders()
    assert rows[0]["external"]["last_fill_fee"] == "0.04"  # type: ignore[index]

    second_runtime = build_integrated_fake_runtime()
    second_app = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="paper-fee-checkpoint",
        symbol=PAIR,
        execution_runtime=second_runtime,
    )
    second_app._restore_active_execution_orders(rows)
    recovered = second_runtime.exchange.query_order(
        client_order_id=submitted.order.client_order_id
    )
    assert recovered is not None
    assert recovered.last_fill_fee == Decimal("0.04")
    assert recovered.fee_currency == "USDT"


def test_sql_fill_recovery_reconciles_missing_fee_account_event(tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.execution import (
        IntentAction,
        OrderIntent,
        OrderType,
        PositionSide as ExecutionSide,
        build_integrated_fake_runtime,
    )
    from freqtrade.hedge.integration.paper_events import (
        PaperAccountEventRecovery,
        SqlPaperExecutionRecovery,
    )
    from freqtrade.persistence.hedge_execution_adapters import (
        SqlExecutionIdempotencyStore,
        SqlExecutionLedger,
        SqlExecutionStore,
    )
    from freqtrade.persistence.hedge_models import HedgeModelBase

    class RecordingSink:
        def __init__(self) -> None:
            self.events = []

        def recover(self):
            return PaperAccountEventRecovery(
                event_ids=frozenset(),
                funding_balance_delta=Decimal("0"),
                last_funding_event_time=None,
            )

        def record(self, event):
            self.events.append(event)
            return True

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'fee-gap.db'}")
    HedgeModelBase.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        store = SqlExecutionStore(sessions)
        first = build_integrated_fake_runtime(
            store=store,
            idempotency=SqlExecutionIdempotencyStore(
                sessions, store, owner_id="fee-gap-first"
            ),
            transaction=SqlExecutionLedger(sessions),
        )
        submitted = first.engine.submit(
            OrderIntent(
                account_id="paper-fee-gap",
                symbol="ETHUSDT",
                position_side=ExecutionSide.LONG,
                action=IntentAction.OPEN,
                quantity=Decimal("0.5"),
                idempotency_key="fee-gap-order",
                order_type=OrderType.LIMIT,
                limit_price=Decimal("100"),
                metadata={"exchange": "paper", "bucket": "CORE"},
            )
        )
        first.engine.apply_exchange_event(
            first.exchange.fill_order(
                submitted.order.client_order_id,
                quantity="0.5",
                price="100",
                exchange_trade_id="fee-gap-fill",
                fee="0.05",
            )
        )

        recovering_store = SqlExecutionStore(sessions)
        second = build_integrated_fake_runtime(
            store=recovering_store,
            idempotency=SqlExecutionIdempotencyStore(
                sessions, recovering_store, owner_id="fee-gap-second"
            ),
            transaction=SqlExecutionLedger(sessions),
        )
        config = _paper_config()
        config["hedge"]["paper"]["account_events_enabled"] = True  # type: ignore[index]
        sink = RecordingSink()
        IntegratedPaperHedgeApplication(
            config=config,
            account_id="paper-fee-gap",
            symbol=PAIR,
            execution_runtime=second,
            execution_recovery=SqlPaperExecutionRecovery(
                sessions, account_id="paper-fee-gap", symbol="ETHUSDT"
            ),
            account_event_sink=sink,
        )
        assert len(sink.events) == 1
        assert sink.events[0].event_id == "paper-fee:fee-gap-fill"
        assert sink.events[0].amount == Decimal("-0.05")
    finally:
        engine.dispose()


def test_signal_provider_ignores_current_unfinished_candle() -> None:
    frame = _frame(3)
    clock = START + timedelta(minutes=12)
    provider = FreqtradeStrategySignalProvider(
        _DataProvider(frame),
        _Strategy(),
        clock=lambda: clock,
    )
    rows = provider.signals_since(PAIR, "5m", after=None)
    assert len(rows) == 1
    assert rows[0].candle_close_time == START + timedelta(minutes=10)


def test_failed_checkpoint_poison_requires_clean_restart() -> None:
    class _FailingStore:
        def load(self):
            return None

        def save(self, payload):
            del payload
            raise OSError("checkpoint unavailable")

    app = IntegratedPaperHedgeApplication(
        config=_paper_config(),
        account_id="paper-poison",
        symbol=PAIR,
        state_store=_FailingStore(),
    )
    with pytest.raises(OSError, match="checkpoint"):
        app.run_market_cycle(_market(START), bar=_bar(START))
    with pytest.raises(RuntimeError, match="requires restart"):
        app.run_market_cycle(
            _market(START + timedelta(minutes=5)),
            bar=_bar(START + timedelta(minutes=5)),
        )


def test_funding_provider_state_can_rollback_after_business_failure() -> None:
    from freqtrade.hedge.integration.paper_events import ExchangeFundingEventProvider

    settlement = START + timedelta(hours=8)

    class CurrentFunding:
        def fetch_funding_rate(self, symbol):
            del symbol
            return {
                "fundingTimestamp": int(settlement.timestamp() * 1000),
                "fundingRate": "0.0001",
                "markPrice": "100",
            }

    provider = ExchangeFundingEventProvider(
        CurrentFunding(),
        symbol=PAIR,
        initial_since_ms=int((settlement - timedelta(minutes=1)).timestamp() * 1000),
        poll_interval_seconds=0,
    )
    state = provider.snapshot_state()
    due_bar = _bar(settlement)
    first = provider.collect(due_bar)
    assert len(first.events) == 1
    provider.restore_state(state)
    replay = provider.collect(due_bar)
    assert replay.events == first.events


def test_backtest_result_has_stable_semantic_fingerprint_and_atomic_hash(tmp_path) -> None:
    import json

    from freqtrade.hedge.simulation.exchange import SimulationResult
    from freqtrade.optimize.hedge_backtesting import HedgeBacktestDataset, _write_result

    dataset = HedgeBacktestDataset(
        events=(),
        pair=PAIR,
        timeframe="5m",
        start=START,
        end=START + timedelta(minutes=5),
        bar_count=1,
        signal_count=1,
        funding_count=0,
        missing_candle_count=0,
        data_fingerprint="a" * 64,
    )
    result = SimulationResult(events=(), snapshots=(), report={"equity": Decimal("1000")})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_artifact, first_semantic, first_native = _write_result(
        path=first_path,
        result=result,
        dataset=dataset,
        strategy="TestStrategy",
        market_rule_source="TEST",
        market_rule_version="v1",
        export_events=False,
    )
    second_artifact, second_semantic, second_native = _write_result(
        path=second_path,
        result=result,
        dataset=dataset,
        strategy="TestStrategy",
        market_rule_source="TEST",
        market_rule_version="v1",
        export_events=False,
    )
    assert first_semantic == second_semantic
    assert first_native.to_dict() == second_native.to_dict()
    assert first_native.schema == "hedge-backtest-result-v4"
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["result_fingerprint"] == first_semantic
    assert first_path.with_suffix(".json.sha256").read_text(encoding="ascii").startswith(
        first_artifact
    )
    assert second_path.with_suffix(".json.sha256").read_text(encoding="ascii").startswith(
        second_artifact
    )
