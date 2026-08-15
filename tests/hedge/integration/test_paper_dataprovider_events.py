from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.integration.controller import effective_candle_max_age_seconds
from freqtrade.hedge.integration.market_data import (
    build_dataprovider_market_input,
    exchange_market_rules,
)
from freqtrade.hedge.integration.paper_events import ExchangeFundingEventProvider
from freqtrade.hedge.integration.signal_provider import FreqtradeStrategySignalProvider
from freqtrade.hedge.paper_config import PaperOhlcvSource, PaperSimulationConfig


class _DataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        del pair, timeframe
        return self.frame, datetime(2026, 8, 1, 0, 5, 1, tzinfo=UTC)


class _Strategy:
    def version(self) -> str:
        return "r2"


class _Exchange:
    markets = {
        "ETH/USDT:USDT": {
            "precision": {"price": 2, "amount": 3},
            "limits": {
                "amount": {"min": 0.001},
                "cost": {"min": 5},
            },
        }
    }

    def get_fee(self, pair, taker_or_maker="maker"):
        del pair
        return 0.00015 if taker_or_maker == "maker" else 0.00035


class _FundingExchange:
    def __init__(self, settlement_ms: int) -> None:
        self.settlement_ms = settlement_ms
        self.history_calls = 0

    def fetch_funding_rate_history(self, symbol, since=None, limit=None):
        del symbol, since, limit
        self.history_calls += 1
        return [
            {
                "timestamp": self.settlement_ms,
                "fundingRate": "0.0001",
                "markPrice": "2000",
            }
        ]

    def fetch_funding_rate(self, symbol):
        del symbol
        return {
            "nextFundingTimestamp": self.settlement_ms + 8 * 60 * 60 * 1000,
            "fundingRate": "0.0002",
            "markPrice": "2000",
        }


def test_legacy_simulation_is_merged_into_one_strong_paper_config() -> None:
    hedge = {
        "paper": {"initial_balance": "1000"},
        "simulation": {
            "maker_fee": "0.0002",
            "taker_fee": "0.0004",
            "partial_fill_ratio": "0.25",
        },
    }
    config = PaperSimulationConfig.from_hedge_mapping(hedge)
    assert config.ohlcv_source is PaperOhlcvSource.DATAPROVIDER
    assert config.max_fill_ratio_per_order == Decimal("0.25")
    assert "simulation" not in hedge
    assert hedge["paper"]["maker_fee_rate"] == "0.0002"


def test_conflicting_legacy_and_canonical_simulation_config_fails_closed() -> None:
    hedge = {
        "paper": {"maker_fee_rate": "0.0003"},
        "simulation": {"maker_fee": "0.0002"},
    }
    with pytest.raises(OperationalException, match="conflicts"):
        PaperSimulationConfig.from_hedge_mapping(hedge)


def test_ticker_compat_requires_explicit_ephemeral_test_mode() -> None:
    with pytest.raises(OperationalException, match="test-only"):
        PaperSimulationConfig.from_hedge_mapping(
            {"paper": {"ohlcv_source": "ticker_compat"}}
        )


def test_signal_and_matching_bar_share_exact_analyzed_dataprovider_ohlcv() -> None:
    open_time = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            {
                "date": open_time,
                "open": 2000,
                "high": 2025,
                "low": 1980,
                "close": 2010,
                "volume": 123.45,
                "hedge_long_score": "0.8",
                "hedge_short_score": "0.2",
            }
        ]
    )
    signal = FreqtradeStrategySignalProvider(_DataProvider(frame), _Strategy()).signals(
        "ETH/USDT:USDT", "5m"
    )
    assert signal.candle is not None
    value = build_dataprovider_market_input(
        exchange=_Exchange(),
        pair="ETH/USDT:USDT",
        candle=signal.candle,
        fallback={},
        ticker={"bid": 2009, "ask": 2011, "last": 999999},
    )
    assert value.bar.open == Decimal("2000")
    assert value.bar.high == Decimal("2025")
    assert value.bar.low == Decimal("1980")
    assert value.bar.close == Decimal("2010")
    assert value.market.mark == Decimal("2010")
    assert value.market.mark != Decimal("999999")
    assert value.bar.timestamp == signal.candle_close_time
    assert value.rules.maker_fee_rate == Decimal("0.00015")
    assert value.rules.taker_fee_rate == Decimal("0.00035")
    assert value.rules.fee_source == "EXCHANGE_ACCOUNT_FEE"


def test_funding_provider_emits_actual_event_once_after_closed_bar_crosses_settlement() -> None:
    settlement = datetime(2026, 8, 1, 8, tzinfo=UTC)
    provider = ExchangeFundingEventProvider(
        _FundingExchange(int(settlement.timestamp() * 1000)),
        symbol="ETH/USDT:USDT",
    )
    from freqtrade.hedge.simulation.exchange import BarEvent

    before = BarEvent(
        timestamp=settlement - timedelta(minutes=5),
        symbol="ETH/USDT:USDT",
        open=Decimal("2000"),
        high=Decimal("2010"),
        low=Decimal("1990"),
        close=Decimal("2000"),
        volume=Decimal("10"),
    )
    assert provider.collect(before).events == ()
    after = BarEvent(
        timestamp=settlement,
        symbol=before.symbol,
        open=before.open,
        high=before.high,
        low=before.low,
        close=before.close,
        volume=before.volume,
    )
    first = provider.collect(after)
    second = provider.collect(after)
    assert len(first.events) == 1
    assert first.events[0].rate == Decimal("0.0001")
    assert second.events == ()
    assert provider._exchange.history_calls == 1
    assert provider.last_success_at is not None
    assert provider.healthy(now=provider.last_success_at + timedelta(seconds=1))


def test_funding_provider_fails_closed_when_exchange_has_no_funding_capability() -> None:
    with pytest.raises(RuntimeError, match="neither funding-rate history"):
        ExchangeFundingEventProvider(object(), symbol="ETH/USDT:USDT")


def test_funding_provider_does_not_charge_arbitrary_prestartup_history() -> None:
    settlement = datetime(2026, 8, 1, 0, tzinfo=UTC)
    provider = ExchangeFundingEventProvider(
        _FundingExchange(int(settlement.timestamp() * 1000)),
        symbol="ETH/USDT:USDT",
        max_age_seconds=3600,
    )
    from freqtrade.hedge.simulation.exchange import BarEvent

    much_later = BarEvent(
        timestamp=settlement + timedelta(hours=7),
        symbol="ETH/USDT:USDT",
        open=Decimal("2000"),
        high=Decimal("2010"),
        low=Decimal("1990"),
        close=Decimal("2000"),
        volume=Decimal("10"),
    )
    assert provider.collect(much_later).events == ()


def test_paper_funding_changes_wallet_once_and_tracks_durable_cursor() -> None:
    from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
    from freqtrade.hedge.planning.context import MarketSnapshot
    from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent

    app = IntegratedPaperHedgeApplication(
        config={
            "hedge": {
                "paper": {
                    "initial_balance": "1000",
                    "long_signal": "1",
                    "short_signal": "0",
                    "funding_source": "none",
                    "ephemeral": True,
                },
                "planner": {"cooldown_seconds": 0, "max_grid_layers": 1},
            }
        },
        account_id="paper-main",
        symbol="ETH/USDT:USDT",
    )
    timestamp = datetime(2026, 8, 1, 8, tzinfo=UTC)
    market = MarketSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=timestamp,
        bid=Decimal("99.9"),
        ask=Decimal("100.1"),
        mark=Decimal("100"),
        tick_size=Decimal("0.1"),
        qty_step=Decimal("0.001"),
    )
    bar = BarEvent(
        timestamp=timestamp,
        symbol=market.symbol,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1000"),
    )
    submitted = app.run_market_cycle(market, bar=bar)
    assert submitted.executions
    assert submitted.fills == ()
    next_time = timestamp + timedelta(minutes=1)
    next_market = MarketSnapshot(
        symbol=market.symbol,
        timestamp=next_time,
        bid=market.bid,
        ask=market.ask,
        mark=market.mark,
        tick_size=market.tick_size,
        qty_step=market.qty_step,
    )
    next_bar = BarEvent(
        timestamp=next_time,
        symbol=bar.symbol,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )
    app.run_market_cycle(next_market, bar=next_bar)
    wallet_before = app.wallet(next_market)
    assert wallet_before.long.quantity > 0
    before = wallet_before.balance
    funding = FundingEvent(
        timestamp=timestamp + timedelta(hours=8),
        symbol=market.symbol,
        rate=Decimal("0.01"),
        mark_price=Decimal("100"),
    )
    first = app._apply_funding_events((funding,))
    second = app._apply_funding_events((funding,))
    expected = (
        wallet_before.short.quantity - wallet_before.long.quantity
    ) * funding.mark_price * funding.rate
    assert len(first) == 1
    assert second == ()
    assert app.wallet(market).balance == before + expected
    assert app.last_funding_event_time == funding.timestamp


def test_semantically_equal_legacy_and_canonical_values_do_not_conflict() -> None:
    hedge = {
        "paper": {"maker_fee_rate": "0.00020", "max_fills_per_bar": 2},
        "simulation": {"maker_fee": Decimal("0.0002"), "max_fills_per_bar": 2},
    }
    config = PaperSimulationConfig.from_hedge_mapping(hedge)
    assert config.maker_fee_rate == Decimal("0.0002")
    assert config.max_fills_per_bar == 2


def test_funding_provider_uses_underlying_ccxt_history_when_wrapper_has_no_history() -> None:
    settlement = datetime(2026, 8, 1, 8, tzinfo=UTC)

    class RawApi:
        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            del symbol, since, limit
            return [
                {
                    "timestamp": int(settlement.timestamp() * 1000),
                    "fundingRate": "0.0001",
                    "markPrice": "2000",
                }
            ]

    class Wrapper:
        _api = RawApi()

        def fetch_funding_rate(self, symbol):
            del symbol
            return {
                "nextFundingTimestamp": int(
                    (settlement + timedelta(hours=8)).timestamp() * 1000
                ),
                "fundingRate": "0.0002",
                "markPrice": "2000",
            }

    from freqtrade.hedge.simulation.exchange import BarEvent

    provider = ExchangeFundingEventProvider(
        Wrapper(),
        symbol="ETH/USDT:USDT",
        initial_since_ms=int((settlement - timedelta(minutes=1)).timestamp() * 1000),
    )
    bar = BarEvent(
        timestamp=settlement,
        symbol="ETH/USDT:USDT",
        open=Decimal("2000"),
        high=Decimal("2010"),
        low=Decimal("1990"),
        close=Decimal("2000"),
        volume=Decimal("10"),
    )
    assert len(provider.collect(bar).events) == 1


def test_existing_durable_account_event_is_not_applied_twice_after_restart() -> None:
    from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
    from freqtrade.hedge.planning.context import MarketSnapshot
    from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent

    class ExistingEventSink:
        def record(self, event):
            del event
            return False

    app = IntegratedPaperHedgeApplication(
        config={
            "hedge": {
                "paper": {
                    "initial_balance": "1000",
                    "long_signal": "1",
                    "short_signal": "0",
                    "funding_source": "none",
                    "ephemeral": True,
                },
                "planner": {"cooldown_seconds": 0, "max_grid_layers": 1},
            }
        },
        account_id="paper-main",
        symbol="ETH/USDT:USDT",
        account_event_sink=ExistingEventSink(),
    )
    timestamp = datetime(2026, 8, 1, 8, tzinfo=UTC)
    market = MarketSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=timestamp,
        bid=Decimal("99.9"),
        ask=Decimal("100.1"),
        mark=Decimal("100"),
        tick_size=Decimal("0.1"),
        qty_step=Decimal("0.001"),
    )
    bar = BarEvent(
        timestamp=timestamp,
        symbol=market.symbol,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1000"),
    )
    submitted = app.run_market_cycle(market, bar=bar)
    assert submitted.executions
    next_time = timestamp + timedelta(minutes=1)
    next_market = MarketSnapshot(
        symbol=market.symbol,
        timestamp=next_time,
        bid=market.bid,
        ask=market.ask,
        mark=market.mark,
        tick_size=market.tick_size,
        qty_step=market.qty_step,
    )
    next_bar = BarEvent(
        timestamp=next_time,
        symbol=bar.symbol,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )
    app.run_market_cycle(next_market, bar=next_bar)
    before = app.wallet(next_market).balance
    funding = FundingEvent(
        timestamp=timestamp + timedelta(hours=8),
        symbol=market.symbol,
        rate=Decimal("0.01"),
        mark_price=Decimal("100"),
    )
    assert app._apply_funding_events((funding,)) == ()
    assert app.wallet(market).balance == before


def test_sql_paper_account_event_sink_is_idempotent_and_enqueues_outbox(tmp_path) -> None:
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.integration.paper_events import (
        SqlPaperAccountEventSink,
        fee_account_event,
    )
    from freqtrade.hedge.planning.context import PositionSide
    from freqtrade.persistence.hedge_models import (
        AccountEvent as AccountEventRow,
        EventOutbox,
        HedgeModelBase,
    )
    from freqtrade.persistence.hedge_service import HedgePersistenceService

    engine = create_engine(f"sqlite:///{tmp_path / 'paper-events.sqlite'}", future=True)
    HedgeModelBase.metadata.create_all(
        engine,
        tables=[AccountEventRow.__table__, EventOutbox.__table__],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    sink = SqlPaperAccountEventSink(
        HedgePersistenceService(factory),
        account_id="paper-main",
        exchange="paper",
        symbol="ETH/USDT:USDT",
        asset="USDT",
    )
    event = fee_account_event(
        fill_event_id="fill-1",
        timestamp=datetime(2026, 8, 1, 1, 0, tzinfo=UTC),
        symbol="ETH/USDT:USDT",
        amount=Decimal("0.42"),
        position_side=PositionSide.LONG,
    )

    assert sink.record(event) is True
    assert sink.record(event) is False

    with factory() as session:
        account_count = session.scalar(select(func.count()).select_from(AccountEventRow))
        outbox_count = session.scalar(select(func.count()).select_from(EventOutbox))
        row = session.scalar(select(AccountEventRow))
        outbox = session.scalar(select(EventOutbox))
    assert account_count == 1
    assert outbox_count == 1
    assert row is not None
    assert row.source == "LOCAL"
    assert row.event_type == "FEE"
    assert row.amount == "-0.42"
    assert outbox is not None
    assert outbox.event_type == "hedge.account.fee"
    recovered = sink.recover()
    assert recovered.event_ids == frozenset({event.event_id})
    assert recovered.funding_balance_delta == Decimal("0")
    assert recovered.last_funding_event_time is None


def test_sql_paper_account_event_recovery_closes_checkpoint_crash_window(tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.integration.paper_events import (
        SqlPaperAccountEventSink,
        funding_account_event,
    )
    from freqtrade.hedge.simulation.exchange import FundingEvent
    from freqtrade.persistence.hedge_models import (
        AccountEvent as AccountEventRow,
        EventOutbox,
        HedgeModelBase,
    )
    from freqtrade.persistence.hedge_service import HedgePersistenceService

    engine = create_engine(f"sqlite:///{tmp_path / 'funding-recovery.sqlite'}", future=True)
    HedgeModelBase.metadata.create_all(
        engine,
        tables=[AccountEventRow.__table__, EventOutbox.__table__],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    persistence = HedgePersistenceService(factory)
    sink = SqlPaperAccountEventSink(
        persistence,
        account_id="paper-main",
        exchange="paper",
        symbol="ETH/USDT:USDT",
    )
    settlement = datetime(2026, 8, 1, 8, tzinfo=UTC)
    event = funding_account_event(
        funding=FundingEvent(
            timestamp=settlement,
            symbol="ETH/USDT:USDT",
            rate=Decimal("0.0001"),
            mark_price=Decimal("2000"),
        ),
        amount=Decimal("-1.25"),
    )

    # Simulate the durable account-event commit completing before the auxiliary
    # Paper checkpoint.  A fresh adapter must recover the cash delta and cursor.
    assert sink.record(event) is True
    # A real-account funding fact may coexist in shadow mode but must never be
    # folded into the simulated Paper wallet.
    persistence.record_account_event(
        event_key="real-funding-1",
        account_id="paper-main",
        exchange="binance",
        event_type="FUNDING",
        asset="USDT",
        amount="99",
        source="WEBSOCKET",
        event_time=settlement,
        symbol="ETH/USDT:USDT",
    )
    fresh = SqlPaperAccountEventSink(
        HedgePersistenceService(factory),
        account_id="paper-main",
        exchange="paper",
        symbol="ETH/USDT:USDT",
    )
    recovered = fresh.recover()
    assert recovered.event_ids == frozenset({event.event_id})
    assert recovered.funding_balance_delta == Decimal("-1.25")
    assert recovered.last_funding_event_time == settlement
    assert fresh.record(event) is False


def test_candle_freshness_zero_uses_timeframe_aware_budget() -> None:
    assert effective_candle_max_age_seconds(0, "5m") == 600
    assert effective_candle_max_age_seconds(0, "4h") == 8 * 60 * 60
    assert effective_candle_max_age_seconds(123, "1d") == 123


def test_invalid_dataprovider_date_fails_closed() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "not-a-datetime",
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
                "volume": 3,
            }
        ]
    )
    with pytest.raises(ValueError, match="date must be a datetime"):
        FreqtradeStrategySignalProvider(_DataProvider(frame), _Strategy()).signals(
            "ETH/USDT:USDT", "5m"
        )


def test_market_rule_fallback_preserves_configured_rules_and_fees() -> None:
    rules = exchange_market_rules(
        exchange=object(),
        pair="ETH/USDT:USDT",
        fallback={
            "tick_size": "0.5",
            "qty_step": "0.25",
            "min_qty": "1",
            "min_notional": "20",
            "maker_fee_rate": "0.001",
            "taker_fee_rate": "0.002",
        },
    )
    assert rules.source == "CONFIG_FALLBACK"
    assert rules.fee_source == "CONFIG_FALLBACK"
    assert rules.tick_size == Decimal("0.5")
    assert rules.qty_step == Decimal("0.25")
    assert rules.min_qty == Decimal("1")
    assert rules.min_notional == Decimal("20")
    assert rules.maker_fee_rate == Decimal("0.001")
    assert rules.taker_fee_rate == Decimal("0.002")


def test_funding_poll_throttle_still_emits_due_pending_settlement() -> None:
    from freqtrade.hedge.simulation.exchange import BarEvent

    settlement = datetime(2026, 8, 1, 8, tzinfo=UTC)

    class CurrentOnly:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_funding_rate(self, symbol):
            del symbol
            self.calls += 1
            return {
                "nextFundingTimestamp": int(settlement.timestamp() * 1000),
                "nextFundingRate": "0.0003",
                "markPrice": "2000",
            }

    exchange = CurrentOnly()
    provider = ExchangeFundingEventProvider(
        exchange,
        symbol="ETH/USDT:USDT",
        poll_interval_seconds=3600,
    )
    before = BarEvent(
        timestamp=settlement - timedelta(minutes=5),
        symbol="ETH/USDT:USDT",
        open=Decimal("2000"),
        high=Decimal("2001"),
        low=Decimal("1999"),
        close=Decimal("2000"),
        volume=Decimal("1"),
    )
    assert provider.collect(before).events == ()
    due = BarEvent(
        timestamp=settlement,
        symbol=before.symbol,
        open=before.open,
        high=before.high,
        low=before.low,
        close=before.close,
        volume=before.volume,
    )
    result = provider.collect(due)
    assert len(result.events) == 1
    assert result.source == "EXCHANGE_FUNDING_CACHE"
    assert exchange.calls == 1


def test_paper_cycle_refreshes_matcher_precision_and_exchange_fee_rates() -> None:
    from freqtrade.hedge.integration.paper_runtime import IntegratedPaperHedgeApplication
    from freqtrade.hedge.planning.context import MarketSnapshot
    from freqtrade.hedge.simulation.exchange import BarEvent

    app = IntegratedPaperHedgeApplication(
        config={
            "hedge": {
                "paper": {
                    "initial_balance": "1000",
                    "long_signal": "0",
                    "short_signal": "0",
                    "funding_source": "none",
                    "ephemeral": True,
                    "tick_size": "9",
                    "qty_step": "9",
                    "maker_fee_rate": "0.01",
                    "taker_fee_rate": "0.02",
                }
            }
        },
        account_id="paper-main",
        symbol="ETH/USDT:USDT",
    )
    timestamp = datetime(2026, 8, 1, 0, 5, tzinfo=UTC)
    market = MarketSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=timestamp,
        bid=Decimal("2000"),
        ask=Decimal("2001"),
        mark=Decimal("2000.5"),
        tick_size=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.002"),
        min_notional=Decimal("10"),
    )
    bar = BarEvent(
        timestamp=timestamp,
        symbol=market.symbol,
        open=market.mark,
        high=Decimal("2010"),
        low=Decimal("1990"),
        close=market.mark,
        volume=Decimal("100"),
    )
    app.run_market_cycle(
        market,
        bar=bar,
        maker_fee_rate=Decimal("0.00015"),
        taker_fee_rate=Decimal("0.00035"),
    )
    assert app.matcher.config.price_tick == Decimal("0.01")
    assert app.matcher.config.qty_step == Decimal("0.001")
    assert app.matcher.config.min_fill_qty == Decimal("0.002")
    assert app.matcher.config.min_fill_notional == Decimal("10")
    assert app.matcher.config.maker_fee_rate == Decimal("0.00015")
    assert app.matcher.config.taker_fee_rate == Decimal("0.00035")


def test_funding_history_wins_over_revised_current_rate_for_same_settlement() -> None:
    from freqtrade.hedge.simulation.exchange import BarEvent

    settlement = datetime(2026, 8, 1, 8, tzinfo=UTC)
    timestamp_ms = int(settlement.timestamp() * 1000)

    class Exchange:
        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            del symbol, since, limit
            return [{"timestamp": timestamp_ms, "fundingRate": "0.0001"}]

        def fetch_funding_rate(self, symbol):
            del symbol
            return {
                "fundingTimestamp": timestamp_ms,
                "fundingRate": "0.0009",
            }

    provider = ExchangeFundingEventProvider(
        Exchange(),
        symbol="ETH/USDT:USDT",
        initial_since_ms=timestamp_ms - 1,
        poll_interval_seconds=0,
    )
    bar = BarEvent(
        timestamp=settlement,
        symbol="ETH/USDT:USDT",
        open=Decimal("2000"),
        high=Decimal("2010"),
        low=Decimal("1990"),
        close=Decimal("2000"),
        volume=Decimal("10"),
    )

    collected = provider.collect(bar)
    assert len(collected.events) == 1
    assert collected.events[0].rate == Decimal("0.0001")
    assert provider.collect(bar).events == ()


def test_production_paper_requires_durable_account_events_and_funding() -> None:
    with pytest.raises(OperationalException, match="funding_source='none'"):
        PaperSimulationConfig.from_hedge_mapping(
            {"paper": {"funding_source": "none"}}
        )
    with pytest.raises(OperationalException, match="account_events_enabled=false"):
        PaperSimulationConfig.from_hedge_mapping(
            {"paper": {"account_events_enabled": False}}
        )
