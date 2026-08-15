from datetime import datetime, timedelta, timezone
from decimal import Decimal

from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent
from freqtrade.optimize.hedge_backtesting import HedgeBacktesting
from freqtrade.util.hedge_dry_run_wallet import HedgeDryRunWallet

START = datetime(2026, 7, 26, tzinfo=timezone.utc)


def events():
    return [
        BarEvent(START, "ETH/USDT:USDT", Decimal("100"), Decimal("102"), Decimal("96"), Decimal("99"), Decimal("1000")),
        BarEvent(START + timedelta(minutes=5), "ETH/USDT:USDT", Decimal("99"), Decimal("104"), Decimal("95"), Decimal("103"), Decimal("1000")),
        FundingEvent(START + timedelta(minutes=7), "ETH/USDT:USDT", Decimal("0.0001"), Decimal("103")),
        BarEvent(START + timedelta(minutes=10), "ETH/USDT:USDT", Decimal("103"), Decimal("108"), Decimal("97"), Decimal("101"), Decimal("1000")),
    ]


def test_backtest_and_dry_run_share_exact_results():
    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))
    bt = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=cfg).run(events())
    dr = HedgeDryRunWallet(initial_balance=Decimal("1000"), planner_config=cfg).replay(events())
    assert bt.snapshots == dr.snapshots
    assert bt.report == dr.report


def test_required_report_fields_exist():
    result = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))).run(events())
    required = {
        "long_pnl", "short_pnl", "core_cost_change_long", "core_cost_change_short",
        "tactical_trading_pnl", "gross_peak", "net_exposure", "funding", "fees",
        "dual_leg_duration_seconds", "add_count", "reduce_count", "max_drawdown",
    }
    assert required <= result.report.keys()


def test_standard_events_are_deterministic():
    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))
    one = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=cfg).run(events())
    two = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=cfg).run(events())
    assert one.events == two.events


def test_same_adapter_can_be_reused_without_state_leakage():
    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))
    adapter = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=cfg)
    one = adapter.run(events())
    two = adapter.run(events())
    assert one == two


def test_mixed_symbols_are_rejected_in_single_symbol_engine():
    import pytest

    mixed = events() + [
        BarEvent(
            START + timedelta(minutes=15),
            "BTC/USDT:USDT",
            Decimal("50000"),
            Decimal("50100"),
            Decimal("49900"),
            Decimal("50000"),
            Decimal("100"),
        )
    ]
    with pytest.raises(ValueError, match="single-symbol"):
        HedgeBacktesting(initial_balance=Decimal("1000")).run(mixed)


def test_report_pnl_reconciles_to_final_equity():
    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0")),
    ).run(events())
    assert result.report["pnl_reconciliation_error"] == Decimal("0")


def test_intrabar_drawdown_is_recorded_even_when_bar_closes_at_open():
    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0.30"),
        tactical_wallet_exposure_long=Decimal("0"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=1,
        cooldown_seconds=0,
        trailing_rebound=Decimal("0"),
    )
    result = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=cfg).run(
        [
            BarEvent(
                START,
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
            BarEvent(
                START + timedelta(minutes=5),
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("50"),
                Decimal("100"),
                Decimal("1000"),
            ),
        ]
    )
    assert result.report["max_drawdown"] > Decimal("0")


def test_partial_entry_layer_counts_once_after_completion():
    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0.30"),
        tactical_wallet_exposure_long=Decimal("0"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=4,
        cooldown_seconds=0,
        trailing_rebound=Decimal("0"),
    )
    engine = HedgeBacktesting(initial_balance=Decimal("1000"), planner_config=cfg).engine
    bars = [
        BarEvent(
            START + timedelta(minutes=5 * index),
            "ETH/USDT:USDT",
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("10"),
        )
        for index in range(4)
    ]
    engine.replay(bars)
    assert engine.long_state.grid_layers_filled == 1


def test_incremental_dry_run_matches_one_shot_backtest():
    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))
    all_events = events()
    expected = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
    ).run(all_events)
    dry_run = HedgeDryRunWallet(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
    )
    first = dry_run.advance(all_events[:2])
    second = dry_run.advance(all_events[2:])
    assert first.snapshots + second.snapshots == expected.snapshots
    assert first.events + second.events == expected.events
    assert second.report == expected.report


def test_incremental_dry_run_rejects_backwards_and_duplicate_events():
    import pytest

    dry_run = HedgeDryRunWallet(initial_balance=Decimal("1000"))
    first = events()[0]
    dry_run.advance([first])
    with pytest.raises(ValueError, match="duplicate standard event"):
        dry_run.advance([first])
    older = BarEvent(
        START - timedelta(minutes=5),
        first.symbol,
        Decimal("100"),
        Decimal("100"),
        Decimal("100"),
        Decimal("100"),
        Decimal("1"),
    )
    with pytest.raises(ValueError, match="cannot move backwards"):
        dry_run.advance([older])


def test_failed_incremental_symbol_validation_does_not_corrupt_state():
    import pytest

    all_events = events()
    dry_run = HedgeDryRunWallet(initial_balance=Decimal("1000"))
    first = dry_run.advance([all_events[0]])
    mixed = BarEvent(
        all_events[1].timestamp,
        "BTC/USDT:USDT",
        Decimal("50000"),
        Decimal("50100"),
        Decimal("49900"),
        Decimal("50000"),
        Decimal("1"),
    )
    with pytest.raises(ValueError, match="single-symbol"):
        dry_run.advance([mixed])
    second = dry_run.advance([all_events[1]])
    expected = HedgeBacktesting(initial_balance=Decimal("1000")).run(
        all_events[:2]
    )
    assert first.snapshots + second.snapshots == expected.snapshots
    assert second.report == expected.report


def test_signal_event_plans_at_close_and_fills_on_next_bar():
    from freqtrade.hedge.simulation.exchange import SignalEvent

    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0"),
        tactical_wallet_exposure_long=Decimal("0.30"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=1,
        cooldown_seconds=0,
        trailing_rebound=Decimal("0"),
    )
    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
        long_signal=Decimal("0"),
    ).run(
        [
            SignalEvent(
                START,
                "ETH/USDT:USDT",
                Decimal("1"),
                Decimal("0"),
            ),
            BarEvent(
                START,
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
            BarEvent(
                START + timedelta(minutes=5),
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
        ]
    )
    assert result.snapshots[0].long_quantity == Decimal("0")
    assert result.snapshots[-1].long_quantity == Decimal("3.0000")
    assert result.events[0].long_signal == Decimal("1")


def test_incremental_signal_events_match_one_shot_replay():
    from freqtrade.hedge.simulation.exchange import SignalEvent

    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))
    stream = [
        SignalEvent(START, "ETH/USDT:USDT", Decimal("1"), Decimal("0")),
        events()[0],
        SignalEvent(
            START + timedelta(minutes=5),
            "ETH/USDT:USDT",
            Decimal("0"),
            Decimal("1"),
        ),
        events()[1],
    ]
    expected = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
    ).run(stream)
    dry_run = HedgeDryRunWallet(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
    )
    first = dry_run.advance(stream[:2])
    second = dry_run.advance(stream[2:])
    assert first.events + second.events == expected.events
    assert first.snapshots + second.snapshots == expected.snapshots
    assert second.report == expected.report


def test_public_planning_port_can_be_injected():
    from freqtrade.hedge.planning.context import PlanningResult

    class EmptyPlanner:
        def __init__(self):
            self.calls = 0

        def plan(self, context):
            self.calls += 1
            return PlanningResult(
                ideal_orders=(),
                submit_orders=(),
                cancel_order_ids=(),
                kept_order_ids=(),
                long_state=context.long_state.next_sequence(),
                short_state=context.short_state.next_sequence(),
            )

    planner = EmptyPlanner()
    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner=planner,
    ).run([events()[0]])
    assert planner.calls == 1
    assert result.snapshots[-1].long_quantity == Decimal("0")
    assert result.snapshots[-1].short_quantity == Decimal("0")


def test_report_contains_liquidity_and_margin_breakdown():
    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0")),
    ).run(events())
    required = {
        "maker_fees",
        "taker_fees",
        "maker_fill_count",
        "taker_fill_count",
        "available_balance",
        "active_order_margin",
        "maintenance_margin",
        "margin_ratio",
        "long_add_count",
        "short_add_count",
    }
    assert required <= result.report.keys()
    assert result.report["maker_fees"] + result.report["taker_fees"] == result.report["fees"]


def test_dry_run_checkpoint_restore_replays_identically():
    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0"))
    stream = events()[:2]
    dry_run = HedgeDryRunWallet(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
    )
    first = dry_run.advance([stream[0]])
    checkpoint = dry_run.checkpoint()
    second = dry_run.advance([stream[1]])
    dry_run.restore(checkpoint)
    repeated = dry_run.advance([stream[1]])
    assert second == repeated
    expected = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
    ).run(stream)
    assert first.snapshots + repeated.snapshots == expected.snapshots
    assert repeated.report == expected.report


def test_failed_incremental_batch_rolls_back_all_prior_events_in_batch():
    import pytest
    from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner

    class FailingPlanner:
        def __init__(self):
            self.calls = 0
            self.delegate = PureHedgePlanner()

        def plan(self, context):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("planned failure")
            return self.delegate.plan(context)

    dry_run = HedgeDryRunWallet(
        initial_balance=Decimal("1000"),
        planner=FailingPlanner(),
    )
    before = dry_run.checkpoint()
    with pytest.raises(RuntimeError, match="planned failure"):
        dry_run.advance(events()[:2])
    after = dry_run.checkpoint()
    assert after.wallet == before.wallet
    assert after.long_state == before.long_state
    assert after.short_state == before.short_state
    assert after.counter == before.counter
    assert after.processed_slots == before.processed_slots


def test_checkpoint_rejects_different_engine_configuration():
    import pytest

    one = HedgeDryRunWallet(initial_balance=Decimal("1000"))
    two = HedgeDryRunWallet(initial_balance=Decimal("2000"))
    checkpoint = one.checkpoint()
    with pytest.raises(ValueError, match="configuration"):
        two.restore(checkpoint)


def test_standard_account_events_cover_fees_and_funding():
    from freqtrade.hedge.simulation.exchange import AccountEvent, AccountEventType

    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal("0")),
    ).run(events())
    account_events = [item for item in result.events if isinstance(item, AccountEvent)]
    assert any(item.event_type is AccountEventType.FEE for item in account_events)
    assert any(item.event_type is AccountEventType.FUNDING for item in account_events)
    assert sum(
        (item.amount for item in account_events if item.event_type is AccountEventType.FEE),
        Decimal("0"),
    ) == -result.report["fees"]


def test_target_progress_and_net_gap_are_reported():
    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0.20"),
        tactical_wallet_exposure_long=Decimal("0"),
        target_net_wallet_exposure=Decimal("0.20"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=1,
        cooldown_seconds=0,
    )
    result = HedgeBacktesting(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
        target_net_quantity=Decimal("2"),
    ).run(
        [
            BarEvent(
                START,
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
            BarEvent(
                START + timedelta(minutes=5),
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
        ]
    )
    assert result.report["target_net_quantity"] == Decimal("2")
    assert result.report["current_net_quantity"] == Decimal("2.0000")
    assert result.report["long_target_progress"] == Decimal("1")
    assert result.report["net_gap_quantity"] == Decimal("0.0000")
    assert result.report["planning_net_gap_quantity"] == Decimal("0.00")


def test_end_to_end_cross_liquidation_stops_future_planning():
    from freqtrade.hedge.simulation.exchange import LiquidationEvent

    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("10"),
        tactical_wallet_exposure_long=Decimal("0"),
        max_wallet_exposure_long=Decimal("20"),
        max_gross_wallet_exposure=Decimal("20"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=1,
        cooldown_seconds=0,
        maintenance_margin_rate=Decimal("0.01"),
        liquidation_fee_rate=Decimal("0.01"),
    )
    stream = [
        BarEvent(
            START,
            "ETH/USDT:USDT",
            Decimal("100"),
            Decimal("100"),
            Decimal("80"),
            Decimal("80"),
            Decimal("1000"),
        ),
        BarEvent(
            START + timedelta(minutes=5),
            "ETH/USDT:USDT",
            Decimal("80"),
            Decimal("90"),
            Decimal("70"),
            Decimal("85"),
            Decimal("1000"),
        ),
    ]
    result = HedgeBacktesting(
        initial_balance=Decimal("100"),
        planner_config=cfg,
        leverage=Decimal("20"),
    ).run(stream)
    liquidations = [item for item in result.events if isinstance(item, LiquidationEvent)]
    assert len(liquidations) == 1
    assert result.report["liquidated"] is True
    assert result.report["liquidation_count"] == 1
    assert result.snapshots[-1].long_quantity == Decimal("0")
    assert result.report["pnl_reconciliation_error"] == Decimal("0")


def test_checkpoint_preserves_target_and_tactical_lot_state():
    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0"),
        tactical_wallet_exposure_long=Decimal("0.20"),
        initial_entry_fraction=Decimal("1"),
        max_grid_layers=1,
        cooldown_seconds=0,
    )
    dry_run = HedgeDryRunWallet(
        initial_balance=Decimal("1000"),
        planner_config=cfg,
        target_net_quantity=Decimal("2"),
    )
    first = dry_run.advance(
        [
            BarEvent(
                START,
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
            BarEvent(
                START + timedelta(minutes=5),
                "ETH/USDT:USDT",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1000"),
            ),
        ]
    )
    checkpoint = dry_run.checkpoint()
    assert checkpoint.target_net_quantity == Decimal("2")
    assert checkpoint.wallet.long.tactical_lots
    next_bar = BarEvent(
        START + timedelta(minutes=10),
        "ETH/USDT:USDT",
        Decimal("100"),
        Decimal("110"),
        Decimal("100"),
        Decimal("110"),
        Decimal("1000"),
    )
    changed = dry_run.advance([next_bar])
    dry_run.restore(checkpoint)
    repeated = dry_run.advance([next_bar])
    assert changed == repeated
    assert first.report["target_net_quantity"] == Decimal("2")
