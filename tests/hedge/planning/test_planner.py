from datetime import datetime, timezone
from decimal import Decimal

from freqtrade.hedge.planning.context import (
    ActiveOrder,
    IntentAction,
    LegPosition,
    MarketSnapshot,
    OrderSide,
    PlannerConfig,
    PlanningContext,
    PositionBucket,
    PositionSide,
    StrategyLegState,
    WalletSnapshot,
)
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner
from freqtrade.hedge.planning.target import calculate_target

NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def wallet(long_qty=Decimal("0"), short_qty=Decimal("0"), active=()):
    return WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("1000"),
        long=LegPosition(PositionSide.LONG, long_qty, Decimal("100") if long_qty else Decimal("0"), long_qty, Decimal("100") if long_qty else Decimal("0"), Decimal("0"), Decimal("0")),
        short=LegPosition(PositionSide.SHORT, short_qty, Decimal("100") if short_qty else Decimal("0"), short_qty, Decimal("100") if short_qty else Decimal("0"), Decimal("0"), Decimal("0")),
        active_orders=active,
        leverage=Decimal("3"),
    )


def context(**kwargs):
    values = {
        "market": MarketSnapshot(
            "ETH/USDT:USDT",
            NOW,
            Decimal("99.9"),
            Decimal("100.1"),
            Decimal("100"),
            Decimal("0.1"),
            Decimal("0.001"),
        ),
        "wallet": wallet(),
        "config": PlannerConfig(cooldown_seconds=0),
        "long_state": StrategyLegState(PositionSide.LONG),
        "short_state": StrategyLegState(PositionSide.SHORT),
        "long_signal": Decimal("1"),
        "short_signal": Decimal("1"),
    }
    unknown = set(kwargs) - set(values)
    if unknown:
        raise TypeError(f"unknown context overrides: {sorted(unknown)}")
    values.update(kwargs)
    return PlanningContext(**values)


def test_same_input_is_deterministic():
    planner = PureHedgePlanner()
    ctx = context()
    assert planner.plan(ctx) == planner.plan(ctx)


def test_long_and_short_have_distinct_intents_and_state():
    result = PureHedgePlanner().plan(context())
    assert {item.position_side for item in result.ideal_orders} == {PositionSide.LONG, PositionSide.SHORT}
    assert result.long_state.side is PositionSide.LONG
    assert result.short_state.side is PositionSide.SHORT


def test_grid_is_bounded_by_max_layers():
    cfg = PlannerConfig(max_grid_layers=2, cooldown_seconds=0)
    result = PureHedgePlanner().plan(context(config=cfg))
    assert len({item.layer for item in result.ideal_orders if not item.reduce_only and item.position_side is PositionSide.LONG}) <= 2
    assert len({item.layer for item in result.ideal_orders if not item.reduce_only and item.position_side is PositionSide.SHORT}) <= 2


def test_projected_gross_never_exceeds_cap():
    cfg = PlannerConfig(max_grid_layers=6, max_gross_wallet_exposure=Decimal("0.30"), max_wallet_exposure_long=Decimal("0.20"), max_wallet_exposure_short=Decimal("0.20"), cooldown_seconds=0)
    ctx = context(config=cfg)
    result = PureHedgePlanner().plan(ctx)
    gross = sum(item.notional for item in result.ideal_orders if not item.reduce_only)
    assert gross <= ctx.wallet.equity * cfg.max_gross_wallet_exposure


def test_unstuck_is_reduce_only_and_reduces_risk():
    cfg = PlannerConfig(unstuck_trigger_gross_exposure=Decimal("0.10"), cooldown_seconds=0)
    result = PureHedgePlanner().plan(context(config=cfg, wallet=wallet(Decimal("2"), Decimal("2"))))
    assert result.ideal_orders
    assert all(item.reduce_only for item in result.ideal_orders)
    assert all(item.action is IntentAction.UNSTUCK for item in result.ideal_orders)


def test_target_core_and_tactical_are_capped():
    ctx = context()
    target = calculate_target(ctx, PositionSide.LONG)
    assert target.total_quantity <= target.maximum_quantity


def test_jitter_protection_keeps_nearby_active_order():
    first = PureHedgePlanner().plan(context())
    desired = first.ideal_orders[0]
    active = ActiveOrder(
        order_id="existing",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=desired.quantity,
        price=desired.price + Decimal("0.1"),
        reduce_only=desired.reduce_only,
        bucket=desired.bucket,
        action=desired.action,
        created_at=NOW,
    )
    result = PureHedgePlanner().plan(context(wallet=wallet(active=(active,))))
    assert "existing" not in result.cancel_order_ids
    assert "existing" in result.kept_order_ids
    assert all(item.intent_id != desired.intent_id for item in result.submit_orders)
    assert any(item.position_side is desired.position_side and item.layer == desired.layer for item in result.ideal_orders)


def test_flat_leg_has_explicit_initial_entry():
    result = PureHedgePlanner().plan(context())
    initial = [item for item in result.ideal_orders if item.reason == "initial_entry"]
    assert {item.position_side for item in initial} == {PositionSide.LONG, PositionSide.SHORT}


def test_core_and_tactical_allocations_are_separate():
    cfg = PlannerConfig(core_wallet_exposure_long=Decimal("0.05"), tactical_wallet_exposure_long=Decimal("0.20"), max_wallet_exposure_long=Decimal("0.30"), cooldown_seconds=0)
    result = PureHedgePlanner().plan(context(config=cfg))
    long_entries = [item for item in result.ideal_orders if item.position_side is PositionSide.LONG and not item.reduce_only]
    assert {item.bucket for item in long_entries} == {PositionBucket.CORE, PositionBucket.TACTICAL}


def test_disabled_short_side_produces_no_short_orders():
    result = PureHedgePlanner().plan(context(config=PlannerConfig(short_enabled=False, cooldown_seconds=0)))
    assert all(item.position_side is PositionSide.LONG for item in result.ideal_orders)
    assert "SHORT:disabled" in result.diagnostics


def test_entry_cooldown_blocks_reentry_plans():
    state = StrategyLegState(PositionSide.LONG, last_entry_at=NOW)
    cfg = PlannerConfig(cooldown_seconds=300)
    result = PureHedgePlanner().plan(context(long_state=state, config=cfg, wallet=wallet(Decimal("0.2"), Decimal("0"))))
    assert all(item.reduce_only for item in result.ideal_orders if item.position_side is PositionSide.LONG)
    assert "LONG:entry_cooldown" in result.diagnostics


def test_close_grid_preserves_core_floor():
    cfg = PlannerConfig(core_wallet_exposure_long=Decimal("0.20"), tactical_wallet_exposure_long=Decimal("0.10"), core_min_fraction=Decimal("0.75"), cooldown_seconds=0)
    long_leg = LegPosition(
        PositionSide.LONG,
        Decimal("3"), Decimal("100"),
        Decimal("2"), Decimal("100"),
        Decimal("1"), Decimal("100"),
    )
    custom_wallet = WalletSnapshot(
        balance=Decimal("1000"), equity=Decimal("1000"), available_balance=Decimal("800"),
        long=long_leg, short=LegPosition(PositionSide.SHORT), leverage=Decimal("3"),
    )
    result = PureHedgePlanner().plan(context(config=cfg, wallet=custom_wallet))
    reductions = sum((item.quantity for item in result.ideal_orders if item.position_side is PositionSide.LONG and item.reduce_only), Decimal("0"))
    target_core = calculate_target(context(config=cfg, wallet=custom_wallet), PositionSide.LONG).core_quantity
    assert long_leg.quantity - reductions >= target_core * cfg.core_min_fraction


def test_planning_does_not_advance_fill_timestamps():
    result = PureHedgePlanner().plan(context())
    assert result.long_state.last_entry_at is None
    assert result.short_state.last_entry_at is None


def test_jitter_protection_compares_order_type_and_time_in_force():
    from freqtrade.hedge.planning.context import OrderType, TimeInForce

    cfg = PlannerConfig(
        unstuck_trigger_gross_exposure=Decimal("0.10"),
        cooldown_seconds=0,
    )
    base_context = context(config=cfg, wallet=wallet(Decimal("2"), Decimal("0")))
    desired = next(
        item
        for item in PureHedgePlanner().plan(base_context).ideal_orders
        if item.position_side is PositionSide.LONG
    )
    stale = ActiveOrder(
        order_id="stale-limit",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=desired.quantity,
        price=desired.price,
        reduce_only=desired.reduce_only,
        bucket=desired.bucket,
        action=desired.action,
        created_at=NOW,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
    )
    result = PureHedgePlanner().plan(
        context(config=cfg, wallet=wallet(Decimal("2"), Decimal("0"), (stale,)))
    )
    assert "stale-limit" in result.cancel_order_ids
    assert any(item.order_type is OrderType.MARKET for item in result.submit_orders)


def test_foreign_symbol_order_is_not_cancelled_by_symbol_planner():
    foreign = ActiveOrder(
        order_id="btc-order",
        symbol="BTC/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        price=Decimal("50000"),
        reduce_only=False,
        bucket=PositionBucket.CORE,
        action=IntentAction.OPEN,
        created_at=NOW,
    )
    result = PureHedgePlanner().plan(context(wallet=wallet(active=(foreign,))))
    assert "btc-order" not in result.cancel_order_ids
    assert "btc-order" in result.kept_order_ids


def test_margin_budget_blocks_new_entry_orders():
    blocked_wallet = WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("0"),
        long=LegPosition(PositionSide.LONG),
        short=LegPosition(PositionSide.SHORT),
        leverage=Decimal("3"),
    )
    result = PureHedgePlanner().plan(context(wallet=blocked_wallet))
    assert not [item for item in result.ideal_orders if not item.reduce_only]
    assert any("exposure_or_margin" in item for item in result.diagnostics)


def test_invalid_initial_min_notional_is_reallocated_to_grid():
    cfg = PlannerConfig(
        core_wallet_exposure_long=Decimal("0"),
        tactical_wallet_exposure_long=Decimal("0.30"),
        long_enabled=True,
        short_enabled=False,
        initial_entry_fraction=Decimal("0.05"),
        max_grid_layers=2,
        cooldown_seconds=0,
    )
    ctx = PlanningContext(
        market=MarketSnapshot(
            "ETH/USDT:USDT",
            NOW,
            Decimal("99.9"),
            Decimal("100.1"),
            Decimal("100"),
            Decimal("0.1"),
            Decimal("0.001"),
            Decimal("0.001"),
            Decimal("50"),
        ),
        wallet=wallet(),
        config=cfg,
        long_signal=Decimal("1"),
        short_signal=Decimal("0"),
    )
    result = PureHedgePlanner().plan(ctx)
    long_entries = [
        item
        for item in result.ideal_orders
        if item.position_side is PositionSide.LONG and not item.reduce_only
    ]
    assert long_entries
    assert all(item.notional >= Decimal("50") for item in long_entries)
    assert sum((item.quantity for item in long_entries), Decimal("0")) >= Decimal("2.99")


def test_close_grid_uses_partial_reduce_fraction():
    cfg = PlannerConfig(
        core_wallet_exposure_long=Decimal("0"),
        tactical_wallet_exposure_long=Decimal("0"),
        tactical_reduce_fraction=Decimal("0.25"),
        core_min_fraction=Decimal("0"),
        take_profit_layers=2,
        cooldown_seconds=0,
    )
    long_leg = LegPosition(
        PositionSide.LONG,
        Decimal("4"),
        Decimal("100"),
        Decimal("0"),
        Decimal("0"),
        Decimal("4"),
        Decimal("100"),
    )
    custom_wallet = WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("800"),
        long=long_leg,
        short=LegPosition(PositionSide.SHORT),
        leverage=Decimal("3"),
    )
    result = PureHedgePlanner().plan(context(config=cfg, wallet=custom_wallet))
    reductions = sum(
        (
            item.quantity
            for item in result.ideal_orders
            if item.position_side is PositionSide.LONG and item.reduce_only
        ),
        Decimal("0"),
    )
    assert reductions == Decimal("1")


def test_semantically_equal_decimal_values_share_intent_id():
    from freqtrade.hedge.planning.context import OrderIntent

    one = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("1.0"),
        price=Decimal("100.00"),
        reduce_only=False,
    )
    two = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("1.00"),
        price=Decimal("100.0"),
        reduce_only=False,
    )
    assert one.intent_id == two.intent_id


def test_active_order_rejects_inconsistent_side_and_reduce_only():
    import pytest

    from freqtrade.hedge.planning.context import (
        ActiveOrder,
        IntentAction,
        OrderSide,
        PositionBucket,
        PositionSide,
    )

    with pytest.raises(ValueError, match="active order side"):
        ActiveOrder(
            order_id="bad-side",
            symbol="BTC/USDT:USDT",
            position_side=PositionSide.LONG,
            order_side=OrderSide.BUY,
            quantity=Decimal("1"),
            price=Decimal("100"),
            reduce_only=True,
            bucket=PositionBucket.TACTICAL,
            action=IntentAction.REDUCE,
            created_at=NOW,
        )


def test_trailing_rebound_must_be_strictly_below_one():
    import pytest

    with pytest.raises(ValueError, match="less than one"):
        PlannerConfig(trailing_rebound=Decimal("1"))


def test_active_entry_beyond_max_layer_is_cancelled():
    active = ActiveOrder(
        order_id="too-deep",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        price=Decimal("90"),
        reduce_only=False,
        bucket=PositionBucket.CORE,
        action=IntentAction.INCREASE,
        created_at=NOW,
        client_order_id="old-intent",
        layer=5,
    )
    ctx = context(
        wallet=WalletSnapshot(
            balance=Decimal("1000"),
            equity=Decimal("1000"),
            available_balance=Decimal("1000"),
            long=LegPosition(PositionSide.LONG),
            short=LegPosition(PositionSide.SHORT),
            active_orders=(active,),
            leverage=Decimal("3"),
        ),
        config=PlannerConfig(max_grid_layers=2, cooldown_seconds=0),
    )
    result = PureHedgePlanner().plan(ctx)
    assert "too-deep" in result.cancel_order_ids
    assert all(order.layer <= 2 for order in result.ideal_orders)


def test_unstuck_respects_minimum_notional():
    long = LegPosition(
        side=PositionSide.LONG,
        quantity=Decimal("1"),
        average_price=Decimal("100"),
        core_quantity=Decimal("0"),
        core_average_price=Decimal("0"),
        tactical_quantity=Decimal("1"),
        tactical_average_price=Decimal("100"),
    )
    ctx = PlanningContext(
        market=MarketSnapshot(
            symbol="BTC/USDT:USDT",
            timestamp=NOW,
            bid=Decimal("100"),
            ask=Decimal("100.01"),
            mark=Decimal("100"),
            min_notional=Decimal("1000"),
        ),
        wallet=WalletSnapshot(
            balance=Decimal("10"),
            equity=Decimal("10"),
            available_balance=Decimal("0"),
            long=long,
            short=LegPosition(PositionSide.SHORT),
            leverage=Decimal("3"),
        ),
        config=PlannerConfig(
            short_enabled=False,
            unstuck_trigger_gross_exposure=Decimal("0.1"),
            unstuck_reduce_fraction=Decimal("0.5"),
            cooldown_seconds=0,
        ),
        long_signal=Decimal("1"),
        short_signal=Decimal("0"),
    )
    result = PureHedgePlanner().plan(ctx)
    assert all(order.action is not IntentAction.UNSTUCK for order in result.ideal_orders)


def test_tactical_take_profit_uses_tactical_cost_basis():
    long_leg = LegPosition(
        side=PositionSide.LONG,
        quantity=Decimal("10"),
        average_price=Decimal("88"),
        core_quantity=Decimal("8"),
        core_average_price=Decimal("80"),
        tactical_quantity=Decimal("2"),
        tactical_average_price=Decimal("120"),
    )
    custom_wallet = WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("1000"),
        long=long_leg,
        short=LegPosition(PositionSide.SHORT),
        leverage=Decimal("3"),
    )
    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0.8"),
        tactical_wallet_exposure_long=Decimal("0.2"),
        max_wallet_exposure_long=Decimal("1"),
        max_gross_wallet_exposure=Decimal("1"),
        take_profit_layers=1,
        tactical_reduce_fraction=Decimal("1"),
        unstuck_trigger_gross_exposure=Decimal("2"),
        cooldown_seconds=0,
    )
    result = PureHedgePlanner().plan(context(config=cfg, wallet=custom_wallet))
    tactical_closes = [
        order
        for order in result.ideal_orders
        if order.reduce_only and order.bucket is PositionBucket.TACTICAL
    ]
    assert tactical_closes
    assert all(order.price > long_leg.tactical_average_price for order in tactical_closes)


def test_replacement_is_debounced_for_young_semantically_matching_order():
    first = PureHedgePlanner().plan(context())
    desired = first.ideal_orders[0]
    cfg = PlannerConfig(
        cooldown_seconds=0,
        replace_price_tolerance_ticks=0,
        replace_qty_tolerance_steps=0,
        replace_min_age_seconds=30,
    )
    active = ActiveOrder(
        order_id="young-order",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=desired.quantity,
        price=desired.price + Decimal("5"),
        reduce_only=desired.reduce_only,
        bucket=desired.bucket,
        action=desired.action,
        created_at=NOW,
        layer=desired.layer,
    )
    result = PureHedgePlanner().plan(
        context(config=cfg, wallet=wallet(active=(active,)))
    )
    assert "young-order" in result.kept_order_ids
    assert "young-order" not in result.cancel_order_ids
    assert any("replacement_debounced" in item for item in result.diagnostics)


def test_old_semantically_matching_order_is_replaced_after_debounce_window():
    from datetime import timedelta

    first = PureHedgePlanner().plan(context())
    desired = first.ideal_orders[0]
    cfg = PlannerConfig(
        cooldown_seconds=0,
        replace_price_tolerance_ticks=0,
        replace_qty_tolerance_steps=0,
        replace_min_age_seconds=30,
    )
    active = ActiveOrder(
        order_id="old-order",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=desired.quantity,
        price=desired.price + Decimal("5"),
        reduce_only=desired.reduce_only,
        bucket=desired.bucket,
        action=desired.action,
        created_at=NOW - timedelta(seconds=31),
        layer=desired.layer,
    )
    result = PureHedgePlanner().plan(
        context(config=cfg, wallet=wallet(active=(active,)))
    )
    assert "old-order" in result.cancel_order_ids
    assert "old-order" not in result.kept_order_ids
    assert any(
        item.position_side is desired.position_side and item.layer == desired.layer
        for item in result.submit_orders
    )


def test_explicit_net_target_exposes_gap_and_suppresses_opposing_tactical_entries():
    cfg = PlannerConfig(
        core_wallet_exposure_long=Decimal("0"),
        core_wallet_exposure_short=Decimal("0"),
        tactical_wallet_exposure_long=Decimal("0.20"),
        tactical_wallet_exposure_short=Decimal("0.20"),
        target_net_wallet_exposure=Decimal("0.10"),
        net_repair_threshold=Decimal("0.01"),
        cooldown_seconds=0,
    )
    ctx = context(config=cfg)
    result = PureHedgePlanner().plan(ctx)
    assert result.target_net_quantity == Decimal("1.000")
    assert result.net_gap_quantity == Decimal("1.000")
    assert result.long_target_quantity > Decimal("0")
    assert result.short_target_quantity == Decimal("0")
    assert all(
        item.position_side is not PositionSide.SHORT or item.reduce_only
        for item in result.ideal_orders
    )


def test_trailing_state_machine_arms_confirms_and_enters_cooldown():
    from dataclasses import replace
    from datetime import timedelta

    from freqtrade.hedge.planning.context import TrailingPhase
    from freqtrade.hedge.planning.trailing import (
        enter_trailing_cooldown,
        update_trailing_state,
    )

    long_leg = LegPosition(
        PositionSide.LONG,
        Decimal("1"),
        Decimal("100"),
        Decimal("0"),
        Decimal("0"),
        Decimal("1"),
        Decimal("100"),
    )
    custom_wallet = WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("900"),
        long=long_leg,
        short=LegPosition(PositionSide.SHORT),
        leverage=Decimal("3"),
    )
    cfg = PlannerConfig(
        trailing_trigger_distance=Decimal("0.01"),
        trailing_rebound=Decimal("0.005"),
        cooldown_seconds=60,
    )
    armed_context = context(
        config=cfg,
        wallet=custom_wallet,
        market=replace(context().market, mark=Decimal("98.5")),
    )
    armed = update_trailing_state(armed_context, PositionSide.LONG)
    assert armed.trailing_phase is TrailingPhase.ARMED
    confirmed_context = PlanningContext(
        market=replace(
            armed_context.market,
            timestamp=NOW + timedelta(seconds=10),
            mark=Decimal("99.1"),
        ),
        wallet=custom_wallet,
        config=cfg,
        long_state=armed,
        short_state=armed_context.short_state,
        long_signal=Decimal("1"),
        short_signal=Decimal("0"),
    )
    confirmed = update_trailing_state(confirmed_context, PositionSide.LONG)
    assert confirmed.trailing_phase is TrailingPhase.CONFIRMED
    assert confirmed.trailing_armed is True
    cooldown = enter_trailing_cooldown(
        confirmed,
        timestamp=confirmed_context.market.timestamp,
        cooldown_seconds=60,
    )
    assert cooldown.trailing_phase is TrailingPhase.COOLDOWN
    assert cooldown.trailing_armed is False


def test_single_order_notional_and_pending_order_limits_are_enforced():
    cfg = PlannerConfig(
        short_enabled=False,
        core_wallet_exposure_long=Decimal("0.20"),
        tactical_wallet_exposure_long=Decimal("0.20"),
        max_wallet_exposure_long=Decimal("0.50"),
        max_single_order_notional=Decimal("50"),
        max_pending_entries=2,
        max_grid_layers=6,
        cooldown_seconds=0,
    )
    result = PureHedgePlanner().plan(context(config=cfg))
    entries = [item for item in result.ideal_orders if not item.reduce_only]
    assert len(entries) <= 2
    assert all(item.notional <= Decimal("50") for item in entries)


def test_tactical_lot_take_profit_is_bound_to_its_batch():
    from datetime import timedelta

    from freqtrade.hedge.planning.context import TacticalLot

    lots = (
        TacticalLot("lot-a", Decimal("1"), Decimal("90"), NOW - timedelta(hours=2), 1),
        TacticalLot("lot-b", Decimal("1"), Decimal("110"), NOW - timedelta(hours=1), 2),
    )
    long_leg = LegPosition(
        PositionSide.LONG,
        Decimal("2"),
        Decimal("100"),
        Decimal("0"),
        Decimal("0"),
        Decimal("2"),
        Decimal("100"),
        tactical_lots=lots,
    )
    custom_wallet = WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("900"),
        long=long_leg,
        short=LegPosition(PositionSide.SHORT),
        leverage=Decimal("3"),
    )
    cfg = PlannerConfig(
        core_wallet_exposure_long=Decimal("0"),
        tactical_wallet_exposure_long=Decimal("0"),
        tactical_reduce_fraction=Decimal("0.5"),
        take_profit_layers=2,
        cooldown_seconds=0,
    )
    result = PureHedgePlanner().plan(context(config=cfg, wallet=custom_wallet))
    lot_orders = [
        item
        for item in result.ideal_orders
        if item.reduce_only and item.position_side is PositionSide.LONG
    ]
    assert {item.tactical_lot_id for item in lot_orders} == {"lot-a", "lot-b"}
    prices = {item.tactical_lot_id: item.price for item in lot_orders}
    assert prices["lot-a"] < prices["lot-b"]


def test_unstuck_budget_exhaustion_blocks_further_loss_realization():
    state = StrategyLegState(
        PositionSide.LONG,
        unstuck_budget_day=NOW.date().isoformat(),
        unstuck_budget_week=f"{NOW.isocalendar().year}-W{NOW.isocalendar().week:02d}",
        unstuck_daily_loss=Decimal("10"),
        unstuck_weekly_loss=Decimal("10"),
    )
    cfg = PlannerConfig(
        unstuck_trigger_gross_exposure=Decimal("0.01"),
        unstuck_daily_loss_budget=Decimal("0.01"),
        unstuck_weekly_loss_budget=Decimal("0.02"),
        cooldown_seconds=0,
    )
    result = PureHedgePlanner().plan(
        context(
            config=cfg,
            long_state=state,
            wallet=wallet(Decimal("2"), Decimal("0")),
        )
    )
    assert not [item for item in result.ideal_orders if item.action is IntentAction.UNSTUCK]


def test_order_diff_reports_modify_delete_and_risk_cancel_classes():
    first = PureHedgePlanner().plan(context())
    desired = next(item for item in first.ideal_orders if not item.reduce_only)
    stale = ActiveOrder(
        order_id="modify-me",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=desired.quantity,
        price=desired.price + Decimal("10"),
        reduce_only=False,
        bucket=desired.bucket,
        action=desired.action,
        created_at=NOW.replace(year=2025),
        layer=desired.layer,
        tactical_lot_id=desired.tactical_lot_id,
    )
    orphan = ActiveOrder(
        order_id="delete-me",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=Decimal("0.001"),
        price=Decimal("10"),
        reduce_only=False,
        bucket=PositionBucket.CORE,
        action=IntentAction.INCREASE,
        created_at=NOW.replace(year=2025),
        layer=99,
    )
    result = PureHedgePlanner().plan(context(wallet=wallet(active=(stale, orphan))))
    assert "modify-me" in result.modify_order_ids
    assert "delete-me" in result.delete_order_ids
    assert set(result.cancel_order_ids) == set(
        result.modify_order_ids + result.delete_order_ids + result.risk_cancel_order_ids
    )


def test_legacy_trailing_armed_state_is_upgraded_to_confirmed_phase():
    from freqtrade.hedge.planning.context import TrailingPhase

    state = StrategyLegState(PositionSide.LONG, trailing_armed=True)
    assert state.trailing_phase is TrailingPhase.CONFIRMED


def test_zero_pending_entry_limit_cancels_existing_entries_and_submits_none():
    first = PureHedgePlanner().plan(context())
    desired = next(item for item in first.ideal_orders if not item.reduce_only)
    active = ActiveOrder(
        order_id="pending",
        symbol=desired.symbol,
        position_side=desired.position_side,
        order_side=desired.order_side,
        quantity=desired.quantity,
        price=desired.price,
        reduce_only=False,
        bucket=desired.bucket,
        action=desired.action,
        created_at=NOW,
        layer=desired.layer,
        tactical_lot_id=desired.tactical_lot_id,
    )
    result = PureHedgePlanner().plan(
        context(
            config=PlannerConfig(max_pending_entries=0, cooldown_seconds=0),
            wallet=wallet(active=(active,)),
        )
    )
    assert not [item for item in result.ideal_orders if not item.reduce_only]
    assert "pending" in result.cancel_order_ids
