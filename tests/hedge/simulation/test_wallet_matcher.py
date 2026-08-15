from datetime import datetime, timezone
from decimal import Decimal

from freqtrade.hedge.planning.context import (
    IntentAction,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionBucket,
    PositionSide,
    TimeInForce,
)
from freqtrade.hedge.simulation.cross_wallet import CrossWallet
from freqtrade.hedge.simulation.exchange import BarEvent, FillEvent, FundingEvent
from freqtrade.hedge.simulation.funding import FundingEngine
from freqtrade.hedge.simulation.matcher import ConservativeMatcher, MatchConfig

NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def intent(side, order_side, qty, price, reduce=False, bucket=PositionBucket.CORE, action=IntentAction.OPEN):
    return OrderIntent.deterministic(
        symbol="ETH/USDT:USDT", position_side=side, order_side=order_side, action=action,
        bucket=bucket, quantity=Decimal(qty), price=Decimal(price), reduce_only=reduce,
    )


def fill(order_id, item, qty, price):
    return FillEvent(
        event_id=f"f-{order_id}", timestamp=NOW, order_id=order_id, intent_id=item.intent_id,
        symbol=item.symbol, position_side=item.position_side, quantity=Decimal(qty), price=Decimal(price),
        fee=Decimal("0"), reduce_only=item.reduce_only, bucket=item.bucket, action=item.action.value, layer=item.layer,
    )


def test_long_short_average_prices_are_independent():
    wallet = CrossWallet(Decimal("1000"))
    long = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    short = intent(PositionSide.SHORT, OrderSide.SELL, "2", "110")
    wallet.accept_order("l", long); wallet.apply_fill(fill("l", long, "1", "100"))
    wallet.accept_order("s", short); wallet.apply_fill(fill("s", short, "2", "110"))
    assert wallet.long.average_price == Decimal("100")
    assert wallet.short.average_price == Decimal("110")
    assert wallet.long.quantity == Decimal("1") and wallet.short.quantity == Decimal("2")


def test_partial_fill_leaves_remaining_order():
    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "5", "100")
    wallet.accept_order("o", item)
    wallet.apply_fill(fill("o", item, "2", "100"))
    assert wallet.remaining("o") == Decimal("3")
    assert wallet.long.quantity == Decimal("2")


def test_reduce_only_cannot_flip_position():
    wallet = CrossWallet(Decimal("1000"))
    add = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("a", add); wallet.apply_fill(fill("a", add, "1", "100"))
    close = intent(PositionSide.LONG, OrderSide.SELL, "2", "110", True, PositionBucket.CORE, IntentAction.CLOSE)
    wallet.accept_order("c", close)
    bar = BarEvent(NOW, close.symbol, Decimal("100"), Decimal("120"), Decimal("90"), Decimal("110"), Decimal("100"))
    fills = ConservativeMatcher().match(bar, wallet)
    assert fills[0].quantity == Decimal("1")


def test_funding_offsets_equal_hedged_notional():
    wallet = CrossWallet(Decimal("1000"))
    long = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    short = intent(PositionSide.SHORT, OrderSide.SELL, "1", "100")
    wallet.accept_order("l", long); wallet.apply_fill(fill("l", long, "1", "100"))
    wallet.accept_order("s", short); wallet.apply_fill(fill("s", short, "1", "100"))
    amount = FundingEngine().apply(wallet, FundingEvent(NOW, long.symbol, Decimal("0.001"), Decimal("100")))
    assert amount == Decimal("0")


def test_active_order_margin_uses_cross_leverage():
    wallet = CrossWallet(Decimal("1000"), leverage=Decimal("5"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "2", "100")
    wallet.accept_order("o", item)
    assert wallet.active_order_margin() == Decimal("40")


def test_conservative_same_bar_path_avoids_optimistic_round_trip():
    wallet = CrossWallet(Decimal("1000"))
    entry = intent(PositionSide.LONG, OrderSide.BUY, "1", "95")
    exit_order = intent(PositionSide.LONG, OrderSide.SELL, "1", "105", True, PositionBucket.TACTICAL, IntentAction.REDUCE)
    wallet.accept_order("entry", entry)
    wallet.accept_order("exit", exit_order)
    bar = BarEvent(NOW, entry.symbol, Decimal("100"), Decimal("110"), Decimal("90"), Decimal("100"), Decimal("100"))
    fills = ConservativeMatcher().match(bar, wallet)
    # Because there is no position at bar open, the conservative path does not assume
    # a same-candle entry followed by a profitable reduce-only exit.
    assert [item.order_id for item in fills] == ["entry"]


def test_fees_reduce_cross_wallet_balance():
    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("o", item)
    fee_fill = FillEvent(
        event_id="fee", timestamp=NOW, order_id="o", intent_id=item.intent_id,
        symbol=item.symbol, position_side=item.position_side, quantity=Decimal("1"), price=Decimal("100"),
        fee=Decimal("0.04"), reduce_only=False, bucket=item.bucket, action=item.action.value, layer=item.layer,
    )
    wallet.apply_fill(fee_fill)
    assert wallet.balance == Decimal("999.96")
    assert wallet.fees_paid == Decimal("0.04")


def test_marketable_limit_fills_at_open_with_price_improvement():
    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "105")
    wallet.accept_order("marketable", item)
    bar = BarEvent(
        NOW,
        item.symbol,
        Decimal("100"),
        Decimal("101"),
        Decimal("99"),
        Decimal("100"),
        Decimal("100"),
    )
    fills = ConservativeMatcher().match(bar, wallet)
    assert fills[0].price == Decimal("100")


def test_crossing_order_is_price_chronological_not_order_id_sorted():
    wallet = CrossWallet(Decimal("1000"))
    lower = intent(PositionSide.LONG, OrderSide.BUY, "1", "95")
    higher = intent(PositionSide.LONG, OrderSide.BUY, "1", "99")
    wallet.accept_order("a-lower", lower)
    wallet.accept_order("z-higher", higher)
    # volume participation is 10%, so only one unit can fill.
    bar = BarEvent(
        NOW,
        lower.symbol,
        Decimal("100"),
        Decimal("100"),
        Decimal("90"),
        Decimal("90"),
        Decimal("10"),
    )
    fills = ConservativeMatcher().match(bar, wallet)
    assert [item.order_id for item in fills] == ["z-higher"]


def test_multiple_reduce_only_orders_cannot_overfill_same_position():
    wallet = CrossWallet(Decimal("1000"))
    add = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("add", add)
    wallet.apply_fill(fill("add", add, "1", "100"))
    first = intent(
        PositionSide.LONG,
        OrderSide.SELL,
        "1",
        "105",
        True,
        PositionBucket.CORE,
        IntentAction.REDUCE,
    )
    second = intent(
        PositionSide.LONG,
        OrderSide.SELL,
        "1",
        "110",
        True,
        PositionBucket.CORE,
        IntentAction.REDUCE,
    )
    wallet.accept_order("first", first)
    wallet.accept_order("second", second)
    bar = BarEvent(
        NOW,
        add.symbol,
        Decimal("100"),
        Decimal("120"),
        Decimal("100"),
        Decimal("120"),
        Decimal("100"),
    )
    fills = ConservativeMatcher().match(bar, wallet)
    assert sum((item.quantity for item in fills), Decimal("0")) == Decimal("1")
    assert len(fills) == 1


def test_ioc_partial_fill_expires_remainder():
    from freqtrade.hedge.planning.context import OrderType, TimeInForce

    wallet = CrossWallet(Decimal("1000"))
    add = intent(PositionSide.LONG, OrderSide.BUY, "2", "100")
    wallet.accept_order("add", add)
    wallet.apply_fill(fill("add", add, "2", "100"))
    close = OrderIntent.deterministic(
        symbol=add.symbol,
        position_side=PositionSide.LONG,
        order_side=OrderSide.SELL,
        action=IntentAction.UNSTUCK,
        bucket=PositionBucket.CORE,
        quantity=Decimal("2"),
        price=Decimal("100"),
        reduce_only=True,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
    )
    wallet.accept_order("ioc", close)
    bar = BarEvent(
        NOW,
        add.symbol,
        Decimal("100"),
        Decimal("100"),
        Decimal("100"),
        Decimal("100"),
        Decimal("5"),
    )
    outcome = ConservativeMatcher().match_outcome(bar, wallet)
    assert outcome.fills[0].quantity == Decimal("0.5")
    assert outcome.expired_order_ids == ("ioc",)


def test_duplicate_fill_event_is_idempotent():
    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("o", item)
    event = fill("o", item, "1", "100")
    assert wallet.apply_fill(event) is True
    assert wallet.apply_fill(event) is False
    assert wallet.long.quantity == Decimal("1")


def test_fill_cannot_exceed_active_order_remaining():
    import pytest

    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("o", item)
    with pytest.raises(ValueError, match="remaining"):
        wallet.apply_fill(fill("o", item, "2", "100"))


def test_dual_leg_duration_starts_when_second_leg_opens():
    from datetime import timedelta

    wallet = CrossWallet(Decimal("1000"))
    long = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    short = intent(PositionSide.SHORT, OrderSide.SELL, "1", "100")
    wallet.accept_order("l", long)
    wallet.apply_fill(fill("l", long, "1", "100"))
    wallet.snapshot(NOW, Decimal("100"))
    later = NOW + timedelta(minutes=5)
    short_fill = FillEvent(
        event_id="short-fill",
        timestamp=later,
        order_id="s",
        intent_id=short.intent_id,
        symbol=short.symbol,
        position_side=short.position_side,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        reduce_only=False,
        bucket=short.bucket,
        action=short.action,
        layer=short.layer,
    )
    wallet.accept_order("s", short)
    wallet.apply_fill(short_fill)
    wallet.snapshot(later, Decimal("100"))
    wallet.snapshot(later + timedelta(minutes=5), Decimal("100"))
    assert wallet.hedge_duration_seconds == Decimal("300.0")


def test_all_ioc_orders_expire_when_volume_is_consumed():
    wallet = CrossWallet(initial_balance=Decimal("1000"))
    first = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("1"),
        price=Decimal("100"),
        reduce_only=False,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
    )
    second = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("1"),
        price=Decimal("100"),
        reduce_only=False,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
    )
    wallet.accept_order("order-first", first)
    wallet.accept_order("order-second", second)
    matcher = ConservativeMatcher(
        MatchConfig(volume_participation=Decimal("0.1"))
    )
    outcome = matcher.match_outcome(
        BarEvent(
            NOW,
            "ETH/USDT:USDT",
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("10"),
        ),
        wallet,
    )
    assert sum(
        (item.quantity for item in outcome.fills),
        Decimal("0"),
    ) == Decimal("1")
    assert len(outcome.expired_order_ids) == 1


def test_tactical_fee_is_allocated_by_actual_reduced_quantity():
    wallet = CrossWallet(initial_balance=Decimal("1000"))
    core = intent(
        PositionSide.LONG,
        OrderSide.BUY,
        "8",
        "100",
        bucket=PositionBucket.CORE,
    )
    tactical = intent(
        PositionSide.LONG,
        OrderSide.BUY,
        "2",
        "100",
        bucket=PositionBucket.TACTICAL,
        action=IntentAction.INCREASE,
    )
    wallet.accept_order("core-open-order", core)
    wallet.apply_fill(fill("core-open-order", core, "8", "100"))
    wallet.accept_order("tactical-open-order", tactical)
    wallet.apply_fill(fill("tactical-open-order", tactical, "2", "100"))

    reduction_intent = intent(
        PositionSide.LONG,
        OrderSide.SELL,
        "4",
        "110",
        True,
        PositionBucket.TACTICAL,
        IntentAction.UNSTUCK,
    )
    wallet.accept_order("mixed-reduce-order", reduction_intent)
    reduction = FillEvent(
        event_id="mixed-reduce-fill",
        timestamp=NOW,
        order_id="mixed-reduce-order",
        intent_id=reduction_intent.intent_id,
        symbol=reduction_intent.symbol,
        position_side=reduction_intent.position_side,
        quantity=Decimal("4"),
        price=Decimal("110"),
        fee=Decimal("4.4"),
        reduce_only=True,
        bucket=PositionBucket.TACTICAL,
        action=IntentAction.UNSTUCK,
    )
    wallet.apply_fill(reduction)
    # Two of four reduced units were tactical, so only half the fee is tactical.
    assert wallet.tactical_fees_paid == Decimal("2.2")



def test_zero_volume_bar_has_no_liquidity():
    wallet = CrossWallet(initial_balance=Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("zero-volume", item)
    outcome = ConservativeMatcher().match_outcome(
        BarEvent(
            NOW,
            item.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("0"),
        ),
        wallet,
    )
    assert outcome.fills == ()
    assert wallet.remaining("zero-volume") == Decimal("1")


def test_missing_volume_uses_unbounded_liquidity_mode():
    wallet = CrossWallet(initial_balance=Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("unknown-volume", item)
    outcome = ConservativeMatcher().match_outcome(
        BarEvent(
            NOW,
            item.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
        ),
        wallet,
    )
    assert [fill.quantity for fill in outcome.fills] == [Decimal("1")]


def test_resting_limit_is_maker_and_same_bar_marketable_limit_is_taker():
    from datetime import timedelta
    from freqtrade.hedge.simulation.exchange import LiquidityRole

    config = MatchConfig(
        maker_fee_rate=Decimal("0.0001"),
        taker_fee_rate=Decimal("0.001"),
    )
    matcher = ConservativeMatcher(config)

    resting_wallet = CrossWallet(Decimal("1000"))
    resting = intent(PositionSide.LONG, OrderSide.BUY, "1", "95")
    resting_wallet.accept_order("resting", resting, NOW - timedelta(minutes=1))
    resting_fill = matcher.match(
        BarEvent(
            NOW,
            resting.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("90"),
            Decimal("95"),
            Decimal("100"),
        ),
        resting_wallet,
    )[0]
    assert resting_fill.liquidity_role is LiquidityRole.MAKER
    assert resting_fill.fee == Decimal("0.0095")

    taker_wallet = CrossWallet(Decimal("1000"))
    marketable = intent(PositionSide.LONG, OrderSide.BUY, "1", "105")
    taker_wallet.accept_order("taker", marketable, NOW)
    taker_fill = matcher.match(
        BarEvent(
            NOW,
            marketable.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
        ),
        taker_wallet,
    )[0]
    assert taker_fill.liquidity_role is LiquidityRole.TAKER
    assert taker_fill.fee == Decimal("0.100")


def test_market_order_applies_directional_slippage_and_price_tick():
    buy_wallet = CrossWallet(Decimal("1000"))
    buy = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("1"),
        price=Decimal("100"),
        reduce_only=False,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
    )
    buy_wallet.accept_order("buy", buy, NOW)
    matcher = ConservativeMatcher(
        MatchConfig(
            fee_rate=Decimal("0"),
            market_slippage_bps=Decimal("10"),
            price_tick=Decimal("0.01"),
        )
    )
    buy_fill = matcher.match(
        BarEvent(
            NOW,
            buy.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
        ),
        buy_wallet,
    )[0]
    assert buy_fill.price == Decimal("100.10")

    sell_wallet = CrossWallet(Decimal("1000"))
    sell = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.SHORT,
        order_side=OrderSide.SELL,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("1"),
        price=Decimal("100"),
        reduce_only=False,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
    )
    sell_wallet.accept_order("sell", sell, NOW)
    sell_fill = matcher.match(
        BarEvent(
            NOW,
            sell.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
        ),
        sell_wallet,
    )[0]
    assert sell_fill.price == Decimal("99.90")


def test_partial_fill_is_quantized_to_exchange_quantity_step():
    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("step", item, NOW)
    matcher = ConservativeMatcher(
        MatchConfig(
            fee_rate=Decimal("0"),
            volume_participation=Decimal("1"),
            qty_step=Decimal("0.1"),
        )
    )
    fills = matcher.match(
        BarEvent(
            NOW,
            item.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("0.15"),
        ),
        wallet,
    )
    assert fills[0].quantity == Decimal("0.1")


def test_volume_below_quantity_step_does_not_create_dust_fill():
    wallet = CrossWallet(Decimal("1000"))
    item = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("dust", item, NOW)
    matcher = ConservativeMatcher(
        MatchConfig(
            fee_rate=Decimal("0"),
            volume_participation=Decimal("1"),
            qty_step=Decimal("0.1"),
        )
    )
    outcome = matcher.match_outcome(
        BarEvent(
            NOW,
            item.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("0.05"),
        ),
        wallet,
    )
    assert outcome.fills == ()
    assert outcome.expired_order_ids == ()


def test_wallet_tracks_maker_and_taker_fee_totals():
    from freqtrade.hedge.simulation.exchange import LiquidityRole

    wallet = CrossWallet(Decimal("1000"))
    maker_order = intent(PositionSide.LONG, OrderSide.BUY, "1", "100")
    wallet.accept_order("maker", maker_order)
    maker_fill = FillEvent(
        event_id="maker-fill",
        timestamp=NOW,
        order_id="maker",
        intent_id=maker_order.intent_id,
        symbol=maker_order.symbol,
        position_side=maker_order.position_side,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0.01"),
        reduce_only=False,
        bucket=maker_order.bucket,
        action=maker_order.action,
        liquidity_role=LiquidityRole.MAKER,
    )
    wallet.apply_fill(maker_fill)
    taker_order = intent(
        PositionSide.SHORT,
        OrderSide.SELL,
        "1",
        "100",
    )
    wallet.accept_order("taker", taker_order)
    taker_fill = FillEvent(
        event_id="taker-fill",
        timestamp=NOW,
        order_id="taker",
        intent_id=taker_order.intent_id,
        symbol=taker_order.symbol,
        position_side=taker_order.position_side,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0.02"),
        reduce_only=False,
        bucket=taker_order.bucket,
        action=taker_order.action,
        liquidity_role=LiquidityRole.TAKER,
    )
    wallet.apply_fill(taker_fill)
    assert wallet.maker_fees_paid == Decimal("0.01")
    assert wallet.taker_fees_paid == Decimal("0.02")
    assert wallet.maker_fill_count == 1
    assert wallet.taker_fill_count == 1


def test_market_slippage_cannot_make_sell_price_non_positive():
    import pytest

    with pytest.raises(ValueError, match="10000 bps"):
        MatchConfig(market_slippage_bps=Decimal("10000"))


def test_tactical_lots_are_created_merged_and_reduced_by_batch_id():
    wallet = CrossWallet(Decimal("1000"))
    add = OrderIntent.deterministic(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.TACTICAL,
        quantity=Decimal("2"),
        price=Decimal("100"),
        reduce_only=False,
        layer=1,
        tactical_lot_id="lot-a",
    )
    wallet.accept_order("add-lot", add)
    wallet.apply_fill(
        FillEvent(
            event_id="fill-lot-add",
            timestamp=NOW,
            order_id="add-lot",
            intent_id=add.intent_id,
            symbol=add.symbol,
            position_side=add.position_side,
            quantity=Decimal("2"),
            price=Decimal("100"),
            fee=Decimal("0.08"),
            reduce_only=False,
            bucket=add.bucket,
            action=add.action,
            layer=add.layer,
            tactical_lot_id="lot-a",
        )
    )
    close = OrderIntent.deterministic(
        symbol=add.symbol,
        position_side=PositionSide.LONG,
        order_side=OrderSide.SELL,
        action=IntentAction.REDUCE,
        bucket=PositionBucket.TACTICAL,
        quantity=Decimal("1"),
        price=Decimal("110"),
        reduce_only=True,
        layer=1,
        tactical_lot_id="lot-a",
    )
    wallet.accept_order("close-lot", close)
    wallet.apply_fill(
        FillEvent(
            event_id="fill-lot-close",
            timestamp=NOW,
            order_id="close-lot",
            intent_id=close.intent_id,
            symbol=close.symbol,
            position_side=close.position_side,
            quantity=Decimal("1"),
            price=Decimal("110"),
            fee=Decimal("0.044"),
            reduce_only=True,
            bucket=close.bucket,
            action=close.action,
            layer=close.layer,
            tactical_lot_id="lot-a",
        )
    )
    lot = wallet.long.tactical_lots["lot-a"]
    assert lot.quantity == Decimal("1")
    assert lot.closed_quantity == Decimal("1")
    assert lot.realized_pnl == Decimal("10")
    assert lot.fees == Decimal("0.124")
    assert wallet.long.tactical_quantity == Decimal("1")


def test_matcher_limits_distinct_entry_layers_per_bar_but_allows_same_layer_buckets():
    wallet = CrossWallet(Decimal("1000"))
    orders = [
        OrderIntent.deterministic(
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            order_side=OrderSide.BUY,
            action=IntentAction.OPEN,
            bucket=bucket,
            quantity=Decimal("1"),
            price=price,
            reduce_only=False,
            layer=layer,
        )
        for bucket, price, layer in (
            (PositionBucket.CORE, Decimal("99"), 1),
            (PositionBucket.TACTICAL, Decimal("99"), 1),
            (PositionBucket.TACTICAL, Decimal("95"), 2),
        )
    ]
    for index, item in enumerate(orders):
        wallet.accept_order(f"layer-{index}", item)
    outcome = ConservativeMatcher(
        MatchConfig(max_entry_layers_per_bar=1, volume_participation=Decimal("1"))
    ).match_outcome(
        BarEvent(
            NOW,
            "ETH/USDT:USDT",
            Decimal("100"),
            Decimal("100"),
            Decimal("90"),
            Decimal("90"),
            Decimal("100"),
        ),
        wallet,
    )
    assert {item.layer for item in outcome.fills} == {1}
    assert {item.bucket for item in outcome.fills} == {
        PositionBucket.CORE,
        PositionBucket.TACTICAL,
    }


def test_cross_liquidation_is_account_level_and_idempotent():
    wallet = CrossWallet(
        Decimal("100"),
        leverage=Decimal("20"),
        maintenance_margin_rate=Decimal("0.01"),
        liquidation_fee_rate=Decimal("0.01"),
    )
    add = intent(PositionSide.LONG, OrderSide.BUY, "10", "100")
    wallet.accept_order("leveraged", add)
    wallet.apply_fill(fill("leveraged", add, "10", "100"))
    outcome = ConservativeMatcher().match_outcome(
        BarEvent(
            NOW,
            add.symbol,
            Decimal("100"),
            Decimal("100"),
            Decimal("80"),
            Decimal("80"),
            Decimal("0"),
        ),
        wallet,
    )
    assert outcome.liquidation_event is not None
    event = outcome.liquidation_event
    assert wallet.apply_liquidation(event) is True
    assert wallet.apply_liquidation(event) is False
    assert wallet.long.quantity == Decimal("0")
    assert wallet.short.quantity == Decimal("0")
    assert wallet.active_orders == {}
    assert wallet.liquidated is True
    assert wallet.liquidation_count == 1
    assert wallet.long.realized_pnl == Decimal("-200")


def test_liquidation_report_reconciles_realized_pnl_and_fee():
    from freqtrade.hedge.simulation.reports import build_report

    wallet = CrossWallet(
        Decimal("100"),
        leverage=Decimal("20"),
        maintenance_margin_rate=Decimal("0.01"),
        liquidation_fee_rate=Decimal("0.01"),
    )
    add = intent(PositionSide.LONG, OrderSide.BUY, "10", "100")
    wallet.accept_order("leveraged", add)
    wallet.apply_fill(fill("leveraged", add, "10", "100"))
    event = wallet.create_liquidation_event(
        timestamp=NOW,
        symbol=add.symbol,
        price=Decimal("80"),
        ordinal=0,
    )
    wallet.apply_liquidation(event)
    report = build_report(wallet, Decimal("80"))
    assert report["liquidated"] is True
    assert report["liquidation_count"] == 1
    assert report["pnl_reconciliation_error"] == Decimal("0")


def test_liquidation_warning_uses_configured_buffer_threshold():
    wallet = CrossWallet(
        Decimal("100"),
        leverage=Decimal("20"),
        maintenance_margin_rate=Decimal("0.01"),
        liquidation_fee_rate=Decimal("0.01"),
        liquidation_buffer_warning_ratio=Decimal("0.05"),
    )
    add = intent(PositionSide.LONG, OrderSide.BUY, "10", "100")
    wallet.accept_order("warning", add)
    wallet.apply_fill(fill("warning", add, "10", "100"))
    assert wallet.liquidation_warning(Decimal("100")) is False
    assert wallet.liquidation_warning(Decimal("90")) is True
