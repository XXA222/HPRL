from decimal import Decimal

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.risk import AccountRiskSnapshot, HedgeRiskEngine, RiskLimits, RiskRequest


def account(**overrides):
    values = dict(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("8000"),
        initial_margin=Decimal("1000"),
        maintenance_margin=Decimal("100"),
        gross_long_notional=Decimal("3000"),
        gross_short_notional=Decimal("1000"),
        net_notional=Decimal("2000"),
        pending_order_notional=Decimal("0"),
        liquidation_buffer_ratio=Decimal("0.50"),
    )
    values.update(overrides)
    return AccountRiskSnapshot(**values)


def limits():
    return RiskLimits(
        max_margin_utilization=Decimal("0.55"),
        min_liquidation_buffer_ratio=Decimal("0.20"),
        max_gross_notional=Decimal("5000"),
        max_gross_exposure_ratio=Decimal("0.80"),
        max_single_order_notional=Decimal("2500"),
        max_leg_notional=Decimal("4500"),
        max_symbol_gross_notional=Decimal("4800"),
    )


def test_ratio_and_quote_notional_are_not_mixed() -> None:
    unit_limits = RiskLimits(
        max_margin_utilization=Decimal("0.55"),
        min_liquidation_buffer_ratio=Decimal("0.20"),
        max_gross_notional=Decimal("5000"),
        max_gross_exposure_ratio=Decimal("0.80"),
    )
    decision = HedgeRiskEngine(unit_limits).evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("2000"),
        account=account(),
        position_side=PositionSide.LONG,
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("0.5")
    assert "GROSS_NOTIONAL_CLIPPED" in decision.reason_codes
    assert "GROSS_EXPOSURE_RATIO_CLIPPED" not in decision.reason_codes


def test_pending_orders_consume_account_capacity() -> None:
    decision = HedgeRiskEngine(limits()).evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("2"),
        reference_price=Decimal("1000"),
        account=account(pending_order_notional=Decimal("900")),
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("0.1")


def test_high_risk_still_allows_controlled_reduce() -> None:
    decision = HedgeRiskEngine(limits()).evaluate(
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("8"),
        reference_price=Decimal("2000"),
        account=account(
            initial_margin=Decimal("9000"),
            liquidation_buffer_ratio=Decimal("0.05"),
        ),
        position_side=PositionSide.LONG,
        confirmed_quantity=Decimal("10"),
        pending_reduce_quantity=Decimal("4"),
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("6")


def test_approval_never_expands_requested_quantity() -> None:
    decision = HedgeRiskEngine(limits()).evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.01"),
        reference_price=Decimal("1000"),
        account=account(),
    )
    assert decision.approved_quantity <= Decimal("0.01")


def test_projected_margin_utilization_clips_new_order() -> None:
    margin_limits = RiskLimits(
        max_margin_utilization=Decimal("0.55"),
        min_liquidation_buffer_ratio=Decimal("0.20"),
        max_gross_notional=Decimal("50000"),
    )
    decision = HedgeRiskEngine(margin_limits).evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("5"),
        reference_price=Decimal("1000"),
        leverage=Decimal("2"),
        account=account(initial_margin=Decimal("5000")),
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("1")
    assert "MARGIN_UTILIZATION_CLIPPED" in decision.reason_codes


def test_available_margin_clips_new_order() -> None:
    margin_limits = RiskLimits(
        max_margin_utilization=Decimal("0.90"),
        min_liquidation_buffer_ratio=Decimal("0.20"),
        max_gross_notional=Decimal("50000"),
    )
    decision = HedgeRiskEngine(margin_limits).evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("2"),
        reference_price=Decimal("1000"),
        leverage=Decimal("5"),
        account=account(available_balance=Decimal("100")),
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("0.5")
    assert "AVAILABLE_MARGIN_CLIPPED" in decision.reason_codes


def test_net_limit_distinguishes_long_and_short_increases() -> None:
    net_limits = RiskLimits(
        max_margin_utilization=Decimal("0.90"),
        min_liquidation_buffer_ratio=Decimal("0.20"),
        max_gross_notional=Decimal("50000"),
        max_net_notional=Decimal("2500"),
    )
    engine = HedgeRiskEngine(net_limits)
    long_decision = engine.evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("1000"),
        position_side=PositionSide.LONG,
        leverage=Decimal("5"),
        account=account(),
    )
    short_decision = engine.evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("1000"),
        position_side=PositionSide.SHORT,
        leverage=Decimal("5"),
        account=account(),
    )
    assert long_decision.approved_quantity == Decimal("0.5")
    assert "NET_NOTIONAL_CLIPPED" in long_decision.reason_codes
    assert short_decision.approved_quantity == Decimal("1")


def test_account_snapshot_rejects_non_boolean_risk_validity() -> None:
    import pytest

    with pytest.raises(Exception, match="risk_data_valid must be a boolean"):
        account(risk_data_valid="false")


def test_net_limit_requires_order_to_finish_inside_limit() -> None:
    limits = RiskLimits(
        max_margin_utilization=Decimal("0.9"),
        min_liquidation_buffer_ratio=Decimal("0.1"),
        max_gross_notional=Decimal("100000"),
        max_net_notional=Decimal("8000"),
    )
    engine = HedgeRiskEngine(limits)
    over_limit = account(
        gross_long_notional=Decimal("10000"),
        gross_short_notional=Decimal("0"),
        net_notional=Decimal("10000"),
        initial_margin=Decimal("0"),
        available_balance=Decimal("10000"),
    )
    insufficient = engine.evaluate_request(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.SHORT,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("1"),
            reference_price=Decimal("1000"),
            leverage=Decimal("10"),
        ),
        account=over_limit,
    )
    assert not insufficient.allowed
    assert "NET_NOTIONAL_REMEDIATION_INSUFFICIENT" in insufficient.reason_codes

    sufficient = engine.evaluate_request(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.SHORT,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("2"),
            reference_price=Decimal("1000"),
            leverage=Decimal("10"),
        ),
        account=over_limit,
    )
    assert sufficient.allowed
    assert sufficient.approved_notional == Decimal("2000")


def test_account_snapshot_rejects_inconsistent_pending_risk_units() -> None:
    import pytest

    with pytest.raises(Exception, match="pending_net_notional_delta"):
        account(
            pending_order_notional=Decimal("100"),
            pending_net_notional_delta=Decimal("101"),
        )
    with pytest.raises(Exception, match="pending_order_initial_margin"):
        account(
            pending_order_notional=Decimal("100"),
            pending_order_initial_margin=Decimal("101"),
        )


def test_liquidation_helpers_cover_long_short_and_account_buffer() -> None:
    from freqtrade.hedge.risk import (
        calculate_account_maintenance_buffer,
        calculate_leg_liquidation_buffer,
        minimum_liquidation_buffer,
    )

    long_buffer = calculate_leg_liquidation_buffer(
        position_side=PositionSide.LONG,
        mark_price=Decimal("100"),
        liquidation_price=Decimal("80"),
    )
    short_buffer = calculate_leg_liquidation_buffer(
        position_side=PositionSide.SHORT,
        mark_price=Decimal("100"),
        liquidation_price=Decimal("130"),
    )
    account_buffer = calculate_account_maintenance_buffer(
        equity=Decimal("1000"),
        maintenance_margin=Decimal("850"),
    )
    assert long_buffer.buffer_ratio == Decimal("0.2")
    assert short_buffer.buffer_ratio == Decimal("0.3")
    assert account_buffer == Decimal("0.15")
    assert minimum_liquidation_buffer(
        (long_buffer, short_buffer),
        account_maintenance_buffer=account_buffer,
    ) == Decimal("0.15")


def test_risk_request_rejects_subunit_leverage_and_nonstring_account() -> None:
    import pytest

    with pytest.raises(Exception, match="greater than or equal to 1"):
        RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("1"),
            reference_price=Decimal("1000"),
            leverage=Decimal("0.5"),
        )
    with pytest.raises(Exception, match="account_id must be a string"):
        account(account_id=123)
