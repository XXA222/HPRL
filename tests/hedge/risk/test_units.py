from decimal import Decimal

from freqtrade.enums.hedge import PositionAction
from freqtrade.hedge.risk import (
    AccountRiskSnapshot,
    HedgeRiskEngine,
    RiskLimits,
)


def account() -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("8000"),
        initial_margin=Decimal("1000"),
        maintenance_margin=Decimal("100"),
        gross_long_notional=Decimal("3000"),
        gross_short_notional=Decimal("1000"),
        net_notional=Decimal("2000"),
        liquidation_buffer_ratio=Decimal("0.50"),
    )


def test_notional_and_ratio_limits_are_not_mixed() -> None:
    engine = HedgeRiskEngine(
        RiskLimits(
            max_gross_notional=Decimal("5000"),
            max_gross_exposure_ratio=Decimal("0.80"),
            max_margin_utilization=Decimal("0.55"),
            min_liquidation_buffer_ratio=Decimal("0.20"),
        )
    )
    decision = engine.evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("2000"),
        account=account(),
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("0.5")
    assert "GROSS_NOTIONAL_CLIPPED" in decision.reason_codes


def test_reduce_is_allowed_when_margin_is_high() -> None:
    high_margin = AccountRiskSnapshot(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("1000"),
        initial_margin=Decimal("9000"),
        maintenance_margin=Decimal("500"),
        gross_long_notional=Decimal("8000"),
        gross_short_notional=Decimal("1000"),
        net_notional=Decimal("7000"),
        liquidation_buffer_ratio=Decimal("0.10"),
    )
    engine = HedgeRiskEngine(
        RiskLimits(
            max_gross_notional=Decimal("10000"),
            max_margin_utilization=Decimal("0.55"),
            min_liquidation_buffer_ratio=Decimal("0.20"),
        )
    )
    decision = engine.evaluate(
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("2000"),
        account=high_margin,
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("1")
