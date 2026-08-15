from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqtrade.freqai.hedge_rl.accounting import (
    FeeLedger,
    FillRecord,
    FundingLedger,
    FundingPosting,
    IdempotentFillLedger,
    PositionAccumulator,
    audit_account_invariants,
    liquidation_buffer,
    mark_to_market_account,
    realized_pnl,
    validate_trade_price_quantity,
)
from freqtrade.freqai.hedge_rl.state import HedgeAccountState, HedgeLegSide, HedgeLegState


def test_round51_fill_ledger_is_idempotent_and_conflict_detecting():
    ledger = IdempotentFillLedger()
    fill = FillRecord(
        "fill-1",
        "order-1",
        HedgeLegSide.LONG,
        True,
        1.0,
        100.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert ledger.record(fill)
    assert not ledger.record(fill)
    with pytest.raises(ValueError):
        ledger.record(
            FillRecord(
                "fill-1",
                "order-1",
                HedgeLegSide.LONG,
                True,
                2.0,
                100.0,
                timestamp=fill.timestamp,
            )
        )


def test_round52_price_quantity_validation_checks_exchange_increments():
    validate_trade_price_quantity(
        price=100.5,
        quantity=0.01,
        min_price=1,
        min_quantity=0.001,
        tick_size=0.1,
        step_size=0.001,
    )
    with pytest.raises(ValueError):
        validate_trade_price_quantity(price=100.55, quantity=0.01, tick_size=0.1)
    with pytest.raises(ValueError):
        validate_trade_price_quantity(price=100, quantity=0.0001, min_quantity=0.001)


def test_round53_partial_fill_accumulator_updates_vwap_then_reduces():
    position = PositionAccumulator(HedgeLegSide.LONG)
    position = position.apply_fill(increasing=True, quantity=1, price=100)
    position = position.apply_fill(increasing=True, quantity=1, price=110)
    assert position.quantity == 2 and position.average_price == 105
    position = position.apply_fill(increasing=False, quantity=0.5, price=120)
    assert position.quantity == 1.5 and position.average_price == 105
    assert position.realized_pnl == 7.5


def test_round54_long_realized_pnl_sign_is_correct():
    assert realized_pnl(side=HedgeLegSide.LONG, entry_price=100, exit_price=110, quantity=2) == 20
    assert realized_pnl(side=HedgeLegSide.LONG, entry_price=100, exit_price=90, quantity=2) == -20


def test_round55_short_realized_pnl_sign_is_correct():
    assert realized_pnl(side=HedgeLegSide.SHORT, entry_price=100, exit_price=90, quantity=2) == 20
    assert realized_pnl(side=HedgeLegSide.SHORT, entry_price=100, exit_price=110, quantity=2) == -20


def test_round56_fee_ledger_reconciles_order_totals():
    ledger = FeeLedger()
    ledger.post("a", 1.25)
    ledger.post("a", 0.75)
    ledger.post("b", 2.0)
    assert ledger.total == 4.0 and ledger.by_order["a"] == 2.0
    assert ledger.reconcile() == 0.0


def test_round57_funding_ledger_is_idempotent_and_side_aware():
    ledger = FundingLedger()
    long = FundingPosting("event-long", HedgeLegSide.LONG, 1000, 0.001)
    short = FundingPosting("event-short", HedgeLegSide.SHORT, 500, 0.001)
    assert ledger.post(long) and ledger.post(short)
    assert not ledger.post(long)
    assert ledger.cashflow == -0.5


def test_round58_mark_to_market_keeps_legs_independent():
    account = HedgeAccountState(
        cash_balance=1000,
        equity=1000,
        peak_equity=1000,
        long=HedgeLegState(HedgeLegSide.LONG, 1, 90),
        short=HedgeLegState(HedgeLegSide.SHORT, 2, 110),
    )
    result = mark_to_market_account(account, mark=100)
    assert result.long_unrealized == 10
    assert result.short_unrealized == 20
    assert result.equity == 1030


def test_round59_liquidation_buffer_uses_gross_notional():
    account = HedgeAccountState(
        cash_balance=1000,
        equity=1000,
        peak_equity=1000,
        long=HedgeLegState(HedgeLegSide.LONG, 2, 100),
        short=HedgeLegState(HedgeLegSide.SHORT, 1, 100),
    )
    result = liquidation_buffer(account, mark=100, maintenance_rate=0.05)
    assert result.maintenance_margin == 15
    assert result.buffer_amount == 985
    assert result.buffer_ratio == 0.985


def test_round60_account_invariant_audit_detects_equity_drift():
    consistent = HedgeAccountState(
        cash_balance=1000,
        equity=1030,
        peak_equity=1030,
        long=HedgeLegState(HedgeLegSide.LONG, 1, 90),
        short=HedgeLegState(HedgeLegSide.SHORT, 2, 110),
    )
    assert audit_account_invariants(consistent, mark=100).valid
    drifted = HedgeAccountState(
        cash_balance=1000,
        equity=1040,
        peak_equity=1040,
        long=consistent.long,
        short=consistent.short,
    )
    report = audit_account_invariants(drifted, mark=100)
    assert not report.valid
    assert "EQUITY_DOES_NOT_MATCH_CASH_PLUS_UNREALIZED" in report.violations
