from datetime import UTC,datetime
from decimal import Decimal
import pytest
from freqtrade.hedge.operations.ledger import FeeFundingLedger,LedgerEvent,LedgerEventType

def test_ledger_is_idempotent_and_reconciles_net():
    l=FeeFundingLedger();t=datetime(2026,8,5,tzinfo=UTC);fee=LedgerEvent("1",LedgerEventType.FEE,Decimal("2"),"USDT",t);assert l.append(fee) and not l.append(fee);l.append(LedgerEvent("2",LedgerEventType.FUNDING,Decimal("1"),"USDT",t));l.append(LedgerEvent("3",LedgerEventType.REALIZED_PNL,Decimal("10"),"USDT",t));assert l.balance().net==Decimal("9")
    with pytest.raises(ValueError):l.append(LedgerEvent("1",LedgerEventType.FEE,Decimal("3"),"USDT",t))
