from decimal import Decimal
from freqtrade.hedge.operations.attribution import AttributionInput,PerformanceAttributor

def test_attribution_reconciles_equity_change():
    a=PerformanceAttributor().calculate(AttributionInput(Decimal("10"),Decimal("2"),Decimal("1"),Decimal("2"),Decimal("1")),equity_change=Decimal("10"));assert a.net_pnl==Decimal("10") and a.reconciliation_error==0
