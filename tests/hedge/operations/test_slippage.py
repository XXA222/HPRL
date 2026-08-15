from decimal import Decimal
from freqtrade.hedge.operations.slippage import DeterministicSlippageModel

def test_slippage_is_bounded_and_directional():
    m=DeterministicSlippageModel(cap_bps=Decimal("50"));buy=m.quote(side="BUY",reference_price=Decimal("100"),order_notional=Decimal("100"),available_notional=Decimal("100"),volatility=Decimal("10"));sell=m.quote(side="SELL",reference_price=Decimal("100"),order_notional=Decimal("100"),available_notional=Decimal("100"),volatility=Decimal("10"));assert buy.slippage_bps==Decimal("50") and buy.execution_price>Decimal("100") and sell.execution_price<Decimal("100")
