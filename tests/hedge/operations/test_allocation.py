from decimal import Decimal
from freqtrade.hedge.operations.allocation import DryRunCapitalAllocator

def test_allocator_respects_reserve_pair_and_side_caps():
    a=DryRunCapitalAllocator(reserve_ratio=Decimal("0.1"),max_pair_ratio=Decimal("0.5"),max_side_ratio=Decimal("0.35"));rows=a.allocate(equity=Decimal("1000"),weights={"BTC":Decimal("2"),"ETH":Decimal("1")},long_bias={"BTC":Decimal("0.8")});btc=next(x for x in rows if x.symbol=="BTC");assert btc.total_notional<=Decimal("500") and btc.long_notional<=Decimal("350") and sum(x.total_notional for x in rows)<=Decimal("900")
