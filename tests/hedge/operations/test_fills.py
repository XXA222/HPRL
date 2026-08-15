from decimal import Decimal
from freqtrade.hedge.operations.fills import FillOrder,PartialFillEngine

def test_partial_fill_respects_participation_ratio_and_step():
    e=PartialFillEngine(participation_rate=Decimal("0.1"),max_fill_ratio=Decimal("0.5"),qty_step=Decimal("0.01"));d=e.fill(FillOrder("o",Decimal("10"),Decimal("101"),"BUY"),bar_volume=Decimal("20"),market_price=Decimal("100"));assert d.filled_quantity==Decimal("2.0") and d.remaining_quantity==Decimal("8.0") and not d.terminal


def test_final_dust_step_converges():
    e=PartialFillEngine(participation_rate=Decimal("0.1"),max_fill_ratio=Decimal("0.5"),qty_step=Decimal("0.01"));d=e.fill(FillOrder("d",Decimal("0.01"),None,"BUY"),bar_volume=Decimal("1"),market_price=Decimal("100"));assert d.terminal and d.filled_quantity==Decimal("0.01")
