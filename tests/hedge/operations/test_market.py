from datetime import UTC,datetime,timedelta
from decimal import Decimal
from freqtrade.hedge.operations.market import MarketDataHealthGate,MarketHealthInput

def test_market_gate_detects_stale_gap_and_divergence():
    gate=MarketDataHealthGate(max_age_seconds=90,max_gap_candles=1,max_divergence_bps=Decimal("20"));t=datetime(2026,8,5,tzinfo=UTC)
    assert gate.evaluate(MarketHealthInput("BTCUSDT",t,t+timedelta(seconds=30),60,Decimal("100"),Decimal("100"))).ready
    d=gate.evaluate(MarketHealthInput("BTCUSDT",t+timedelta(minutes=3),t+timedelta(minutes=5),60,Decimal("102"),Decimal("100")));assert not d.ready and {"MARKET_DATA_STALE","MARKET_CANDLE_GAP","MARK_INDEX_DIVERGENCE"}.issubset(d.reasons)
