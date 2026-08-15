from datetime import UTC,datetime,timedelta
from freqtrade.hedge.operations.universe import PairUniversePolicy

def test_pair_quarantine_and_auto_release():
    t=datetime(2026,8,5,tzinfo=UTC);p=PairUniversePolicy(("BTC/USDT:USDT","BTCUSDT","ETHUSDT"),failure_threshold=2,quarantine_seconds=60);assert p.symbols==("BTCUSDT","ETHUSDT")
    p.record_failure("BTCUSDT",at=t,reason="x");p.record_failure("BTCUSDT",at=t,reason="x");assert p.active_symbols(at=t)==("ETHUSDT",);assert "BTCUSDT" in p.active_symbols(at=t+timedelta(seconds=61))
