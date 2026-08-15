from datetime import UTC,datetime,timedelta
from decimal import Decimal
from freqtrade.hedge.operations.admission import AdmissionIntent,IntentAdmissionGate

def test_reduce_only_bypasses_new_risk_blocks_but_duplicates_do_not():
    g=IntentAdmissionGate(min_notional=Decimal("5"),max_notional=Decimal("100"),max_open_orders=1);now=datetime(2026,8,5,tzinfo=UTC)
    blocked=g.evaluate(AdmissionIntent("a","BTC","LONG",Decimal("1"),Decimal("10")),open_orders=1,new_risk_enabled=False,cooldown_until=now+timedelta(minutes=1),now=now);assert not blocked.approved
    reduce=g.evaluate(AdmissionIntent("b","BTC","LONG",Decimal("1"),Decimal("1"),True),open_orders=99,new_risk_enabled=False,cooldown_until=now+timedelta(minutes=1),now=now);assert reduce.approved
    assert not g.evaluate(AdmissionIntent("b","BTC","LONG",Decimal("1"),Decimal("1"),True),open_orders=0,new_risk_enabled=True,cooldown_until=None,now=now).approved
