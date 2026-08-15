from datetime import UTC,datetime,timedelta
import pytest
from freqtrade.hedge.operations.session import RunSession,SessionStatus

def test_session_lifecycle_and_monotonic_cycles():
    t=datetime(2026,8,5,tzinfo=UTC);s=RunSession.create(account_id="a",symbols=("btcusdt","BTCUSDT"),config={"x":1},started_at=t)
    assert s.symbols==("BTCUSDT",);s=s.transition(SessionStatus.RUNNING);s,c1=s.next_cycle(t+timedelta(minutes=1));s,c2=s.next_cycle(t+timedelta(minutes=2));assert c1.endswith("000000000001") and c2.endswith("000000000002")
    with pytest.raises(ValueError):s.next_cycle(t+timedelta(minutes=2))
