from datetime import UTC,datetime
from freqtrade.hedge.operations.dashboard import OperationsDashboardSnapshot

def test_dashboard_snapshot_is_fail_closed_without_risk():
    s=OperationsDashboardSnapshot(datetime(2026,8,5,tzinfo=UTC),"s",None,("BTC",),"RUNNING",True,True,None,None,None,(),(),True);assert s.ready and s.summary()["active_alert_count"]==0
    s2=OperationsDashboardSnapshot(datetime(2026,8,5,tzinfo=UTC),"s",None,("BTC",),"PAUSED",True,True,None,None,None,(),(),False);assert not s2.ready
