from datetime import UTC,datetime
from freqtrade.hedge.operations.alerts import AlertManager,AlertSeverity

def test_alert_dedup_escalation_and_acknowledgement():
    m=AlertManager(escalate_after=3);t=datetime(2026,8,5,tzinfo=UTC);m.emit(key="x",severity=AlertSeverity.WARNING,message="x",at=t);m.emit(key="x",severity=AlertSeverity.WARNING,message="x",at=t);r=m.emit(key="x",severity=AlertSeverity.WARNING,message="x",at=t);assert r.severity==AlertSeverity.ERROR and len(m.active())==1;m.acknowledge("x");assert not m.active()
