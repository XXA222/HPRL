from decimal import Decimal
from freqtrade.hedge.operations.shadow import ShadowObservation,StrategyShadowComparator

def test_shadow_summary_is_idempotent_and_risk_aware():
    c=StrategyShadowComparator();x=ShadowObservation("1",Decimal("1"),Decimal("2"),Decimal("1"),Decimal("2"),Decimal("1"),Decimal("3"));assert c.record(x) and not c.record(x);s=c.summary();assert s.shadow_outperformed and s.risk_excess_cycles==1 and s.target_mae==Decimal("1")
