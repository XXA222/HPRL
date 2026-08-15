from datetime import UTC,datetime,timedelta
from decimal import Decimal
from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.simulation.exchange import BarEvent,SignalEvent
from freqtrade.hedge.simulation.replay import EventReplayEngine

def bar(t,p="100"):return BarEvent(timestamp=t,symbol="BTCUSDT",open=Decimal(p),high=Decimal("101"),low=Decimal("99"),close=Decimal(p),volume=Decimal("100"))
def test_replay_carries_strategy_directive_and_checkpoint():
    t=datetime(2026,1,1,tzinfo=UTC);engine=EventReplayEngine(initial_balance=Decimal("1000"),planner_config=PlannerConfig(max_wallet_exposure_long=Decimal("0.4"),max_wallet_exposure_short=Decimal("0.4"),max_gross_wallet_exposure=Decimal("0.65")))
    sig=SignalEvent(timestamp=t,symbol="BTCUSDT",long_signal=Decimal("1"),short_signal=Decimal("0"),target_net_ratio=Decimal("0.2"),confidence=Decimal("0.5"),long_exposure_scale=Decimal("0.5"),allow_new_risk=True,regime="BULL")
    engine.replay((sig,bar(t+timedelta(minutes=1)),))
    cp=engine.checkpoint();assert cp.signal_directive.target_net_ratio==Decimal("0.2") and cp.signal_directive.regime=="BULL"
    clone=EventReplayEngine(initial_balance=Decimal("1000"),planner_config=engine.planner_config);clone.restore(cp);assert clone.checkpoint().signal_directive==cp.signal_directive

def test_allow_new_risk_false_preserves_no_new_orders():
    t=datetime(2026,1,1,tzinfo=UTC);engine=EventReplayEngine(initial_balance=Decimal("1000"))
    result=engine.replay((SignalEvent(timestamp=t,symbol="BTCUSDT",long_signal=Decimal("1"),short_signal=Decimal("0"),allow_new_risk=False),bar(t+timedelta(minutes=1)),))
    assert not [e for e in result.events if type(e).__name__=="OrderAcceptedEvent"]
