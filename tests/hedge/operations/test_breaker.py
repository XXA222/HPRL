from decimal import Decimal
from freqtrade.hedge.operations.breaker import BreakerState,DrawdownCircuitBreaker

def test_drawdown_breaker_hysteresis_and_kill_latch():
    b=DrawdownCircuitBreaker();assert b.evaluate(Decimal("100")).state is BreakerState.NORMAL;assert b.evaluate(Decimal("89")).state is BreakerState.PAUSED;assert b.evaluate(Decimal("95")).state is BreakerState.PAUSED;assert b.evaluate(Decimal("79")).state is BreakerState.KILLED;assert b.evaluate(Decimal("100")).state is BreakerState.KILLED;assert b.manual_reset(equity=Decimal("100")).state is BreakerState.NORMAL
