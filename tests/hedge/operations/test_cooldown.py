from datetime import UTC,datetime,timedelta
from decimal import Decimal
from freqtrade.hedge.operations.cooldown import AdaptiveCooldownManager

def test_cooldown_grows_exponentially_and_expires():
    t=datetime(2026,8,5,tzinfo=UTC);m=AdaptiveCooldownManager(base_seconds=60,max_seconds=300);a=m.record_trade("BTC",pnl=Decimal("-1"),at=t);b=m.record_trade("BTC",pnl=Decimal("-1"),at=t);assert (b.until-t).total_seconds()==120 and b.loss_streak==2;assert not m.status("BTC",at=t+timedelta(seconds=121)).active


def test_profitable_trade_clears_loss_cooldown():
    t=datetime(2026,8,5,tzinfo=UTC);m=AdaptiveCooldownManager();m.record_trade("BTC",pnl=Decimal("-1"),at=t);assert m.status("BTC",at=t).active;m.record_trade("BTC",pnl=Decimal("1"),at=t);assert not m.status("BTC",at=t).active
