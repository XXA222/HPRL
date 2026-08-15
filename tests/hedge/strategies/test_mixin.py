import pytest
from freqtrade.strategy.hedge import HedgeStrategyMixin
class Frame:
    def __init__(self,columns):self.columns=columns
def test_scores_and_legacy_are_accepted():
    HedgeStrategyMixin.validate_hedge_dataframe(Frame(["open","high","low","close","volume","hedge_long_score","hedge_short_score"]))
    HedgeStrategyMixin.validate_hedge_dataframe(Frame(["open","high","low","close","volume","enter_long","enter_short"]))
def test_unknown_hedge_column_rejected():
    with pytest.raises(ValueError):HedgeStrategyMixin.validate_hedge_dataframe(Frame(["open","high","low","close","volume","hedge_long_score","hedge_short_score","hedge_magic"]))
