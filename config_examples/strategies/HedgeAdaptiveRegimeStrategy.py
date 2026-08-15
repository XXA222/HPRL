"""Clean-mainline adaptive regime example; not a profitability claim."""
from __future__ import annotations
import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.strategy.hedge import HedgeStrategyMixin

class HedgeAdaptiveRegimeStrategy(HedgeStrategyMixin, IStrategy):
    INTERFACE_VERSION=3
    can_short=True
    timeframe="1m"
    process_only_new_candles=True
    startup_candle_count=240
    minimal_roi={"0":10.0}
    stoploss=-0.99
    use_exit_signal=True
    def version(self)->str:return "hedge-adaptive-regime-mainline"
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        del metadata
        fast=dataframe["close"].ewm(span=16,adjust=False,min_periods=16).mean()
        slow=dataframe["close"].ewm(span=64,adjust=False,min_periods=64).mean()
        ret=dataframe["close"].pct_change()
        vol=ret.rolling(120,min_periods=60).std().clip(lower=0.00035)
        trend=((fast-slow)/dataframe["close"]/vol).clip(-3,3)
        long=(0.5+trend/6).clip(0,1);short=(0.5-trend/6).clip(0,1)
        baseline=vol.rolling(240,min_periods=120).median()
        stress=(vol/(baseline.replace(0,np.nan))).clip(0.5,4).fillna(4)
        confidence=(1-(stress-1).clip(0,2)/2).clip(0.15,1)
        risk=(1/stress).clip(0.20,1)
        valid=fast.notna()&slow.notna()&baseline.notna()
        dataframe["hedge_long_score"]=np.where(valid,long,0.0)
        dataframe["hedge_short_score"]=np.where(valid,short,0.0)
        dataframe["hedge_target_net_ratio"]=np.where(valid,(trend/3).clip(-0.35,0.35),0.0)
        dataframe["hedge_confidence"]=np.where(valid,confidence,0.0)
        dataframe["hedge_risk_scale"]=np.where(valid,risk,0.0)
        dataframe["hedge_long_exposure_scale"]=np.where(trend>=0,1.0,0.65)
        dataframe["hedge_short_exposure_scale"]=np.where(trend<=0,1.0,0.65)
        dataframe["hedge_allow_new_risk"]=valid & (stress<2.5)
        dataframe["hedge_regime"]=np.select([stress>=2.5,trend>=0.5,trend<=-0.5],["HIGH_VOL","BULL","BEAR"],default="RANGE")
        dataframe["hedge_reason"]=np.where(stress>=2.5,"VOLATILITY_FAIL_CLOSED","EMA_VOL_REGIME")
        dataframe["hedge_model_version"]=self.version()
        return dataframe
    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        del metadata
        dataframe.loc[(dataframe["hedge_long_score"]>=0.60)&dataframe["hedge_allow_new_risk"],"enter_long"]=1
        dataframe.loc[(dataframe["hedge_short_score"]>=0.60)&dataframe["hedge_allow_new_risk"],"enter_short"]=1
        return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        del metadata
        dataframe.loc[dataframe["hedge_long_score"]<0.50,"exit_long"]=1
        dataframe.loc[dataframe["hedge_short_score"]<0.50,"exit_short"]=1
        return dataframe
