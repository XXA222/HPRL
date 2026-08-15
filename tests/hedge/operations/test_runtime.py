from datetime import UTC,datetime,timedelta
from decimal import Decimal
from pathlib import Path
from freqtrade.hedge.operations.runtime import DryRunOperationsRuntime,OperationsCycleInput

def test_runtime_composes_twenty_features_and_persists(tmp_path:Path):
    rt=DryRunOperationsRuntime(account_id="a",symbols=("BTCUSDT",),config={"hedge":{"operations":{"warmup_candles":2}}},state_path=tmp_path/"state.json");t=datetime(2026,8,5,tzinfo=UTC);x=OperationsCycleInput(t,"BTCUSDT",60,Decimal("100"),Decimal("100"),Decimal("1000"),Decimal("1000"),Decimal("100"),Decimal("50"),Decimal("20"),Decimal("1"),Decimal("2"),Decimal("0"),Decimal("1"),Decimal("0.1"),2,{})
    snap=rt.observe(x);assert snap.ready and (tmp_path/"state.json").exists();assert rt.certificate(at=t+timedelta(seconds=1)).ready


def test_runtime_resumes_matching_session_and_blocks_corrupt_state(tmp_path:Path):
    t=datetime(2026,8,5,tzinfo=UTC);path=tmp_path/"state.json";cfg={"hedge":{"operations":{"warmup_candles":1}}};rt=DryRunOperationsRuntime(account_id="a",symbols=("BTCUSDT",),config=cfg,state_path=path);x=OperationsCycleInput(t,"BTCUSDT",60,Decimal("100"),Decimal("100"),Decimal("1000"),Decimal("1000"),Decimal("0"),Decimal("0"),Decimal("0"),Decimal("0"),Decimal("0"),Decimal("0"),Decimal("0"),Decimal("0"),1,{});rt.observe(x);sid=rt.session.session_id;rt2=DryRunOperationsRuntime(account_id="a",symbols=("BTCUSDT",),config=cfg,state_path=path);assert rt2.session.session_id==sid and rt2.session.cycle_sequence==1;path.write_text("{}",encoding="utf-8");rt3=DryRunOperationsRuntime(account_id="a",symbols=("BTCUSDT",),config=cfg,state_path=path);assert rt3.recovery_reasons
