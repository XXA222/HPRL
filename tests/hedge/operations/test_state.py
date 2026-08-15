from datetime import UTC,datetime
from pathlib import Path
from freqtrade.hedge.operations.session import RunSession,SessionStatus
from freqtrade.hedge.operations.state import AtomicRunStateStore

def test_atomic_state_roundtrip_and_fail_closed(tmp_path:Path):
    s=RunSession.create(account_id="a",symbols=("BTCUSDT",),config={},started_at=datetime(2026,8,5,tzinfo=UTC)).transition(SessionStatus.RUNNING);store=AtomicRunStateStore(tmp_path/"state.json");store.save(s,{"x":1});loaded,extras=store.load();assert loaded==s and extras=={"x":1}
    (tmp_path/"state.json").write_text("{}",encoding="utf-8");session,extras,reasons=store.load_fail_closed();assert session is None and reasons and reasons[0].startswith("STATE_RECOVERY_FAILED")
