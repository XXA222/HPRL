from datetime import UTC,datetime
from decimal import Decimal
from pathlib import Path
from freqtrade.hedge.control.dryrun import DryRunControlMode,DryRunControlState
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry,JsonlDryRunTelemetryStore

def item(i=1):return DryRunCycleTelemetry(cycle_id=str(i),account_id="a",symbol="BTC",timestamp=datetime.now(UTC),mark_price=Decimal("100"),equity=Decimal("1000"),available_balance=Decimal("900"),gross_notional=Decimal("100"),net_quantity=Decimal("1"))

def test_control_persists_and_corruption_fails_closed(tmp_path):
    p=tmp_path/"control.json";s=DryRunControlState(p);s.pause_new_risk(actor="u",reason="x")
    assert DryRunControlState(p).snapshot().mode is DryRunControlMode.NEW_RISK_PAUSED
    p.write_text("{bad",encoding="utf-8")
    recovered=DryRunControlState(p).snapshot();assert not recovered.new_risk_enabled and recovered.reason=="CONTROL_STATE_RECOVERY_FAILED"

def test_jsonl_telemetry_survives_restart_and_old_rows(tmp_path):
    p=tmp_path/"cycles.jsonl";s=JsonlDryRunTelemetryStore(p,capacity=10);s.append(item())
    assert JsonlDryRunTelemetryStore(p,capacity=10).latest().cycle_id=="1"
    p.write_text(p.read_text()+"{truncated\n",encoding="utf-8")
    assert len(JsonlDryRunTelemetryStore(p,capacity=10).list())==1

def test_disk_failure_is_non_authoritative(tmp_path,monkeypatch):
    s=JsonlDryRunTelemetryStore(tmp_path/"cycles.jsonl",capacity=10)
    def boom(*a,**k):raise OSError("disk full")
    monkeypatch.setattr(Path,"open",boom)
    s.append(item())
    assert s.latest() is not None and "disk full" in s.last_error
