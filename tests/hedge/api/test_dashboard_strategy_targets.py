from datetime import UTC,datetime
from decimal import Decimal
from types import SimpleNamespace
from freqtrade.hedge.control.dryrun import DryRunControlState
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry,DryRunTelemetryStore
from freqtrade.rpc.api_server.hedge_dashboard import HedgeDashboardQuery

def test_dashboard_uses_planner_targets_not_actual_positions():
    store=DryRunTelemetryStore(10);store.append(DryRunCycleTelemetry(cycle_id="1",account_id="a",symbol="BTC",timestamp=datetime.now(UTC),mark_price=Decimal("100"),equity=Decimal("1000"),available_balance=Decimal("900"),gross_notional=Decimal("300"),net_quantity=Decimal("1"),target_net_quantity=Decimal("2"),net_gap_quantity=Decimal("1"),long_quantity=Decimal("3"),short_quantity=Decimal("2"),long_target_quantity=Decimal("4"),short_target_quantity=Decimal("2")))
    app=SimpleNamespace(telemetry=store,dryrun_control=DryRunControlState(),execution=None)
    bot=SimpleNamespace(hedge_application=app,hedge_runtime=None)
    o=HedgeDashboardQuery(account_id="a",symbol="BTC",bot_provider=lambda:bot).overview()
    assert o.latest.long_quantity==3 and o.latest.long_target_quantity==4 and o.latest.target_net_quantity==2
