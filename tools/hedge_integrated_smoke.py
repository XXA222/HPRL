from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal

sys.path.insert(0, os.getcwd())

from freqtrade.enums.hedge import PositionMode
from freqtrade.hedge.config import HedgeRuntimeConfig
from freqtrade.hedge.integration import IntegratedPaperHedgeApplication
from freqtrade.hedge.integration.paper_state import JsonPaperStateStore
from freqtrade.hedge.planning.context import MarketSnapshot
from freqtrade.hedge.simulation.exchange import AccountEvent, AccountEventType, BarEvent
from freqtrade.hedge.runtime import HedgeRuntime
from freqtrade.hedge.readonly import runtime_config_from_freqtrade


_BASE_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def market(price: str, step: int) -> MarketSnapshot:
    mark = Decimal(price)
    return MarketSnapshot(
        symbol="ETH/USDT:USDT",
        timestamp=_BASE_TIME + timedelta(minutes=step),
        bid=mark - Decimal("0.10"),
        ask=mark + Decimal("0.10"),
        mark=mark,
        tick_size=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal(5),
    )


def closed_bar(snapshot: MarketSnapshot) -> BarEvent:
    """Build the closed DataProvider candle required by production Paper mode."""

    return BarEvent(
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
        open=snapshot.mark,
        high=snapshot.mark + Decimal(1),
        low=snapshot.mark - Decimal(1),
        close=snapshot.mark,
        volume=Decimal(1000),
    )


class CaptureAccountEventSink:
    """Small in-memory evidence sink used only by this offline smoke."""

    def __init__(self) -> None:
        self.events: list[AccountEvent] = []

    def record(self, event: AccountEvent) -> bool:
        self.events.append(event)
        return True

    def recover(self) -> None:
        return None


def main() -> int:
    readonly_config = runtime_config_from_freqtrade(
        {
            "exchange": {
                "key": "smoke-key",
                "secret": "smoke-secret",
                "pair_whitelist": ["ETH/USDT:USDT"],
            },
            "hedge": {
                "max_clock_skew_ms": 5000,
                "target_leverage": "3",
            },
        }
    )
    if readonly_config.max_clock_skew_ms != 5000 or readonly_config.target_leverage != 3:
        raise RuntimeError("readonly runtime configuration bridge did not apply clock/leverage")

    config = {
        "hedge": {
            "target_leverage": "3",
            "max_gross_notional": "1000",
            "max_gross_exposure_ratio": "1.0",
            "max_margin_utilization": "0.80",
            "min_liquidation_buffer_ratio": "0.05",
            "max_single_order_notional": "250",
            "paper": {
                "initial_balance": "1000",
                "auto_fill": True,
                "long_signal": "1",
                "short_signal": "1",
                "ohlcv_source": "dataprovider",
                "require_closed_candle": True,
            },
            "planner": {
                "cooldown_seconds": 0,
                "max_grid_layers": 3,
                "core_wallet_exposure_long": "0.05",
                "tactical_wallet_exposure_long": "0.10",
                "tactical_wallet_exposure_short": "0.10",
            },
        }
    }
    with tempfile.TemporaryDirectory(prefix="hedge-integrated-smoke-") as temp_dir:
        state_path = os.path.join(temp_dir, "paper-state.json")
        account_events = CaptureAccountEventSink()
        app = IntegratedPaperHedgeApplication(
            config=config,
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            state_store=JsonPaperStateStore(state_path),
            account_event_sink=account_events,
        )
        missing_bar_probe = market("2000", -1)
        try:
            app.run_market_cycle(missing_bar_probe)
        except ValueError as exc:
            if "closed DataProvider BarEvent" not in str(exc):
                raise
        else:
            raise RuntimeError("production Paper smoke must reject a missing closed BarEvent")

        snapshots = [
            market("2000", 0),
            market("1980", 1),
            market("2010", 2),
        ]
        cycles = [
            app.run_market_cycle(snapshot, bar=closed_bar(snapshot))
            for snapshot in snapshots
        ]
        runtime = HedgeRuntime(
            HedgeRuntimeConfig(
                position_mode=PositionMode.HEDGE,
                enabled=True,
                managed_pair="ETH/USDT:USDT",
                exchange_adapter="binance",
                read_only=True,
                account_id="hedge-main",
                live_trading_enabled=False,
                operation_mode="paper",
            )
        )
        app.publish_runtime(runtime)
        view = runtime.view()
        wallet = app.wallet()
        checkpoint = JsonPaperStateStore(state_path).load()
        event_types = {event.event_type for event in account_events.events}
        result = {
            "cycles": len(cycles),
            "planned": sum(len(item.planning.submit_orders) for item in cycles),
            "submitted": sum(len(item.executions) for item in cycles),
            "filled": sum(len(item.fills) for item in cycles),
            "account_events": len(account_events.events),
            "fee_event_present": AccountEventType.FEE in event_types,
            "checkpoint_present": checkpoint is not None,
            "checkpoint_schema_version": (
                None if checkpoint is None else checkpoint.get("schema_version")
            ),
            "long_quantity": str(wallet.long.quantity),
            "long_average": str(wallet.long.average_price),
            "short_quantity": str(wallet.short.quantity),
            "short_average": str(wallet.short.average_price),
            "equity": str(wallet.equity),
            "runtime_positions": len(view.positions),
            "runtime_risk_valid": bool(view.risk and view.risk.effective_risk_data_valid),
            "runtime_ready": view.ready,
            "readonly_max_clock_skew_ms": readonly_config.max_clock_skew_ms,
            "readonly_target_leverage": readonly_config.target_leverage,
        }
        if (
            result["filled"] <= 0
            or not result["fee_event_present"]
            or not result["checkpoint_present"]
            or wallet.long.quantity <= 0
            or wallet.short.quantity <= 0
            or len(view.positions) != 2
            or not result["runtime_risk_valid"]
        ):
            raise RuntimeError(f"integrated paper main-path smoke failed: {result}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
