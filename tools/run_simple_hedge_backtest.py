"""Run a deterministic dual-leg Hedge strategy backtest.

The default data is a synthetic ETH/USDT perpetual 5-minute series so the
command works offline.  A user CSV may be supplied with timestamp/open/high/
low/close/volume columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent
from freqtrade.hedge.strategies.simple_ma_hedge import SimpleDualLegMaHedgeStrategy
from freqtrade.optimize.hedge_backtesting import HedgeBacktesting

D = Decimal
SYMBOL = "ETH/USDT:USDT"


def _decimal(value: object) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"non-finite decimal: {value!r}")
    return result


def _parse_timestamp(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_csv(path: Path, symbol: str) -> list[BarEvent]:
    bars: list[BarEvent] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            volume_text = (row.get("volume") or "").strip()
            bars.append(
                BarEvent(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    symbol=symbol,
                    open=_decimal(row["open"]),
                    high=_decimal(row["high"]),
                    low=_decimal(row["low"]),
                    close=_decimal(row["close"]),
                    volume=None if not volume_text else _decimal(volume_text),
                )
            )
    if not bars:
        raise ValueError("CSV contains no bars")
    return bars


def synthetic_bars(count: int, symbol: str) -> list[BarEvent]:
    if count < 40:
        raise ValueError("at least 40 bars are required")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    closes: list[Decimal] = []
    for index in range(count):
        if index < count // 2:
            trend = index * 0.18
        else:
            trend = (count // 2) * 0.18 - (index - count // 2) * 0.12
        value = 2000 + trend + 65 * math.sin(index / 10) + 22 * math.sin(index / 3.7)
        closes.append(D(str(round(value, 4))))

    result: list[BarEvent] = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index else close
        high = max(open_price, close) + D("8") + D(str(round(abs(math.sin(index)) * 4, 4)))
        low = min(open_price, close) - D("8") - D(str(round(abs(math.sin(index / 2)) * 4, 4)))
        result.append(
            BarEvent(
                timestamp=start + timedelta(minutes=5 * index),
                symbol=symbol,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=D("1200"),
            )
        )
    return result


def with_funding(events: Iterable[SignalEvent | BarEvent]) -> list[SignalEvent | BarEvent | FundingEvent]:
    output: list[SignalEvent | BarEvent | FundingEvent] = []
    bar_index = 0
    for event in events:
        output.append(event)
        if isinstance(event, BarEvent):
            bar_index += 1
            if bar_index % 96 == 0:
                output.append(
                    FundingEvent(
                        timestamp=event.timestamp + timedelta(seconds=1),
                        symbol=event.symbol,
                        rate=D("0.0001"),
                        mark_price=event.close,
                    )
                )
    return output


def planner_config() -> PlannerConfig:
    return PlannerConfig(
        core_wallet_exposure_long=D("0.12"),
        core_wallet_exposure_short=D("0.12"),
        tactical_wallet_exposure_long=D("0.10"),
        tactical_wallet_exposure_short=D("0.10"),
        max_wallet_exposure_long=D("0.28"),
        max_wallet_exposure_short=D("0.28"),
        max_gross_wallet_exposure=D("0.50"),
        initial_entry_fraction=D("0.60"),
        max_grid_layers=3,
        grid_spacing=D("0.006"),
        grid_spacing_growth=D("1.30"),
        grid_qty_growth=D("1.10"),
        trailing_rebound=D("0"),
        take_profit_spacing=D("0.006"),
        take_profit_layers=2,
        tactical_reduce_fraction=D("0.50"),
        cooldown_seconds=0,
        replace_min_age_seconds=0,
        max_pending_entries=3,
        unstuck_trigger_gross_exposure=D("0.48"),
    )


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_outputs(output_dir: Path, result, signals: list[SignalEvent], source: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(result.report)
    final = result.snapshots[-1]
    payload = {
        "strategy": "SimpleDualLegMaHedgeStrategy",
        "source": source,
        "bar_count": len(result.snapshots) - len([event for event in result.events if isinstance(event, FundingEvent)]),
        "report": report,
        "final_snapshot": {
            name: getattr(final, name)
            for name in final.__dataclass_fields__
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "equity_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "balance",
                "equity",
                "long_quantity",
                "short_quantity",
                "gross_notional",
                "net_notional",
                "fees",
                "funding",
                "liquidated",
            ]
        )
        for snapshot in result.snapshots:
            writer.writerow(
                [
                    snapshot.timestamp.isoformat(),
                    snapshot.balance,
                    snapshot.equity,
                    snapshot.long_quantity,
                    snapshot.short_quantity,
                    snapshot.gross_notional,
                    snapshot.net_notional,
                    snapshot.fees,
                    snapshot.funding,
                    snapshot.liquidated,
                ]
            )

    with (output_dir / "signals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "long_signal", "short_signal"])
        for signal in signals:
            writer.writerow([signal.timestamp.isoformat(), signal.long_signal, signal.short_signal])

    summary = [
        "SimpleDualLegMaHedgeStrategy backtest",
        f"source={source}",
        f"final_equity={report['final_equity']}",
        f"return_pct={(report['final_equity'] / D('1000') - D('1')) * D('100')}",
        f"long_quantity={final.long_quantity}",
        f"short_quantity={final.short_quantity}",
        f"add_count={report['add_count']}",
        f"reduce_count={report['reduce_count']}",
        f"fees={report['fees']}",
        f"funding={report['funding']}",
        f"max_drawdown={report['max_drawdown']}",
        f"liquidated={report['liquidated']}",
        f"pnl_reconciliation_error={report['pnl_reconciliation_error']}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--bars", type=int, default=288)
    parser.add_argument("--initial-balance", default="1000")
    parser.add_argument("--leverage", default="3")
    parser.add_argument("--fee-rate", default="0.0004")
    parser.add_argument("--output-dir", type=Path, default=Path("user_data/backtest_results/simple_dual_leg_hedge"))
    args = parser.parse_args()

    if args.csv:
        bars = load_csv(args.csv, args.symbol)
        source = str(args.csv.resolve())
    else:
        bars = synthetic_bars(args.bars, args.symbol)
        source = f"synthetic:{args.bars}x5m"

    strategy = SimpleDualLegMaHedgeStrategy()
    strategy_events = list(strategy.events(bars))
    signals = [event for event in strategy_events if isinstance(event, SignalEvent)]
    events = with_funding(strategy_events)
    result = HedgeBacktesting(
        initial_balance=_decimal(args.initial_balance),
        planner_config=planner_config(),
        leverage=_decimal(args.leverage),
        fee_rate=_decimal(args.fee_rate),
        long_signal=D("0.20"),
        short_signal=D("0.20"),
    ).run(events)

    write_outputs(args.output_dir, result, signals, source)
    final = result.snapshots[-1]
    report = result.report
    console = {
        "strategy": "SimpleDualLegMaHedgeStrategy",
        "source": source,
        "output_dir": str(args.output_dir.resolve()),
        "final_equity": str(report["final_equity"]),
        "return_pct": str((report["final_equity"] / _decimal(args.initial_balance) - D("1")) * D("100")),
        "long_quantity": str(final.long_quantity),
        "short_quantity": str(final.short_quantity),
        "add_count": report["add_count"],
        "reduce_count": report["reduce_count"],
        "fees": str(report["fees"]),
        "funding": str(report["funding"]),
        "max_drawdown": str(report["max_drawdown"]),
        "liquidated": report["liquidated"],
        "pnl_reconciliation_error": str(report["pnl_reconciliation_error"]),
    }
    print(json.dumps(console, ensure_ascii=False, indent=2))
    if report["liquidated"] or report["pnl_reconciliation_error"] != D("0"):
        return 2
    if final.long_quantity <= D("0") or final.short_quantity <= D("0"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
