"""Backtest result convergence for Hedge simulations.

The adapter keeps the authoritative Hedge event/snapshot model and adds a stable,
FreqUI-friendly report surface.  It deliberately does not fabricate ordinary Trade
rows; dual-leg and Core/Tactical semantics remain explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from json import dumps
from math import sqrt
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from .models import ZERO, finite_decimal, utc_datetime


def _number(value: object, default: Decimal = ZERO) -> Decimal:
    try:
        return finite_decimal(value, field_name="metric")
    except (TypeError, ValueError):
        return default


def _get(row: object, name: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _iso(value: object) -> str:
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat()
    return str(value or "")


def _max_drawdown(equity: Sequence[Decimal]) -> tuple[Decimal, int, int]:
    if not equity:
        return ZERO, 0, 0
    peak = equity[0]
    peak_index = 0
    worst = ZERO
    worst_peak = 0
    worst_end = 0
    for index, value in enumerate(equity):
        if value > peak:
            peak = value
            peak_index = index
        if peak <= ZERO:
            continue
        drawdown = (peak - value) / peak
        if drawdown > worst:
            worst = drawdown
            worst_peak = peak_index
            worst_end = index
    return worst, worst_peak, worst_end


def _period_returns(equity: Sequence[Decimal]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(equity, equity[1:]):
        if previous > ZERO:
            returns.append(float(current / previous - Decimal("1")))
    return returns


def _risk_metrics(equity: Sequence[Decimal], periods_per_year: int) -> dict[str, float]:
    returns = _period_returns(equity)
    if not returns:
        return {"sharpe": 0.0, "sortino": 0.0, "volatility": 0.0}
    mean = fmean(returns)
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    downside = [min(item, 0.0) for item in returns]
    downside_deviation = sqrt(fmean([item * item for item in downside])) if downside else 0.0
    scale = sqrt(max(periods_per_year, 1))
    return {
        "sharpe": 0.0 if volatility == 0 else mean / volatility * scale,
        "sortino": 0.0 if downside_deviation == 0 else mean / downside_deviation * scale,
        "volatility": volatility * scale,
    }


def _event_pnl(event: object) -> Decimal:
    for name in ("realized_pnl", "pnl", "profit_abs", "realized"):
        value = _get(event, name)
        if value is not None:
            return _number(value)
    return ZERO


def _event_type(event: object) -> str:
    return str(_get(event, "event_type", _get(event, "type", ""))).upper()


@dataclass(frozen=True, slots=True)
class HedgeBacktestMetrics:
    starting_balance: Decimal
    final_equity: Decimal
    absolute_profit: Decimal
    total_return: Decimal
    max_drawdown: Decimal
    max_gross_notional: Decimal
    max_margin_utilization: Decimal
    fees: Decimal
    funding: Decimal
    realized_pnl: Decimal
    event_count: int
    fill_count: int
    winning_realizations: int
    losing_realizations: int
    profit_factor: Decimal | None
    expectancy: Decimal
    sharpe: float
    sortino: float
    volatility: float


@dataclass(frozen=True, slots=True)
class HedgeBacktestArtifact:
    strategy_name: str
    pairs: tuple[str, ...]
    timeframe: str
    timerange: str
    metrics: HedgeBacktestMetrics
    snapshots: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "hedge-backtest-result-v4"

    def to_dict(self) -> dict[str, Any]:
        metrics = {
            item.name: (
                str(getattr(self.metrics, item.name))
                if isinstance(getattr(self.metrics, item.name), Decimal)
                else getattr(self.metrics, item.name)
            )
            for item in fields(self.metrics)
        }
        payload = {
            "schema": self.schema,
            "strategy": self.strategy_name,
            "pairs": list(self.pairs),
            "timeframe": self.timeframe,
            "timerange": self.timerange,
            "metrics": metrics,
            "snapshots": [dict(item) for item in self.snapshots],
            "events": [dict(item) for item in self.events],
            "metadata": dict(self.metadata),
        }
        canonical = dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        payload["result_sha256"] = sha256(canonical.encode()).hexdigest()
        return payload

    def frequi_projection(self) -> dict[str, Any]:
        """Return a standard-shaped strategy summary without flattening dual-leg facts."""

        m = self.metrics
        return {
            "strategy": {
                self.strategy_name: {
                    "trades": [],
                    "locks": [],
                    "best_pair": {"key": self.pairs[0] if self.pairs else "", "profit_sum": float(m.absolute_profit)},
                    "worst_pair": {"key": self.pairs[-1] if self.pairs else "", "profit_sum": float(m.absolute_profit)},
                    "profit_total_abs": float(m.absolute_profit),
                    "profit_total": float(m.total_return),
                    "profit_total_long": None,
                    "profit_total_short": None,
                    "backtest_start": self.metadata.get("start", ""),
                    "backtest_end": self.metadata.get("end", ""),
                    "max_drawdown_account": float(m.max_drawdown),
                    "starting_balance": float(m.starting_balance),
                    "final_balance": float(m.final_equity),
                    "trade_count": m.fill_count,
                    "wins": m.winning_realizations,
                    "losses": m.losing_realizations,
                    "draws": max(0, m.fill_count - m.winning_realizations - m.losing_realizations),
                    "sharpe": m.sharpe,
                    "sortino": m.sortino,
                    "hedge_native": self.to_dict(),
                }
            },
            "strategy_comparison": [
                {
                    "key": self.strategy_name,
                    "trades": m.fill_count,
                    "profit_total_abs": float(m.absolute_profit),
                    "profit_total": float(m.total_return),
                    "max_drawdown_account": float(m.max_drawdown),
                    "sharpe": m.sharpe,
                    "sortino": m.sortino,
                }
            ],
            "metadata": {self.strategy_name: dict(self.metadata)},
        }


class HedgeBacktestResultAdapter:
    """Build deterministic metrics from any simulation exposing events and snapshots."""

    def __init__(self, *, periods_per_year: int = 365 * 24 * 60) -> None:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.periods_per_year = periods_per_year

    @staticmethod
    def _snapshot_row(snapshot: object) -> dict[str, Any]:
        names = (
            "timestamp",
            "equity",
            "wallet_balance",
            "available_balance",
            "long_quantity",
            "short_quantity",
            "long_average_price",
            "short_average_price",
            "gross_notional",
            "net_notional",
            "margin_utilization",
            "realized_pnl",
            "unrealized_pnl",
            "fees",
            "funding",
            "liquidation_buffer_ratio",
        )
        row: dict[str, Any] = {}
        for name in names:
            value = _get(snapshot, name)
            if value is None:
                continue
            row[name] = _iso(value) if name == "timestamp" else str(value)
        return row

    @staticmethod
    def _event_row(event: object) -> dict[str, Any]:
        if isinstance(event, Mapping):
            return {str(k): (v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, Decimal) else v) for k, v in event.items()}
        row: dict[str, Any] = {}
        slots = getattr(type(event), "__slots__", ())
        names = tuple(slots) if slots else tuple(getattr(event, "__dict__", {}))
        for name in names:
            value = getattr(event, name)
            row[name] = value.isoformat() if isinstance(value, datetime) else str(value) if isinstance(value, Decimal) else getattr(value, "value", value)
        return row

    def build(
        self,
        simulation: object,
        *,
        strategy_name: str,
        pairs: Iterable[str],
        timeframe: str,
        timerange: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> HedgeBacktestArtifact:
        snapshots_raw = tuple(getattr(simulation, "snapshots", ()))
        events_raw = tuple(getattr(simulation, "events", ()))
        snapshots = tuple(self._snapshot_row(item) for item in snapshots_raw)
        events = tuple(self._event_row(item) for item in events_raw)
        report = getattr(simulation, "report", {})
        report_start = _number(
            _get(report, "starting_balance", _get(report, "initial_balance", ZERO))
        )
        report_final = _number(_get(report, "final_equity", report_start))
        equity = [_number(_get(item, "equity")) for item in snapshots_raw]
        if not equity:
            equity = [report_start, report_final]
        else:
            if report_start > ZERO and equity[0] != report_start:
                equity.insert(0, report_start)
            if equity[-1] != report_final:
                equity.append(report_final)
        start = report_start if report_start > ZERO else equity[0]
        final = report_final if report_final > ZERO else equity[-1]
        profit = final - start
        report_return = _get(report, "total_return_ratio")
        total_return = (
            _number(report_return)
            if report_return is not None
            else (ZERO if start <= ZERO else profit / start)
        )
        drawdown, drawdown_peak, drawdown_end = _max_drawdown(equity)
        report_drawdown = _number(_get(report, "max_drawdown", ZERO))
        if report_drawdown > drawdown:
            drawdown = report_drawdown
        realized_values = [_event_pnl(item) for item in events_raw if _event_pnl(item) != ZERO]
        gross_profit = sum((item for item in realized_values if item > ZERO), ZERO)
        gross_loss = -sum((item for item in realized_values if item < ZERO), ZERO)
        profit_factor = None if gross_loss == ZERO else gross_profit / gross_loss
        fills = [item for item in events_raw if "FILL" in _event_type(item) or _get(item, "trade_id") is not None]
        max_gross = max(
            max(
                (_number(_get(item, "gross_notional")) for item in snapshots_raw),
                default=ZERO,
            ),
            _number(_get(report, "gross_peak", ZERO)),
        )
        max_margin = max((_number(_get(item, "margin_utilization")) for item in snapshots_raw), default=ZERO)
        last = snapshots_raw[-1] if snapshots_raw else {}
        fees = _number(_get(report, "fees", _get(last, "fees", ZERO)))
        funding = _number(_get(report, "funding", _get(last, "funding", ZERO)))
        realized = _number(_get(last, "realized_pnl", sum(realized_values, ZERO)))
        risk = _risk_metrics(equity, self.periods_per_year)
        metrics = HedgeBacktestMetrics(
            starting_balance=start,
            final_equity=final,
            absolute_profit=profit,
            total_return=total_return,
            max_drawdown=drawdown,
            max_gross_notional=max_gross,
            max_margin_utilization=max_margin,
            fees=fees,
            funding=funding,
            realized_pnl=realized,
            event_count=len(events_raw),
            fill_count=len(fills),
            winning_realizations=sum(1 for item in realized_values if item > ZERO),
            losing_realizations=sum(1 for item in realized_values if item < ZERO),
            profit_factor=profit_factor,
            expectancy=(ZERO if not realized_values else sum(realized_values, ZERO) / len(realized_values)),
            sharpe=risk["sharpe"],
            sortino=risk["sortino"],
            volatility=risk["volatility"],
        )
        merged_metadata = dict(metadata or {})
        if snapshots_raw:
            merged_metadata.setdefault("start", _iso(_get(snapshots_raw[0], "timestamp")))
            merged_metadata.setdefault("end", _iso(_get(snapshots_raw[-1], "timestamp")))
        merged_metadata.update(
            {
                "drawdown_peak_index": drawdown_peak,
                "drawdown_end_index": drawdown_end,
                "dual_leg_authority": True,
                "ordinary_trade_rows_fabricated": False,
            }
        )
        return HedgeBacktestArtifact(
            strategy_name=strategy_name,
            pairs=tuple(dict.fromkeys(str(item).upper() for item in pairs)),
            timeframe=str(timeframe),
            timerange=str(timerange),
            metrics=metrics,
            snapshots=snapshots,
            events=events,
            metadata=merged_metadata,
        )


@dataclass(frozen=True, slots=True)
class PortfolioBacktestComparison:
    artifacts: tuple[HedgeBacktestArtifact, ...]

    def ranked(self, *, metric: str = "total_return", descending: bool = True) -> tuple[HedgeBacktestArtifact, ...]:
        if metric not in HedgeBacktestMetrics.__dataclass_fields__:
            raise ValueError(f"unsupported comparison metric: {metric}")
        return tuple(
            sorted(
                self.artifacts,
                key=lambda item: getattr(item.metrics, metric),
                reverse=descending,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for rank, artifact in enumerate(self.ranked(), start=1):
            rows.append(
                {
                    "rank": rank,
                    "strategy": artifact.strategy_name,
                    "pairs": list(artifact.pairs),
                    "total_return": str(artifact.metrics.total_return),
                    "max_drawdown": str(artifact.metrics.max_drawdown),
                    "sharpe": artifact.metrics.sharpe,
                    "sortino": artifact.metrics.sortino,
                    "result_sha256": artifact.to_dict()["result_sha256"],
                }
            )
        return {"schema": "hedge-backtest-comparison-v1", "results": rows}
