"""immutable operations snapshot for API/UI projection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .alerts import AlertRecord
from .attribution import PerformanceAttribution
from .breaker import BreakerDecision
from .common import ensure_aware
from .risk import PortfolioRiskSnapshot


@dataclass(frozen=True, slots=True)
class OperationsDashboardSnapshot:
    generated_at: datetime
    session_id: str
    cycle_id: str | None
    symbols: tuple[str, ...]
    state: str
    market_ready: bool
    warmup_ready: bool
    risk: PortfolioRiskSnapshot | None
    breaker: BreakerDecision | None
    attribution: PerformanceAttribution | None
    active_alerts: tuple[AlertRecord, ...]
    diagnostics: tuple[str, ...]
    new_risk_enabled: bool
    last_candle_age_seconds: Decimal | None = None
    candle_gap_seconds: Decimal | None = None
    strategy_cycle_count: int = 0
    order_count: int = 0
    fill_count: int = 0
    active_order_count: int = 0
    reconciliation_fresh: bool = False
    runtime_quality_level: int = 1
    runtime_quality_state: str = "RUNNING_UNVERIFIED"

    def __post_init__(self) -> None:
        ensure_aware(self.generated_at)

    @property
    def ready(self) -> bool:
        return (
            self.state == "RUNNING"
            and self.market_ready
            and self.warmup_ready
            and (self.risk is None or self.risk.ready)
            and self.new_risk_enabled
        )

    def summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "state": self.state,
            "symbols": self.symbols,
            "ready": self.ready,
            "new_risk_enabled": self.new_risk_enabled,
            "market_ready": self.market_ready,
            "warmup_ready": self.warmup_ready,
            "gross_ratio": None if self.risk is None else str(self.risk.gross_ratio),
            "margin_ratio": None if self.risk is None else str(self.risk.margin_ratio),
            "drawdown": None if self.breaker is None else str(self.breaker.drawdown),
            "net_pnl": None if self.attribution is None else str(self.attribution.net_pnl),
            "active_alert_count": len(self.active_alerts),
            "diagnostics": self.diagnostics,
            "generated_at": self.generated_at.isoformat(),
            "last_candle_age_seconds": (
                None if self.last_candle_age_seconds is None else str(self.last_candle_age_seconds)
            ),
            "candle_gap_seconds": (
                None if self.candle_gap_seconds is None else str(self.candle_gap_seconds)
            ),
            "strategy_cycle_count": self.strategy_cycle_count,
            "order_count": self.order_count,
            "fill_count": self.fill_count,
            "active_order_count": self.active_order_count,
            "reconciliation_fresh": self.reconciliation_fresh,
            "runtime_quality_level": self.runtime_quality_level,
            "runtime_quality_state": self.runtime_quality_state,
        }
