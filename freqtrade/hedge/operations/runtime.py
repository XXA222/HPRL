"""Composition root for durable Dry-run operational health and readiness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .alerts import AlertManager, AlertSeverity
from .attribution import AttributionInput, PerformanceAttributor
from .breaker import DrawdownCircuitBreaker
from .common import ensure_aware
from .config import operations_config
from .dashboard import OperationsDashboardSnapshot
from .market import MarketDataHealthGate, MarketHealthInput
from .readiness import DryRunReadinessBuilder, ReadinessCheck
from .risk import PortfolioRiskMonitor, RiskLeg
from .session import RunSession, SessionStatus
from .state import AtomicRunStateStore
from .warmup import StrategyWarmupGate, WarmupRequirement


@dataclass(frozen=True, slots=True)
class OperationsCycleInput:
    timestamp: datetime
    symbol: str
    timeframe_seconds: int
    mark_price: Decimal
    index_price: Decimal | None
    equity: Decimal
    initial_equity: Decimal
    long_notional: Decimal
    short_notional: Decimal
    margin_used: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    funding_pnl: Decimal
    fees: Decimal
    slippage_cost: Decimal
    base_candles: int
    informative_candles: dict[str, int]
    observed_at: datetime | None = None
    order_count: int = 0
    fill_count: int = 0
    active_order_count: int = 0
    reconciliation_fresh: bool = False
    api_healthy: bool = False
    dashboard_healthy: bool = False


class DryRunOperationsRuntime:
    def __init__(
        self,
        *,
        account_id: str,
        symbols: tuple[str, ...],
        config: Mapping[str, Any],
        state_path: str | Path | None = None,
    ) -> None:
        operations = operations_config(config)
        candidate = RunSession.create(
            account_id=account_id,
            symbols=symbols,
            config=operations,
        ).transition(SessionStatus.RUNNING)

        self.state_store = None if state_path is None else AtomicRunStateStore(state_path)
        self.recovery_reasons: tuple[str, ...] = ()
        self.session = candidate
        self.total_orders = 0
        self.total_fills = 0

        if self.state_store is not None:
            recovered, extras, reasons = self.state_store.load_fail_closed()
            self.recovery_reasons = reasons
            if recovered is not None:
                identity_matches = (
                    recovered.account_id == candidate.account_id
                    and recovered.symbols == candidate.symbols
                    and recovered.config_sha256 == candidate.config_sha256
                )
                if identity_matches and recovered.status is SessionStatus.RUNNING:
                    self.session = recovered
                    try:
                        self.total_orders = max(0, int(extras.get("total_orders", 0)))
                        self.total_fills = max(0, int(extras.get("total_fills", 0)))
                    except (TypeError, ValueError):
                        self.recovery_reasons = ("STATE_COUNTERS_INVALID",)
                elif not identity_matches:
                    self.recovery_reasons = ("STATE_IDENTITY_MISMATCH",)
                else:
                    self.recovery_reasons = (
                        f"STATE_NOT_RESUMABLE:{recovered.status.value}",
                    )

        self.market = MarketDataHealthGate(
            max_age_seconds=int(operations.get("market_max_age_seconds", 90)),
            max_gap_candles=int(operations.get("market_max_gap_candles", 2)),
            max_divergence_bps=Decimal(
                str(operations.get("mark_index_max_divergence_bps", "50"))
            ),
        )
        self.warmup = StrategyWarmupGate(
            WarmupRequirement(
                int(operations.get("warmup_candles", 100)),
                tuple(
                    (str(key), int(value))
                    for key, value in dict(operations.get("informative_warmup", {})).items()
                ),
            )
        )
        self.risk = PortfolioRiskMonitor(
            max_gross_ratio=Decimal(str(operations.get("max_gross_ratio", "0.80"))),
            max_margin_ratio=Decimal(str(operations.get("max_margin_ratio", "0.55"))),
            max_net_ratio=Decimal(str(operations.get("max_net_ratio", "0.50"))),
        )
        self.breaker = DrawdownCircuitBreaker(
            warning=Decimal(str(operations.get("drawdown_warning", "0.05"))),
            pause=Decimal(str(operations.get("drawdown_pause", "0.10"))),
            kill=Decimal(str(operations.get("drawdown_kill", "0.20"))),
            recovery=Decimal(str(operations.get("drawdown_recovery", "0.03"))),
        )
        self.alerts = AlertManager()
        self.attributor = PerformanceAttributor()
        self.readiness = DryRunReadinessBuilder()
        self.latest: OperationsDashboardSnapshot | None = None

    def observe(self, item: OperationsCycleInput) -> OperationsDashboardSnapshot:
        timestamp = ensure_aware(item.timestamp)
        observed = ensure_aware(item.observed_at or item.timestamp)
        self.session, cycle_id = self.session.next_cycle(timestamp)

        market = self.market.evaluate(
            MarketHealthInput(
                item.symbol,
                timestamp,
                observed,
                item.timeframe_seconds,
                item.mark_price,
                item.index_price,
            )
        )
        warmup = self.warmup.evaluate(
            base_available=item.base_candles,
            informative_available=item.informative_candles,
        )
        risk = self.risk.snapshot(
            timestamp=timestamp,
            equity=item.equity,
            legs=(
                RiskLeg(
                    item.symbol,
                    item.long_notional,
                    item.short_notional,
                    item.margin_used,
                    item.unrealized_pnl,
                ),
            ),
        )
        breaker = self.breaker.evaluate(item.equity)
        attribution = self.attributor.calculate(
            AttributionInput(
                item.realized_pnl,
                item.unrealized_pnl,
                item.funding_pnl,
                item.fees,
                item.slippage_cost,
            ),
            equity_change=item.equity - item.initial_equity,
        )

        diagnostics = (
            list(self.recovery_reasons)
            + list(market.reasons)
            + list(warmup.missing)
            + list(risk.reasons)
        )
        if not market.ready:
            self.alerts.emit(
                key="MARKET_NOT_READY",
                severity=AlertSeverity.ERROR,
                message=",".join(market.reasons),
                at=timestamp,
            )
        if not risk.ready:
            self.alerts.emit(
                key="RISK_NOT_READY",
                severity=AlertSeverity.CRITICAL,
                message=",".join(risk.reasons),
                at=timestamp,
            )

        new_risk_enabled = (
            not self.recovery_reasons
            and market.ready
            and warmup.ready
            and risk.ready
            and breaker.new_risk_enabled
        )
        self.total_orders += max(0, int(item.order_count))
        self.total_fills += max(0, int(item.fill_count))

        quality_level, quality_state = self._quality_state(item, market.ready, warmup.ready)
        snapshot = OperationsDashboardSnapshot(
            timestamp,
            self.session.session_id,
            cycle_id,
            self.session.symbols,
            self.session.status.value,
            market.ready,
            warmup.ready,
            risk,
            breaker,
            attribution,
            self.alerts.active(),
            tuple(diagnostics),
            new_risk_enabled,
            last_candle_age_seconds=market.age_seconds,
            candle_gap_seconds=market.gap_seconds,
            strategy_cycle_count=self.session.cycle_sequence,
            order_count=self.total_orders,
            fill_count=self.total_fills,
            active_order_count=max(0, int(item.active_order_count)),
            reconciliation_fresh=bool(item.reconciliation_fresh),
            runtime_quality_level=quality_level,
            runtime_quality_state=quality_state,
        )
        self.latest = snapshot
        if self.state_store is not None:
            self.state_store.save(
                self.session,
                {
                    "latest": snapshot.summary(),
                    "total_orders": self.total_orders,
                    "total_fills": self.total_fills,
                },
            )
        return snapshot

    def _quality_state(
        self,
        item: OperationsCycleInput,
        market_ready: bool,
        warmup_ready: bool,
    ) -> tuple[int, str]:
        level = 2
        state = "RUNNING_UNVERIFIED"
        if market_ready:
            level, state = 3, "STRATEGY_STALE"
        if market_ready and warmup_ready:
            level, state = 4, "FUNCTION_UNVERIFIED"
        if market_ready and warmup_ready and self.total_orders > 0 and self.total_fills > 0:
            level, state = 5, "LEDGER_UNVERIFIED"
        if level >= 5 and item.reconciliation_fresh:
            level, state = 6, "UI_UNVERIFIED"
        if level >= 6 and item.api_healthy and item.dashboard_healthy:
            level, state = 7, "CREDIBLE"
        if not market_ready:
            state = "DATA_STALE"
        return level, state

    def certificate(self, *, at: datetime):
        latest = self.latest

        def check(name: str, passed: bool, failure: str) -> ReadinessCheck:
            return ReadinessCheck(name, passed, "" if passed else failure)

        checks = (
            check(
                "SESSION_RUNNING",
                self.session.status is SessionStatus.RUNNING,
                self.session.status.value,
            ),
            check(
                "STATE_RECOVERY",
                not self.recovery_reasons,
                ",".join(self.recovery_reasons),
            ),
            check("CYCLE_OBSERVED", latest is not None, "NO_CYCLE"),
            check(
                "MARKET_READY",
                bool(latest and latest.market_ready),
                "MARKET_NOT_READY",
            ),
            check(
                "WARMUP_READY",
                bool(latest and latest.warmup_ready),
                "WARMUP_NOT_READY",
            ),
            check(
                "RISK_READY",
                bool(latest and latest.risk and latest.risk.ready),
                "RISK_NOT_READY",
            ),
            ReadinessCheck("MAINNET_WRITES_LOCKED", True, ""),
        )
        return self.readiness.build(
            session_id=self.session.session_id,
            checks=checks,
            at=at,
        )
