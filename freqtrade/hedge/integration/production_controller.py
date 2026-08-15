"""Lifecycle adapter that runs the R3.1 main loop from Freqtrade analyzed candles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from uuid import uuid4
from datetime import UTC, datetime
from typing import Any, Mapping

from freqtrade.hedge.integration.production_market import build_production_market_snapshot
from freqtrade.hedge.integration.production_main_loop import (
    HedgeMainLoopCycle,
    ProductionEquivalentHedgeMainLoop,
    RecoveryReport,
)

from .coordinator import HedgeRuntimeCoordinator
from .main_loop_config import ProductionMainLoopConfig
from .production_context import PlanningContextEvidence, ReadonlyPlanningContextBuilder


@dataclass(frozen=True, slots=True)
class ProductionControllerCycle:
    processed: bool
    reason: str
    candle_close_time: datetime | None = None
    main_loop: HedgeMainLoopCycle | None = None
    context_evidence: PlanningContextEvidence | None = None


class ProductionHedgeController:
    """Attach the fail-closed production-equivalent loop to the common bot cycle."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        loop: ProductionEquivalentHedgeMainLoop,
        loop_config: ProductionMainLoopConfig,
        context_builder: ReadonlyPlanningContextBuilder,
        readonly_coordinator: HedgeRuntimeCoordinator,
        exchange: Any,
        signal_provider: Any,
        timeframe: str,
        decision_cursor_store: Any | None = None,
    ) -> None:
        self.config = config
        self.loop = loop
        self.loop_config = loop_config
        self.context_builder = context_builder
        self.readonly_coordinator = readonly_coordinator
        self.exchange = exchange
        self.signal_provider = signal_provider
        self.timeframe = str(timeframe)
        self.decision_cursor_store = decision_cursor_store
        self._cursor_owner = f"production-controller:{uuid4().hex}"
        self._started = False
        self._last_candle_close: datetime | None = None
        self._recovery: RecoveryReport | None = None

    @property
    def recovery_report(self) -> RecoveryReport | None:
        return self._recovery

    def start(self) -> None:
        if self._started:
            return
        if not self.readonly_coordinator.started:
            raise RuntimeError("production main loop requires the read-only coordinator")
        if self.loop_config.recover_on_start:
            pair = self._managed_pair()
            self._recovery = self.loop.recover_pending(symbol=pair)
            if self._recovery.errors:
                raise RuntimeError("production main-loop recovery failed")
        self._started = True

    def stop(self) -> None:
        self._started = False

    def after_strategy_analyze(self) -> ProductionControllerCycle:
        if not self._started:
            self.start()
        snapshot = self.readonly_coordinator.readonly_runtime.snapshot()
        health = snapshot.direction2_health
        blockers: list[str] = []
        if self.loop_config.require_rest_fresh and not health.rest_fresh:
            blockers.append("REST_NOT_FRESH")
        if self.loop_config.require_stream_fresh and not (
            health.stream_connected and health.stream_fresh
        ):
            blockers.append("STREAM_NOT_FRESH")
        if self.loop_config.require_reconciliation_consistent and not health.reconciliation_consistent:
            blockers.append("RECONCILIATION_NOT_CONSISTENT")
        if not health.clock_synchronized:
            blockers.append("CLOCK_NOT_SYNCHRONIZED")
        if not health.configuration_valid:
            blockers.append("ACCOUNT_CONFIGURATION_INVALID")
        if blockers:
            return ProductionControllerCycle(False, ",".join(blockers))

        pair = self._managed_pair()
        signal = self.signal_provider.signals(pair, self.timeframe)
        if signal.candle is None:
            return ProductionControllerCycle(False, "NO_ANALYZED_CANDLE")
        if signal.candle_close_time > datetime.now(UTC):
            return ProductionControllerCycle(
                False,
                "CANDLE_NOT_CLOSED",
                candle_close_time=signal.candle_close_time,
            )
        if self._last_candle_close is not None and signal.candle_close_time <= self._last_candle_close:
            return ProductionControllerCycle(
                False,
                "NO_NEW_CLOSED_CANDLE",
                candle_close_time=signal.candle_close_time,
            )
        ticker_raw = self.exchange.fetch_ticker(pair)
        ticker = ticker_raw if isinstance(ticker_raw, Mapping) else None
        hedge = self.config.get("hedge", {})
        fallback = hedge.get("paper", {}) if isinstance(hedge, Mapping) else {}
        if not isinstance(fallback, Mapping):
            fallback = {}
        market = build_production_market_snapshot(
            exchange=self.exchange,
            pair=pair,
            candle=signal.candle,
            fallback=fallback,
            ticker=ticker,
        )
        account_view = self.readonly_coordinator.readonly_runtime.account_view()
        if str(account_view.account_id) != self.loop.account_id:
            raise RuntimeError("readonly account_id does not match main-loop account_id")
        built = self.context_builder.build(
            account_view=account_view,
            market=market,
            signal=signal,
        )
        decision_id = self._decision_id(signal)
        if self.decision_cursor_store is not None:
            claimed = self.decision_cursor_store.begin_decision_cursor(
                account_id=self.loop.account_id,
                symbol=pair,
                timeframe=self.timeframe,
                candle_close=signal.candle_close_time,
                decision_id=decision_id,
                lease_owner=self._cursor_owner,
            )
            if not claimed:
                return ProductionControllerCycle(
                    False,
                    "NO_NEW_CLOSED_CANDLE",
                    candle_close_time=signal.candle_close_time,
                )
        cycle = self.loop.run_cycle(
            built.context,
            strategy_allows_new_risk=bool(getattr(signal, "allow_new_risk", True)),
        )
        if cycle.errors:
            raise RuntimeError("production main-loop cycle contains action errors")
        if self.decision_cursor_store is not None:
            self.decision_cursor_store.complete_decision_cursor(
                account_id=self.loop.account_id,
                symbol=pair,
                timeframe=self.timeframe,
                decision_id=decision_id,
                lease_owner=self._cursor_owner,
            )
        self._last_candle_close = signal.candle_close_time
        return ProductionControllerCycle(
            True,
            "PROCESSED",
            candle_close_time=signal.candle_close_time,
            main_loop=cycle,
            context_evidence=built.evidence,
        )

    @staticmethod
    def _decision_id(signal: Any) -> str:
        candle = signal.candle
        target_net = getattr(signal, "target_net", None)
        target_net_ratio = getattr(signal, "target_net_ratio", None)
        payload = {
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "candle_close_time": signal.candle_close_time.isoformat(),
            "feature_timestamp": signal.feature_timestamp.isoformat(),
            "long_score": str(getattr(signal, "long_score", 0)),
            "short_score": str(getattr(signal, "short_score", 0)),
            "target_net": None if target_net is None else str(target_net),
            "target_net_ratio": None if target_net_ratio is None else str(target_net_ratio),
            "confidence": str(getattr(signal, "confidence", 1)),
            "risk_scale": str(getattr(signal, "risk_scale", 1)),
            "long_exposure_scale": str(getattr(signal, "long_exposure_scale", 1)),
            "short_exposure_scale": str(getattr(signal, "short_exposure_scale", 1)),
            "allow_new_risk": bool(getattr(signal, "allow_new_risk", True)),
            "regime": str(getattr(signal, "regime", "UNKNOWN")),
            "strategy_reason": str(
                getattr(signal, "strategy_reason", getattr(signal, "reason", "LEGACY_SIGNAL"))
            ),
            "model_version": str(getattr(signal, "model_version", "legacy")),
            "candle": None if candle is None else {
                "open": str(candle.open),
                "high": str(candle.high),
                "low": str(candle.low),
                "close": str(candle.close),
                "volume": None if candle.volume is None else str(candle.volume),
            },
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def _managed_pair(self) -> str:
        pair = str(self.config.get("managed_pair", "")).strip()
        if not pair:
            raise RuntimeError("managed_pair is required for the production main loop")
        return pair
