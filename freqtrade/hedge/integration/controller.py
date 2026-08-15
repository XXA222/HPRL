"""Lifecycle controller that plugs Hedge into Freqtrade's common data path."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from dataclasses import dataclass
from typing import Any, Mapping

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.hedge.runtime import HedgeRuntime

from .composition import HedgeCompositionRoot
from .market_data import (
    MarketRuleSnapshot,
    build_dataprovider_market_input,
    build_ticker_compat_market_input,
)
from .paper_events import ExchangeFundingEventProvider
from .paper_runtime import IntegratedPaperHedgeApplication, PaperCycleResult
from .production_controller import ProductionControllerCycle, ProductionHedgeController
from .signal_provider import FreqtradeStrategySignalProvider, SignalSnapshot
from freqtrade.hedge.paper_config import PaperFundingSource, PaperOhlcvSource

logger = logging.getLogger(__name__)


def effective_candle_max_age_seconds(configured: int, timeframe: str) -> int:
    """Resolve zero to a timeframe-aware two-candle freshness budget."""

    if configured < 0:
        raise ValueError("configured candle max age cannot be negative")
    return configured or max(60, 2 * timeframe_to_seconds(timeframe))


@dataclass(frozen=True, slots=True)
class HedgeControllerCycle:
    signal: SignalSnapshot | None
    market_rules: MarketRuleSnapshot | None
    paper_result: PaperCycleResult | None
    skipped: bool = False
    skip_reason: str | None = None
    processed_candles: int = 0


class HedgeController:
    """Coordinates source projections after Freqtrade has analyzed candles.

    This class intentionally has no access to Freqtrade's entry/exit methods.
    Exchange writes remain behind the separate, disabled Hedge execution port.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        runtime: HedgeRuntime,
        composition: HedgeCompositionRoot,
        exchange: Any,
        dataprovider: Any,
        strategy: Any,
        paper_application: IntegratedPaperHedgeApplication | None,
    ) -> None:
        self._config = config
        self.runtime = runtime
        self.composition = composition
        self.exchange = exchange
        self.dataprovider = dataprovider
        self.strategy = strategy
        self.paper_application = paper_application
        self.signal_provider = FreqtradeStrategySignalProvider(dataprovider, strategy)
        self.native_convergence = None
        self.model_gate = None
        self.producer_gate = None
        self.production_controller: ProductionHedgeController | None = None
        self.last_production_cycle: ProductionControllerCycle | None = None
        assembly = composition.production_main_loop_assembly
        if assembly is not None and assembly.cycle_owner == "PRODUCTION_CONTROLLER":
            coordinator = composition.readonly_coordinator
            if coordinator is None:
                raise RuntimeError("production main loop requires readonly coordinator")
            self.production_controller = ProductionHedgeController(
                config=config,
                loop=assembly.loop,
                loop_config=assembly.config,
                context_builder=assembly.context_builder,
                readonly_coordinator=coordinator,
                exchange=exchange,
                signal_provider=self.signal_provider,
                timeframe=strategy.timeframe,
            )
        self._funding_provider = None
        if (
            paper_application is not None
            and paper_application.paper_config.funding_source is PaperFundingSource.EXCHANGE
        ):
            last_funding_time = paper_application.last_funding_event_time
            self._funding_provider = ExchangeFundingEventProvider(
                exchange,
                symbol=runtime.config.managed_pair
                or str(config.get("managed_pair", "")),
                max_age_seconds=paper_application.paper_config.funding_max_age_seconds,
                poll_interval_seconds=(
                    paper_application.paper_config.funding_poll_interval_seconds
                ),
                initial_since_ms=(
                    None
                    if last_funding_time is None
                    else int(last_funding_time.timestamp() * 1000)
                ),
            )

    def start(self) -> None:
        self.runtime.heartbeat()
        self.composition.start()
        self.composition.refresh()
        if self.production_controller is not None:
            self.production_controller.start()

    def stop(self) -> None:
        if self.production_controller is not None:
            self.production_controller.stop()
        self.composition.stop()

    def bind_native_convergence(self, coordinator: object) -> None:
        """Bind official Freqtrade state/protection/capital adapters to all loops."""

        if not all(hasattr(coordinator, name) for name in ("new_risk_enabled", "admit", "status")):
            raise TypeError("native convergence coordinator is incompatible")
        self.native_convergence = coordinator
        if self.paper_application is not None:
            self.paper_application.add_new_risk_provider(coordinator.new_risk_enabled)
            self.paper_application.bind_order_admission_provider(coordinator.admit)
            transformer = getattr(coordinator, "transform_planner_intent", None)
            if callable(transformer):
                self.paper_application.bind_order_transformer(transformer)
            fill_observer = getattr(coordinator, "notify_order_filled", None)
            if callable(fill_observer):
                self.paper_application.add_fill_observer(fill_observer)
        assembly = self.composition.production_main_loop_assembly
        if assembly is not None:
            assembly.loop.bind_order_admission_provider(coordinator.admit)
            transformer = getattr(coordinator, "transform_planner_intent", None)
            if callable(transformer):
                assembly.loop.bind_order_transformer(transformer)

    def bind_model_gate(self, gate: object) -> None:
        if not callable(getattr(gate, "observe", None)):
            raise TypeError("model gate must provide observe(signal)")
        self.model_gate = gate

    def bind_producer_gate(self, gate: object) -> None:
        if not callable(getattr(gate, "observe", None)):
            raise TypeError("producer gate must provide observe(signal)")
        self.producer_gate = gate

    def _observe_native_signal(self, signal: SignalSnapshot) -> None:
        if self.model_gate is None and self.producer_gate is None:
            return
        from freqtrade.hedge.integration.candle_cursor import bar_fingerprint
        from freqtrade.hedge.native.freqai import HedgeSignalEnvelope

        if signal.candle is None:
            raise ValueError("native signal evidence requires an analyzed candle")
        manifest = getattr(self.model_gate, "manifest", None)
        feature_schema = (
            getattr(manifest, "feature_schema", None)
            or str(
                self._config.get("hedge", {})
                .get("freqai_native", {})
                .get("expected_feature_schema", "strategy-columns-v1")
            )
        )
        model_version = getattr(manifest, "model_version", None) or signal.model_version
        envelope = HedgeSignalEnvelope(
            pair=signal.symbol,
            timestamp=signal.candle_close_time,
            long_score=signal.long_score,
            short_score=signal.short_score,
            target_net_ratio=signal.target_net_ratio,
            target_gross_ratio=None,
            confidence=signal.confidence,
            risk_scale=signal.risk_scale,
            model_version=str(model_version),
            feature_schema=feature_schema,
            candle_fingerprint=bar_fingerprint(signal.candle.to_bar_event()),
            producer_id="freqtrade-strategy",
            metadata={
                "reason": signal.reason,
                "strategy_reason": signal.strategy_reason,
                "regime": signal.regime,
            },
        )
        if self.model_gate is not None:
            self.model_gate.observe(envelope)
        if self.producer_gate is not None:
            self.producer_gate.observe(envelope, received_at=datetime.now(UTC))

    def cancel_managed_orders(self) -> tuple[int, tuple[str, ...]]:
        """Cancel only Hedge-owned orders for stop, reload and process cleanup."""

        errors: list[str] = []
        canceled_count = 0
        assembly = self.composition.production_main_loop_assembly
        if assembly is not None and assembly.cycle_owner == "PRODUCTION_CONTROLLER":
            report = assembly.loop.cancel_managed_orders()
            canceled_count += len(report.canceled)
            errors.extend(
                f"{item.reference}:{item.error_type}:{item.message}" for item in report.errors
            )
        elif self.paper_application is not None:
            # In Paper-owned mode the application owns the planner-to-client mapping.
            # Calling the assembly loop here would miss those orders.
            canceled, paper_errors = self.paper_application.cancel_managed_orders()
            canceled_count += len(canceled)
            errors.extend(paper_errors)
        return canceled_count, tuple(errors)

    def native_status(self) -> dict[str, object] | None:
        if self.native_convergence is None:
            return None
        return self.native_convergence.status()

    def after_strategy_analyze(self) -> HedgeControllerCycle:
        """Refresh facts and process every unseen closed DataProvider candle.

        Fresh runs start at the latest analyzed candle. Recovered runs catch up
        every candle after the durable cursor, bounded by the typed Paper
        limits. This prevents open orders from skipping intermediate high/low
        paths after downtime.
        """

        self.runtime.heartbeat()
        self.composition.refresh()
        if self.production_controller is not None:
            self.last_production_cycle = self.production_controller.after_strategy_analyze()
        application = self.paper_application
        if application is None:
            return HedgeControllerCycle(signal=None, market_rules=None, paper_result=None)

        pair = self.runtime.config.managed_pair or str(self._config.get("managed_pair", ""))
        if not pair:
            self.runtime.halt("HEDGE_MANAGED_PAIR_MISSING")
            raise RuntimeError("Hedge managed_pair is missing")
        paper_config = application.paper_config
        previous_market = application.last_market
        try:
            signals = self.signal_provider.signals_since(
                pair,
                self.strategy.timeframe,
                after=None if previous_market is None else previous_market.timestamp,
                cursor_fingerprint=application.last_bar_fingerprint,
                max_catchup_candles=paper_config.max_catchup_candles,
                max_missing_candles=paper_config.max_missing_candles,
                reject_revised_candle=paper_config.reject_revised_candle,
            )
        except ValueError as exc:
            self.runtime.halt("PAPER_DATAPROVIDER_CURSOR_INVALID")
            raise RuntimeError("Paper DataProvider catch-up validation failed") from exc
        if not signals:
            return HedgeControllerCycle(
                signal=None,
                market_rules=None,
                paper_result=None,
                skipped=True,
                skip_reason="NO_NEW_CLOSED_CANDLE",
                processed_candles=0,
            )

        hedge = self._config.get("hedge", {})
        if not isinstance(hedge, Mapping):
            hedge = {}
        paper = hedge.get("paper", {})
        if not isinstance(paper, Mapping):
            paper = {}

        # Only the newest candle may use the current ticker for spread
        # refinement. Historical catch-up candles use their own close for bid
        # and ask so present-time prices cannot leak backwards.
        ticker = None
        try:
            raw_ticker = self.exchange.fetch_ticker(pair)
            ticker = raw_ticker if isinstance(raw_ticker, Mapping) else None
        except Exception as exc:
            logger.warning("Paper ticker spread refinement failed: %s", type(exc).__name__)

        latest = signals[-1]
        now = datetime.now(UTC)
        if latest.candle is not None:
            if paper_config.require_closed_candle and latest.candle.close_time > now:
                self.runtime.halt("PAPER_DATAPROVIDER_CANDLE_NOT_CLOSED")
                raise RuntimeError("Paper refused an unfinished DataProvider candle")
            age = max(0.0, (now - latest.candle.close_time).total_seconds())
            effective_max_age = effective_candle_max_age_seconds(
                paper_config.candle_max_age_seconds,
                latest.timeframe,
            )
            if age > effective_max_age:
                self.runtime.halt("PAPER_DATAPROVIDER_CANDLE_STALE")
                raise RuntimeError(
                    "Paper DataProvider latest candle is stale by "
                    f"{age:.1f} seconds (limit={effective_max_age}s)"
                )

        result: PaperCycleResult | None = None
        market_input = None
        funding_healthy = paper_config.funding_source is PaperFundingSource.NONE
        try:
            for index, signal in enumerate(signals):
                try:
                    self._observe_native_signal(signal)
                except Exception as exc:
                    self.runtime.halt(
                        f"HEDGE_NATIVE_SIGNAL_EVIDENCE_FAILED:{type(exc).__name__}"
                    )
                    raise RuntimeError("Hedge native signal evidence failed closed") from exc
                if paper_config.ohlcv_source is PaperOhlcvSource.DATAPROVIDER:
                    candle = signal.candle
                    if candle is None:
                        self.runtime.halt("PAPER_DATAPROVIDER_CANDLE_MISSING")
                        raise RuntimeError("Paper requires analyzed DataProvider OHLCV")
                    market_input = build_dataprovider_market_input(
                        exchange=self.exchange,
                        pair=pair,
                        candle=candle,
                        fallback=paper,
                        ticker=ticker if index == len(signals) - 1 else None,
                    )
                else:
                    if ticker is None:
                        self.runtime.halt("PAPER_TICKER_COMPAT_UNAVAILABLE")
                        raise RuntimeError("ticker_compat requires a ticker response")
                    market_input = build_ticker_compat_market_input(
                        exchange=self.exchange,
                        pair=pair,
                        ticker=ticker,
                        fallback=paper,
                        event_time=signal.candle_close_time,
                    )

                funding_events = ()
                funding_state = None
                if self._funding_provider is not None:
                    snapshot_state = getattr(self._funding_provider, "snapshot_state", None)
                    if callable(snapshot_state):
                        funding_state = snapshot_state()
                    try:
                        funding = self._funding_provider.collect(market_input.bar)
                        funding_events = funding.events
                        funding_healthy = self._funding_provider.healthy()
                    except Exception as exc:
                        self.runtime.halt(f"PAPER_FUNDING_SOURCE_FAILED:{type(exc).__name__}")
                        raise RuntimeError("Paper funding source failed closed") from exc

                try:
                    result = application.run_market_cycle(
                        market_input.market,
                        bar=market_input.bar,
                        funding_events=funding_events,
                        long_signal=signal.long_score,
                        short_signal=signal.short_score,
                        target_net_quantity=signal.target_net,
                        target_net_ratio=signal.target_net_ratio,
                        confidence=signal.confidence,
                        risk_scale=signal.risk_scale,
                        long_exposure_scale=signal.long_exposure_scale,
                        short_exposure_scale=signal.short_exposure_scale,
                        allow_new_risk=signal.allow_new_risk,
                        regime=signal.regime,
                        strategy_reason=signal.strategy_reason,
                        model_version=signal.model_version,
                        maker_fee_rate=market_input.rules.maker_fee_rate,
                        taker_fee_rate=market_input.rules.taker_fee_rate,
                    )
                except Exception:
                    restore_state = (
                        None
                        if self._funding_provider is None
                        else getattr(self._funding_provider, "restore_state", None)
                    )
                    if funding_state is not None and callable(restore_state):
                        restore_state(funding_state)
                    raise

            assert market_input is not None and result is not None
            application.publish_runtime(
                self.runtime,
                market_data_fresh=True,
                funding_source_healthy=funding_healthy,
                market_source=market_input.source,
            )
        except Exception as exc:
            self.runtime.halt(f"PAPER_CYCLE_FAILED:{type(exc).__name__}")
            raise

        logger.debug(
            "Hedge cycle source=%s rule_source=%s rule_version=%s "
            "signal_reason=%s processed_candles=%d",
            self.runtime.config.operation_mode,
            market_input.rules.source,
            market_input.rules.version,
            latest.reason,
            len(signals),
        )
        return HedgeControllerCycle(
            signal=latest,
            market_rules=market_input.rules,
            paper_result=result,
            processed_candles=len(signals),
        )
