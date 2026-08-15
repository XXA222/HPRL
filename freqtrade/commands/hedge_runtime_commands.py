"""Safe local Hedge backtest and durable Paper command entry points."""

from __future__ import annotations

import json
import logging
import signal
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def _config_decimal(value: object, *, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise OperationalException(f"{field_name} must be a finite decimal") from exc
    if not result.is_finite():
        raise OperationalException(f"{field_name} must be a finite decimal")
    return result


def start_hedge_backtesting(args: dict[str, Any]) -> None:
    """Run the shared Hedge planner/matcher on normal Freqtrade historical data."""

    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.optimize.hedge_backtesting import run_freqtrade_hedge_backtest

    config = setup_optimize_configuration(args, RunMode.BACKTEST)
    raw_export = args.get("hedge_export_filename")
    export_path = None if raw_export is None else Path(str(raw_export))
    run = run_freqtrade_hedge_backtest(
        config,
        export_path=export_path,
        export_events=bool(args.get("hedge_export_events", False)),
    )
    native_artifact = getattr(run, "native_artifact", None)
    summary = {
        "status": "PASS",
        "mode": "hedge-backtesting",
        "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
        "strategy": run.strategy,
        "pair": run.dataset.pair,
        "timeframe": run.dataset.timeframe,
        "start": run.dataset.start.isoformat(),
        "end": run.dataset.end.isoformat(),
        "bars": run.dataset.bar_count,
        "signals": run.dataset.signal_count,
        "funding_events": run.dataset.funding_count,
        "market_rule_source": run.market_rule_source,
        "market_rule_version": run.market_rule_version,
        "report": {key: str(value) for key, value in run.result.report.items()},
        "result_file": str(run.export_path),
        "artifact_sha256": str(getattr(run, "artifact_sha256", "")),
        "result_fingerprint": str(getattr(run, "result_fingerprint", "")),
        "native_schema": str(getattr(native_artifact, "schema", "")),
        "freqtrade_projection_embedded": native_artifact is not None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_paper_runtime_config(config: dict[str, Any]) -> None:  # noqa: C901
    from freqtrade.hedge.config import validate_hedge_config
    from freqtrade.hedge.errors import HedgeDataError
    from freqtrade.hedge.paper_config import PaperFundingSource, PaperOhlcvSource

    try:
        hedge = validate_hedge_config(config)
    except (HedgeDataError, TypeError, ValueError) as exc:
        raise OperationalException(f"invalid Hedge Paper configuration: {exc}") from exc
    paper = hedge.paper
    if not hedge.enabled or paper is None:
        raise OperationalException("hedge-paper requires hedge_mode_enabled=true")
    if hedge.operation_mode != "paper":
        raise OperationalException("hedge-paper requires hedge.operation_mode='paper'")
    if not bool(config.get("dry_run", False)):
        raise OperationalException("hedge-paper requires dry_run=true")
    if not hedge.read_only or hedge.live_trading_enabled:
        raise OperationalException(
            "hedge-paper requires read_only=true and live_trading_enabled=false"
        )
    pair = hedge.managed_pair or str(config.get("managed_pair", "")).strip()
    exchange = config.get("exchange", {})
    whitelist = exchange.get("pair_whitelist", ()) if isinstance(exchange, dict) else ()
    if pair and whitelist and pair not in whitelist:
        raise OperationalException(
            "hedge managed_pair must be present in exchange.pair_whitelist"
        )
    wallet = config.get("dry_run_wallet")
    if wallet is not None and _config_decimal(
        wallet,
        field_name="dry_run_wallet",
    ) != paper.initial_balance:
        raise OperationalException(
            "dry_run_wallet must equal hedge.paper.initial_balance for deterministic Paper"
        )
    target_leverage = config.get("target_leverage")
    if target_leverage is None:
        raw_hedge = config.get("hedge", {})
        target_leverage = raw_hedge.get("target_leverage") if isinstance(raw_hedge, dict) else None
    if target_leverage is not None and _config_decimal(
        target_leverage,
        field_name="target_leverage",
    ) != paper.leverage:
        raise OperationalException(
            "target_leverage must equal hedge.paper.leverage"
        )
    max_open = config.get("max_open_trades")
    if max_open not in (None, 1):
        raise OperationalException(
            "single-pair hedge-paper currently requires max_open_trades=1"
        )
    if not paper.ephemeral:
        db_url = str(config.get("db_url", "")).strip()
        if not db_url or db_url in {"sqlite://", "sqlite:///:memory:"} or ":memory:" in db_url:
            raise OperationalException(
                "durable hedge-paper requires a file SQLite or PostgreSQL db_url"
            )
        if paper.state_backend != "sql":
            raise OperationalException(
                "durable hedge-paper requires hedge.paper.state_backend='sql'"
            )
        if paper.ohlcv_source is not PaperOhlcvSource.DATAPROVIDER:
            raise OperationalException(
                "durable hedge-paper requires DataProvider OHLCV"
            )
        if paper.funding_source is not PaperFundingSource.EXCHANGE:
            raise OperationalException(
                "durable hedge-paper requires exchange funding events"
            )
        if not paper.account_events_enabled:
            raise OperationalException(
                "durable hedge-paper requires account_events_enabled=true"
            )


def start_hedge_paper(args: dict[str, Any]) -> int:
    """Start the real-data, SQL-durable Paper runtime with hard safety checks."""

    from freqtrade.configuration import Configuration
    from freqtrade.worker import Worker

    config = Configuration(args, None).get_config()
    _validate_paper_runtime_config(config)

    def term_handler(signum, frame):
        raise KeyboardInterrupt()

    worker = None
    try:
        signal.signal(signal.SIGTERM, term_handler)
        worker = Worker(args, config=config)
        worker.run()
    finally:
        if worker is not None:
            logger.info("hedge-paper worker exiting")
            worker.exit()
    return 0

def start_hedge_research_optimize(args: dict[str, Any]) -> None:
    """Run resumable research optimization while preserving native Hyperopt."""

    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.hedge.optimization.freqtrade_adapter import run_freqtrade_hedge_optimization

    config = setup_optimize_configuration(args, RunMode.BACKTEST)
    hedge = config.setdefault("hedge", {})
    if not isinstance(hedge, dict):
        raise OperationalException("hedge must be a JSON object")
    optimization = hedge.setdefault("optimization", {})
    if not isinstance(optimization, dict):
        raise OperationalException("hedge.optimization must be a JSON object")
    overrides = {
        "study_name": args.get("hedge_study_name"),
        "trials": args.get("hedge_trials"),
        "workers": args.get("hedge_workers"),
        "sampler": args.get("hedge_sampler"),
        "output_directory": args.get("hedge_optimization_output"),
    }
    for key, value in overrides.items():
        if value is not None:
            optimization[key] = value
    try:
        run = run_freqtrade_hedge_optimization(config)
    except ValueError as exc:
        raise OperationalException(
            f"invalid Hedge optimization configuration: {exc}"
        ) from exc
    status_counts: dict[str, int] = {}
    for trial in run.result.trials:
        status_counts[trial.status.value] = status_counts.get(trial.status.value, 0) + 1
    summary = {
        "status": "PASS",
        "mode": "hedge-research-optimize",
        "study_name": run.result.study_name,
        "study_fingerprint": run.result.study_fingerprint,
        "dataset_fingerprint": run.result.dataset_fingerprint,
        "pair": run.dataset_pair,
        "timeframe": run.dataset_timeframe,
        "start": run.dataset_start.isoformat(),
        "end": run.dataset_end.isoformat(),
        "bars": run.dataset_bar_count,
        "trial_count": len(run.result.trials),
        "resumed_trials": run.result.resumed_trials,
        "status_counts": status_counts,
        "best_trial_id": run.result.best_trial_id,
        "pareto_trial_ids": list(run.result.pareto_trial_ids),
        "summary_json": str(run.artifacts.summary_json),
        "trials_csv": str(run.artifacts.trials_csv),
        "pareto_json": str(run.artifacts.pareto_json),
        "best_parameters_json": (
            None
            if run.artifacts.best_parameters_json is None
            else str(run.artifacts.best_parameters_json)
        ),
        "manifest_json": str(run.artifacts.manifest_json),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

