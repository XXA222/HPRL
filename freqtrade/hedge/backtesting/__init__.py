"""Deterministic Hedge backtesting and parameter-optimization toolkit."""

from .advanced_metrics import (
    compound_annual_growth_rate,
    conditional_value_at_risk,
    exposure_ratio,
    omega_ratio,
    profit_factor,
    recovery_factor,
    tail_ratio,
    trade_expectancy,
    ulcer_index,
    win_rate,
)
from .contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
    OptimizationSummary,
    SearchMethod,
    SplitMode,
)
from .data_quality import DataQualityReport, build_data_quality_report
from .dataset import (
    build_dataset,
    dataset_fingerprint,
    slice_dataset,
    validate_event_stream,
)
from .runner import HedgeBacktestRunner, objective_score, planner_config_with_overrides


__all__ = [
    "BacktestDataset",
    "BacktestEvaluation",
    "Candidate",
    "DataQualityReport",
    "EngineConfig",
    "HedgeBacktestRunner",
    "ObjectiveConfig",
    "OptimizationSummary",
    "SearchMethod",
    "SplitMode",
    "build_data_quality_report",
    "build_dataset",
    "compound_annual_growth_rate",
    "conditional_value_at_risk",
    "dataset_fingerprint",
    "exposure_ratio",
    "objective_score",
    "omega_ratio",
    "planner_config_with_overrides",
    "profit_factor",
    "recovery_factor",
    "slice_dataset",
    "tail_ratio",
    "trade_expectancy",
    "ulcer_index",
    "validate_event_stream",
    "win_rate",
]
