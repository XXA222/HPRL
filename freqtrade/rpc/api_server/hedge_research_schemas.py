"""Pydantic contracts for the local-only Hedge research API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ResearchKindLiteral = Literal[
    "BACKTEST",
    "OPTIMIZATION",
    "ML_TRAIN",
    "ML_EVAL",
    "RL_TRAIN",
    "RL_EVAL",
]


class ResearchBudgetSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_seconds: int = Field(default=3600, ge=1, le=604800)
    max_trials: int = Field(default=100, ge=1, le=1_000_000)
    max_workers: int = Field(default=1, ge=1, le=256)
    max_artifact_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=8 * 1024**3)


class ResearchSubmitSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ResearchKindLiteral
    name: str = Field(min_length=1, max_length=96)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=32)
    priority: int = Field(default=50, ge=0, le=100)
    budget: ResearchBudgetSchema = Field(default_factory=ResearchBudgetSchema)
    auto_execute: bool = False


class ResearchProgressSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    progress: float = Field(ge=0.0, le=1.0)
    message: str = Field(default="", max_length=1000)


class ResearchMetricSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=96)
    value: float
    step: int | None = Field(default=None, ge=0)


class ResearchCompleteSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any]


class BacktestAnalyzeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equity: list[float] = Field(min_length=2, max_length=2_000_000)
    benchmark_equity: list[float] | None = Field(default=None, max_length=2_000_000)
    periods_per_year: int = Field(default=365 * 24 * 60, ge=1)


class ResearchObjectiveSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=96)
    weight: float = Field(default=1.0, gt=0.0)
    maximize: bool = True


class OptimizationRankSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, float]] = Field(min_length=1, max_length=100_000)
    objectives: list[ResearchObjectiveSchema] = Field(min_length=1, max_length=32)


class MLEvaluateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["regression", "binary"]
    actual: list[float]
    predicted: list[float]
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


class RLEvaluateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewards: list[float] = Field(min_length=1, max_length=1_000_000)
    drawdowns: list[float] | None = Field(default=None, max_length=1_000_000)


class ResearchCommandPlanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ResearchKindLiteral
    config_path: str = Field(min_length=1, max_length=4096)
    strategy: str = Field(min_length=1, max_length=128)
    timerange: str = Field(default="", max_length=128)
    trials: int | None = Field(default=None, ge=1, le=1_000_000)
    workers: int | None = Field(default=None, ge=1, le=256)


class ResumeTrainingSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_execute: bool = True
    name: str | None = Field(default=None, min_length=1, max_length=96)


class WalkForwardPlanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str = Field(pattern=r"^[0-9]{8}$")
    end: str = Field(pattern=r"^[0-9]{8}$")
    train_days: int = Field(ge=1, le=3650)
    eval_days: int = Field(ge=1, le=3650)
    step_days: int | None = Field(default=None, ge=1, le=3650)
    expanding: bool = False
    max_folds: int = Field(default=100, ge=1, le=1000)


class WalkForwardSubmitSchema(WalkForwardPlanSchema):
    kind: ResearchKindLiteral
    name: str = Field(min_length=1, max_length=96)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=32)
    priority: int = Field(default=50, ge=0, le=100)
    budget: ResearchBudgetSchema = Field(default_factory=ResearchBudgetSchema)
    auto_execute: bool = True


class PromotionPolicySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_sharpe: float | None = None
    max_drawdown: float | None = None
    min_reward: float | None = None
    max_loss: float | None = None
    min_profit: float | None = None
    require_model_files: bool = True

class OptimizationReplaySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timerange: str = Field(default="", max_length=128)
    auto_execute: bool = True
    name: str | None = Field(default=None, min_length=1, max_length=96)

class OptimizationReplayBatchSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=5, ge=1, le=20)
    timerange: str = Field(default="", max_length=128)
    auto_execute: bool = True

class ResearchPipelineSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=96)
    config_path: str = Field(min_length=1, max_length=4096)
    strategy: str = Field(min_length=1, max_length=128)
    optimization_timerange: str = Field(min_length=1, max_length=128)
    oos_timerange: str = Field(min_length=1, max_length=128)
    training_kind: Literal["ML_TRAIN", "RL_TRAIN"] = "RL_TRAIN"
    training_device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_threads: int = Field(default=4, ge=1, le=256)
    walk_forward_start: str = Field(pattern=r"^[0-9]{8}$")
    walk_forward_end: str = Field(pattern=r"^[0-9]{8}$")
    train_days: int = Field(default=60, ge=1, le=3650)
    eval_days: int = Field(default=15, ge=1, le=3650)
    step_days: int = Field(default=15, ge=1, le=3650)
    trials: int = Field(default=100, ge=1, le=1_000_000)
    workers: int = Field(default=1, ge=1, le=256)
    top_n: int = Field(default=5, ge=1, le=20)
    max_folds: int = Field(default=50, ge=1, le=1000)
    expanding: bool = False
    continual_learning: bool = True
    require_training_approval: bool = False
    priority: int = Field(default=50, ge=0, le=100)
    max_seconds: int = Field(default=14_400, ge=1, le=604800)
    max_artifact_bytes: int = Field(default=512 * 1024 * 1024, ge=1, le=8 * 1024**3)
    oos_metric: str = Field(default="auto", min_length=1, max_length=96)
    min_oos_success_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    walk_forward_metric: str = Field(default="sharpe", min_length=1, max_length=96)
    min_walk_forward_success_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    stability_penalty: float = Field(default=0.5, ge=0.0, le=100.0)
    max_stage_retries: int = Field(default=1, ge=0, le=10)
    training_parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=32)
    promotion_policy: PromotionPolicySchema = Field(default_factory=PromotionPolicySchema)
    auto_start: bool = True
