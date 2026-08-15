"""Local-only API for Hedge backtest/optimization/ML/RL research workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from freqtrade.hedge.control.auth import HedgePrincipal, HedgeRole
from freqtrade.hedge.research.backtest import benchmark_excess, summarize_equity
from freqtrade.hedge.research.command_plan import build_command_plan
from freqtrade.hedge.research.contracts import ResearchBudget, ResearchKind, ResearchRequest
from freqtrade.hedge.research.ml import binary_metrics, regression_metrics
from freqtrade.hedge.research.optimization import ObjectiveWeight, pareto_front, rank_trials
from freqtrade.hedge.research.pipeline import ResearchPipelineSpec
from freqtrade.hedge.research.promotion import PromotionPolicy
from freqtrade.hedge.research.rl import episode_summary
from freqtrade.hedge.research.service import HedgeResearchService
from freqtrade.rpc.api_server.hedge_auth import require_role
from freqtrade.rpc.api_server.hedge_research_schemas import (
    BacktestAnalyzeSchema,
    MLEvaluateSchema,
    OptimizationRankSchema,
    OptimizationReplayBatchSchema,
    OptimizationReplaySchema,
    PromotionPolicySchema,
    ResearchCommandPlanSchema,
    ResearchCompleteSchema,
    ResearchMetricSchema,
    ResearchPipelineSchema,
    ResearchProgressSchema,
    ResearchSubmitSchema,
    ResumeTrainingSchema,
    RLEvaluateSchema,
    WalkForwardPlanSchema,
    WalkForwardSubmitSchema,
)


def create_hedge_research_router(
    *,
    service: HedgeResearchService,
    principal_dependency: Callable[..., HedgePrincipal],
) -> APIRouter:
    router = APIRouter(prefix="/hedge/research", tags=["hedge-research"])
    viewer = require_role(HedgeRole.VIEWER, principal_dependency)
    operator = require_role(HedgeRole.OPERATOR, principal_dependency)
    _register_read_routes(router, service=service, viewer=viewer)
    _register_job_routes(router, service=service, viewer=viewer, operator=operator)
    _register_analysis_routes(router, viewer=viewer)
    return router


def _register_read_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    viewer: Callable[..., HedgePrincipal],
) -> None:
    @router.get("/capabilities")
    def capabilities(_: HedgePrincipal = Depends(viewer)) -> dict[str, Any]:
        return service.capabilities()

    @router.get("/dashboard")
    def dashboard(_: HedgePrincipal = Depends(viewer)) -> dict[str, Any]:
        payload = service.dashboard()
        payload["executor"] = service.executor_status()
        return payload

    @router.get("/executor")
    def executor(_: HedgePrincipal = Depends(viewer)) -> dict[str, Any]:
        return service.executor_status()

    @router.get("/jobs")
    def jobs(limit: int = 200, _: HedgePrincipal = Depends(viewer)) -> dict[str, Any]:
        bounded = max(1, min(int(limit), 2000))
        rows = service.list_jobs(limit=bounded)
        return {"jobs": rows, "count": len(rows)}

    @router.get("/jobs/{job_id}")
    def job(job_id: str, _: HedgePrincipal = Depends(viewer)) -> dict[str, Any]:
        try:
            payload = service.get(job_id)
            payload["runtime"] = service.runtime(job_id)
            return payload
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/jobs/{job_id}/runtime")
    def runtime(job_id: str, _: HedgePrincipal = Depends(viewer)) -> dict[str, Any]:
        try:
            return service.runtime(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/jobs/{job_id}/log")
    def log_tail(
        job_id: str,
        lines: int = 300,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.log_tail(job_id, lines=lines)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/jobs/{job_id}/optimization-replays")
    def optimization_replays(
        job_id: str,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            service.get(job_id)
            return service.optimization_replays(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/experiments")
    def experiments(
        limit: int = 500,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        rows = service.list_experiments(limit=max(1, min(int(limit), 5000)))
        return {"experiments": rows, "count": len(rows)}

    @router.get("/experiments/leaderboard")
    def leaderboard(
        metric: str = "sharpe",
        maximize: bool = True,
        limit: int = 50,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        rows = service.leaderboard(
            metric=metric.strip(),
            maximize=bool(maximize),
            limit=max(1, min(int(limit), 500)),
        )
        return {"metric": metric, "maximize": maximize, "rows": rows, "count": len(rows)}

    @router.get("/experiments/{experiment_id}")
    def experiment(
        experiment_id: str,
        refresh: bool = False,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.experiment(experiment_id, refresh=refresh)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/experiments/{experiment_id}/tensorboard")
    def experiment_tensorboard(
        experiment_id: str,
        max_points: int = 1000,
        max_tags: int = 100,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.experiment_tensorboard(
                experiment_id,
                max_points_per_tag=max(10, min(int(max_points), 10000)),
                max_tags=max(1, min(int(max_tags), 500)),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/promotions")
    def promotions(
        limit: int = 200,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        rows = service.list_promotions(limit=max(1, min(int(limit), 2000)))
        return {"promotions": rows, "count": len(rows)}

    @router.get("/promotions/{promotion_id}")
    def promotion(
        promotion_id: str,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.promotion(promotion_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/pipelines")
    def pipelines(
        limit: int = 100,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        rows = service.list_pipelines(limit=max(1, min(int(limit), 1000)))
        return {"pipelines": rows, "count": len(rows)}

    @router.get("/pipelines/{pipeline_id}")
    def pipeline(
        pipeline_id: str,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.pipeline(pipeline_id)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/models")
    def models(
        limit: int = 200,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        rows = service.model_catalog(limit=max(1, min(int(limit), 1000)))
        return {"models": rows, "count": len(rows)}

    @router.get("/walk-forward")
    def walk_forward_groups(
        limit: int = 100,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        rows = service.list_walk_forward_groups(limit=max(1, min(int(limit), 1000)))
        return {"groups": rows, "count": len(rows)}

    @router.post("/walk-forward/plan")
    def walk_forward_plan(
        request: WalkForwardPlanSchema,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.plan_walk_forward(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/command-plan")
    def command_plan(
        request: ResearchCommandPlanSchema,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, object]:
        try:
            plan = build_command_plan(
                ResearchKind(request.kind),
                config_path=Path(request.config_path),
                strategy=request.strategy,
                timerange=request.timerange,
                trials=request.trials,
                workers=request.workers,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return plan.to_dict()


def _register_pipeline_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    operator: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/pipelines")
    def submit_pipeline(
        request: ResearchPipelineSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            payload = request.model_dump()
            policy = PromotionPolicy(**payload.pop("promotion_policy"))
            auto_start = bool(payload.pop("auto_start", True))
            payload["training_kind"] = ResearchKind(
                str(payload["training_kind"])
            )
            payload["tags"] = tuple(
                str(item)
                for item in payload.get("tags", ())
            )
            spec = ResearchPipelineSpec(
                **payload,
                promotion_policy=policy,
            )
            return service.submit_pipeline(spec, auto_start=auto_start)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/pipelines/{pipeline_id}/start")
    def start_pipeline(
        pipeline_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _pipeline_call(service.start_pipeline, pipeline_id)

    @router.post("/pipelines/{pipeline_id}/pause")
    def pause_pipeline(
        pipeline_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _pipeline_call(service.pause_pipeline, pipeline_id)

    @router.post("/pipelines/{pipeline_id}/resume")
    def resume_pipeline(
        pipeline_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _pipeline_call(service.resume_pipeline, pipeline_id)

    @router.post("/pipelines/{pipeline_id}/approve-training")
    def approve_pipeline_training(
        pipeline_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _pipeline_call(
            service.approve_pipeline_training,
            pipeline_id,
        )

    @router.post("/pipelines/{pipeline_id}/cancel")
    def cancel_pipeline(
        pipeline_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _pipeline_call(service.cancel_pipeline, pipeline_id)

    @router.post("/pipelines/{pipeline_id}/reconsider-promotion")
    def reconsider_pipeline_promotion(
        pipeline_id: str,
        request: PromotionPolicySchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.reconsider_pipeline_promotion(
                pipeline_id,
                PromotionPolicy(**request.model_dump()),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/pipelines/{pipeline_id}/retry")
    def retry_pipeline(
        pipeline_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _pipeline_call(service.retry_pipeline, pipeline_id)


def _register_job_submission_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    operator: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/jobs")
    def submit(
        request: ResearchSubmitSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        budget = ResearchBudget(**request.budget.model_dump())
        research_request = ResearchRequest(
            kind=ResearchKind(request.kind),
            name=request.name,
            parameters=dict(request.parameters),
            tags=tuple(request.tags),
            priority=request.priority,
            budget=budget,
        )
        payload = service.submit(research_request)
        if request.auto_execute:
            try:
                payload["runtime"] = service.execute(
                    str(payload["job_id"])
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail=str(exc),
                ) from exc
        return payload

    @router.post("/jobs/{job_id}/replay-best")
    def replay_best(
        job_id: str,
        request: OptimizationReplaySchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.replay_best_optimization(
                job_id,
                timerange=request.timerange,
                auto_execute=request.auto_execute,
                name=request.name,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/replay-top")
    def replay_top(
        job_id: str,
        request: OptimizationReplayBatchSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.replay_top_optimization(
                job_id,
                limit=request.limit,
                timerange=request.timerange,
                auto_execute=request.auto_execute,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/execute")
    def execute(
        job_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.execute(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _register_executor_control_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    operator: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/executor/pause")
    def pause_executor(
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.pause_executor()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/executor/resume")
    def resume_executor(
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.resume_executor()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/experiments/{experiment_id}/resume")
    def resume_training(
        experiment_id: str,
        request: ResumeTrainingSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.resume_training(
                experiment_id,
                auto_execute=request.auto_execute,
                name=request.name,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _register_walk_forward_promotion_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    viewer: Callable[..., HedgePrincipal],
    operator: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/walk-forward/submit")
    def walk_forward_submit(
        request: WalkForwardSubmitSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            research_request = ResearchRequest(
                kind=ResearchKind(request.kind),
                name=request.name,
                parameters=dict(request.parameters),
                tags=tuple(request.tags),
                priority=request.priority,
                budget=ResearchBudget(
                    **request.budget.model_dump()
                ),
            )
            return service.submit_walk_forward(
                research_request,
                start=request.start,
                end=request.end,
                train_days=request.train_days,
                eval_days=request.eval_days,
                step_days=request.step_days,
                expanding=request.expanding,
                max_folds=request.max_folds,
                auto_execute=request.auto_execute,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/walk-forward/{group_id}")
    def walk_forward_group(
        group_id: str,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return service.walk_forward_group(group_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/experiments/{experiment_id}/promotion/evaluate")
    def promotion_evaluate(
        experiment_id: str,
        request: PromotionPolicySchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.evaluate_promotion(
                experiment_id,
                PromotionPolicy(**request.model_dump()),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/experiments/{experiment_id}/promote")
    def promote(
        experiment_id: str,
        request: PromotionPolicySchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.promote(
                experiment_id,
                PromotionPolicy(**request.model_dump()),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _register_job_progress_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    operator: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/jobs/{job_id}/begin")
    def begin(
        job_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        return _job_call(service.begin, job_id)

    @router.post("/jobs/{job_id}/progress")
    def progress(
        job_id: str,
        request: ResearchProgressSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.progress(
                job_id,
                request.progress,
                message=request.message,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/metric")
    def metric(
        job_id: str,
        request: ResearchMetricSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.metric(
                job_id,
                request.name,
                request.value,
                step=request.step,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/complete")
    def complete(
        job_id: str,
        request: ResearchCompleteSchema,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.complete(job_id, request.result)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _register_job_control_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    operator: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/jobs/{job_id}/pause")
    def pause(
        job_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.pause_execution(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/resume")
    def resume(
        job_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.resume_execution(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/retry")
    def retry(
        job_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.retry(job_id, auto_execute=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/cancel")
    def cancel(
        job_id: str,
        _: HedgePrincipal = Depends(operator),
    ) -> dict[str, Any]:
        try:
            return service.cancel_execution(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _register_artifact_route(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    viewer: Callable[..., HedgePrincipal],
) -> None:
    @router.get("/jobs/{job_id}/artifacts/{relative_path:path}")
    def artifact(
        job_id: str,
        relative_path: str,
        _: HedgePrincipal = Depends(viewer),
    ) -> FileResponse:
        try:
            path = service.artifact_path(job_id, relative_path)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            filename=path.name,
            headers={
                "Cache-Control": "no-store, max-age=0",
                "X-Content-Type-Options": "nosniff",
            },
        )


def _register_job_routes(
    router: APIRouter,
    *,
    service: HedgeResearchService,
    viewer: Callable[..., HedgePrincipal],
    operator: Callable[..., HedgePrincipal],
) -> None:
    _register_pipeline_routes(
        router,
        service=service,
        operator=operator,
    )
    _register_job_submission_routes(
        router,
        service=service,
        operator=operator,
    )
    _register_executor_control_routes(
        router,
        service=service,
        operator=operator,
    )
    _register_walk_forward_promotion_routes(
        router,
        service=service,
        viewer=viewer,
        operator=operator,
    )
    _register_job_progress_routes(
        router,
        service=service,
        operator=operator,
    )
    _register_job_control_routes(
        router,
        service=service,
        operator=operator,
    )
    _register_artifact_route(
        router,
        service=service,
        viewer=viewer,
    )


def _register_analysis_routes(
    router: APIRouter,
    *,
    viewer: Callable[..., HedgePrincipal],
) -> None:
    @router.post("/analyze/backtest")
    def analyze_backtest(
        request: BacktestAnalyzeSchema,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            summary = asdict(
                summarize_equity(
                    request.equity,
                    periods_per_year=request.periods_per_year,
                )
            )
            if request.benchmark_equity is not None:
                summary["benchmark_excess"] = benchmark_excess(
                    request.equity,
                    request.benchmark_equity,
                )
            return summary
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/analyze/optimization")
    def analyze_optimization(
        request: OptimizationRankSchema,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            objectives = tuple(
                ObjectiveWeight(
                    name=item.name,
                    weight=item.weight,
                    maximize=item.maximize,
                )
                for item in request.objectives
            )
            return {
                "ranked_indices": list(rank_trials(request.rows, objectives)),
                "pareto_indices": list(pareto_front(request.rows, objectives)),
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/analyze/ml")
    def analyze_ml(
        request: MLEvaluateSchema,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            if request.task == "regression":
                return regression_metrics(request.actual, request.predicted)
            return binary_metrics(
                request.actual,
                request.predicted,
                threshold=request.threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/analyze/rl")
    def analyze_rl(
        request: RLEvaluateSchema,
        _: HedgePrincipal = Depends(viewer),
    ) -> dict[str, Any]:
        try:
            return episode_summary(request.rewards, request.drawdowns)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


def _pipeline_call(
    function: Callable[[str], dict[str, Any]],
    pipeline_id: str,
) -> dict[str, Any]:
    try:
        return function(pipeline_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _job_call(function: Callable[[str], dict[str, Any]], job_id: str) -> dict[str, Any]:
    try:
        return function(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


_RESEARCH_SECURITY = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
}


def create_hedge_research_ui_router() -> APIRouter:
    router = APIRouter(tags=["hedge-research-ui"])
    assets = Path(__file__).with_name("hedge_research_ui")

    @router.get("/hedge-research-dashboard", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(
            (assets / "index.html").read_text(encoding="utf-8"),
            headers=_RESEARCH_SECURITY,
        )

    @router.get("/hedge-research-dashboard/assets/{name}", include_in_schema=False)
    def asset(name: str) -> FileResponse:
        if name not in {"app.js", "styles.css"}:
            raise HTTPException(status_code=404, detail="asset not found")
        media = "application/javascript" if name.endswith(".js") else "text/css"
        return FileResponse(assets / name, media_type=media, headers=_RESEARCH_SECURITY)

    return router
