"""Auditable 200-round research capability registry and deterministic validators."""

# Domain imports stay grouped by research capability for audit readability.
# ruff: noqa: I001, S101

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from .backtest import (
    benchmark_excess,
    compare_runs,
    fee_sensitivity,
    funding_sensitivity,
    maximum_drawdown,
    rolling_returns,
    scenario_matrix,
    summarize_equity,
)
from .contracts import (
    ResearchArtifact,
    ResearchBudget,
    ResearchKind,
    ResearchMetric,
    ResearchRequest,
    ResearchState,
)
from .dashboard import compare_job_metrics, dashboard_summary, metric_series
from .datasets import (
    assert_no_overlap,
    chronological_split,
    fingerprint_rows,
    validate_monotonic_timestamps,
    validate_ohlcv_rows,
    walk_forward_folds,
)
from .jobs import ResearchJobStore
from .ml import (
    MLExperimentConfig,
    binary_metrics,
    calibration_bins,
    population_stability_index,
    promotion_decision as ml_promotion_decision,
    regression_metrics,
)
from .optimization import (
    ObjectiveWeight,
    SearchDimension,
    aggregate_fold_scores,
    constraints_satisfied,
    early_stop,
    grid_candidates,
    pareto_front,
    random_candidates,
    rank_trials,
    scalar_score,
    top_k_diverse,
)
from .rl import (
    RLExperimentConfig,
    action_mask_health,
    compare_policies,
    episode_summary,
    evaluation_schedule,
    promotion_decision as rl_promotion_decision,
    reward_component_balance,
    seed_schedule,
)
from .service import HedgeResearchService
from .workspace import ResearchWorkspace, normalize_relative_path, sha256_bytes


@dataclass(frozen=True, slots=True)
class RoundSpec:
    round_no: int
    domain: str
    primary_feature: str
    secondary_feature: str
    validator: str


def _expect_error(error: type[BaseException], fn: Callable[[], object]) -> None:
    try:
        fn()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _validate_backtest_summary() -> None:
    summary = summarize_equity((100.0, 101.0, 99.0, 105.0), periods_per_year=365)
    assert summary.end == 105.0 and summary.total_return > 0 and summary.max_drawdown > 0
    assert abs(maximum_drawdown((100.0, 90.0, 110.0)) - 0.1) < 1e-12


def _validate_backtest_rolling() -> None:
    rows = rolling_returns((100.0, 101.0, 102.0, 104.0), window=2)
    assert len(rows) == 2 and rows[-1] > 0
    _expect_error(ValueError, lambda: rolling_returns((1.0, 2.0), window=2))


def _validate_backtest_benchmark() -> None:
    assert benchmark_excess((100.0, 120.0), (100.0, 110.0)) > 0
    _expect_error(ValueError, lambda: benchmark_excess((1.0, 2.0), (1.0,)))


def _validate_backtest_costs() -> None:
    fees = fee_sensitivity(0.2, (1.0, 2.0), (1.0, 5.0))
    funding = funding_sensitivity(0.2, (0.5, -0.25), (0.0, 0.001))
    assert fees["5bps"] < fees["1bps"] and funding["0.001"] < funding["0"]


def _validate_backtest_scenarios() -> None:
    rows = scenario_matrix(
        0.2,
        fee_multipliers=(1.0, 2.0),
        slippage_multipliers=(1.0, 3.0),
        cost_fraction=0.02,
    )
    assert len(rows) == 4 and rows[-1]["stressed_return"] < rows[0]["stressed_return"]


def _validate_backtest_compare() -> None:
    rows = compare_runs(({"score": 1.0}, {"score": 3.0}, {"score": 2.0}), metric="score")
    assert [row["score"] for row in rows] == [3.0, 2.0, 1.0]


def _validate_dataset_split() -> None:
    split = chronological_split(100, embargo=2)
    assert_no_overlap(split)
    assert split.train[1] < split.validation[0] < split.test[0]


def _validate_dataset_walkforward() -> None:
    folds = walk_forward_folds(100, train=40, validation=10, test=10, step=10, embargo=2)
    assert len(folds) >= 3
    for fold in folds:
        assert_no_overlap(fold)


def _validate_dataset_time() -> None:
    now = datetime.now(UTC)
    validate_monotonic_timestamps((now, now + timedelta(minutes=1)))
    _expect_error(ValueError, lambda: validate_monotonic_timestamps((now, now)))


def _validate_dataset_ohlcv() -> None:
    validate_ohlcv_rows(
        ({"timestamp": "x", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 1},)
    )
    _expect_error(
        ValueError,
        lambda: validate_ohlcv_rows(
            ({"timestamp": "x", "open": 10, "high": 9, "low": 8, "close": 11, "volume": 1},)
        ),
    )


def _validate_dataset_fingerprint() -> None:
    left = fingerprint_rows(({"a": 1},), metadata={"symbol": "BTCUSDT"})
    right = fingerprint_rows(({"a": 2},), metadata={"symbol": "BTCUSDT"})
    assert left != right and len(left) == 64


def _validate_opt_dimensions() -> None:
    SearchDimension("x", (1, 2, 3))
    _expect_error(ValueError, lambda: SearchDimension("x", (1, 1)))


def _validate_opt_grid() -> None:
    rows = grid_candidates((SearchDimension("a", (1, 2)), SearchDimension("b", (3, 4))))
    assert len(rows) == 4 and rows[0] == {"a": 1, "b": 3}


def _validate_opt_random() -> None:
    dims = (SearchDimension("a", tuple(range(10))),)
    assert random_candidates(dims, trials=4, seed=9) == random_candidates(dims, trials=4, seed=9)


def _validate_opt_scalar() -> None:
    score = scalar_score(
        {"return": 2.0, "drawdown": 0.5},
        (ObjectiveWeight("return", 2), ObjectiveWeight("drawdown", 1, False)),
    )
    assert score > 0


def _validate_opt_constraints() -> None:
    ok, violations = constraints_satisfied(
        {"dd": 0.1, "ret": 0.2},
        {"dd": ("<=", 0.2), "ret": (">=", 0.1)},
    )
    assert ok and not violations


def _validate_opt_pareto() -> None:
    objectives = (ObjectiveWeight("ret", 1), ObjectiveWeight("dd", 1, False))
    front = pareto_front(
        (
            {"ret": 1.0, "dd": 0.2},
            {"ret": 2.0, "dd": 0.1},
            {"ret": 1.5, "dd": 0.3},
        ),
        objectives,
    )
    assert front == (1,)


def _validate_opt_rank() -> None:
    order = rank_trials(({"score": 1.0}, {"score": 3.0}), (ObjectiveWeight("score", 1),))
    assert order == (1, 0)


def _validate_opt_diverse() -> None:
    chosen = top_k_diverse(({"x": 1}, {"x": 1}, {"x": 2}), (3.0, 2.0, 1.0), k=2)
    assert chosen == (0, 2)


def _validate_opt_early_stop() -> None:
    assert early_stop((1.0, 2.0, 2.0, 2.0), patience=2)
    assert not early_stop((1.0, 2.0, 3.0), patience=2)


def _validate_opt_fold_aggregate() -> None:
    assert aggregate_fold_scores((1.0, 2.0, 3.0), stability_penalty=0.2) < 2.0


def _validate_ml_config() -> None:
    MLExperimentConfig(target="y", features=("a", "b"))
    _expect_error(ValueError, lambda: MLExperimentConfig(target="a", features=("a",)))


def _validate_ml_regression() -> None:
    metrics = regression_metrics((1.0, 2.0, 3.0), (1.1, 2.0, 2.9))
    assert metrics["mae"] > 0 and metrics["rmse"] >= metrics["mae"]


def _validate_ml_binary() -> None:
    metrics = binary_metrics((0, 1, 1, 0), (0.1, 0.9, 0.8, 0.2))
    assert metrics["accuracy"] == 1.0 and metrics["brier"] < 0.1


def _validate_ml_calibration() -> None:
    rows = calibration_bins((0, 1, 1, 0), (0.1, 0.9, 0.7, 0.2), bins=5)
    assert rows and sum(row["count"] for row in rows) == 4.0


def _validate_ml_psi() -> None:
    assert population_stability_index((1.0, 1.0), (1.0, 1.0)) == 0.0
    assert population_stability_index((0.0, 0.1, 0.2), (0.8, 0.9, 1.0), bins=3) > 0


def _validate_ml_promotion() -> None:
    ok, violations = ml_promotion_decision(
        {"f1": 0.8, "dd": 0.1},
        minimums={"f1": 0.7},
        maximums={"dd": 0.2},
    )
    assert ok and not violations


def _validate_rl_config() -> None:
    RLExperimentConfig(total_timesteps=100, eval_interval=20, eval_episodes=2)
    _expect_error(ValueError, lambda: RLExperimentConfig(total_timesteps=10, eval_interval=20))


def _validate_rl_schedule() -> None:
    schedule = evaluation_schedule(
        RLExperimentConfig(total_timesteps=95, eval_interval=20, eval_episodes=2)
    )
    assert schedule[-1] == 95


def _validate_rl_seeds() -> None:
    assert seed_schedule(1, 3) == seed_schedule(1, 3) and len(set(seed_schedule(1, 3))) == 3


def _validate_rl_episode() -> None:
    summary = episode_summary((1.0, 2.0, 3.0), (0.1, 0.2, 0.15))
    assert summary["mean_reward"] == 2.0 and summary["max_drawdown"] == 0.2


def _validate_rl_mask() -> None:
    health = action_mask_health(((True, True, False), (True, False, False)))
    assert health["minimum_valid_actions"] == 1.0
    _expect_error(ValueError, lambda: action_mask_health(((False, True),)))


def _validate_rl_rewards() -> None:
    balanced = reward_component_balance(({"pnl": 1.0}, {"pnl": 3.0, "risk": -1.0}))
    assert balanced["pnl"] == 2.0 and balanced["risk"] == -0.5


def _validate_rl_promotion() -> None:
    ok, violations = rl_promotion_decision(
        {"mean_reward": 2.0, "max_drawdown": 0.1, "reward_std": 0.2},
        minimum_mean_reward=1.0,
        maximum_drawdown=0.2,
        maximum_reward_std=0.3,
    )
    assert ok and not violations


def _validate_rl_compare() -> None:
    order = compare_policies(
        (
            {"mean_reward": 1.0, "max_drawdown": 0.1, "reward_std": 0.2},
            {"mean_reward": 2.0, "max_drawdown": 0.3, "reward_std": 0.1},
        )
    )
    assert order[0] == 1


def _sample_store() -> ResearchJobStore:
    store = ResearchJobStore()
    first = store.create(ResearchRequest(ResearchKind.BACKTEST, "a"))
    second = store.create(ResearchRequest(ResearchKind.ML_EVAL, "b"))
    store.transition(first.job_id, ResearchState.RUNNING, progress=0.25)
    store.add_metric(first.job_id, ResearchMetric("score", 1.5, 1))
    store.transition(second.job_id, ResearchState.RUNNING)
    store.transition(second.job_id, ResearchState.SUCCEEDED)
    return store


def _validate_dashboard_summary() -> None:
    summary = dashboard_summary(_sample_store().list_jobs())
    assert summary["total"] == 2 and summary["running"] == 1


def _validate_dashboard_metrics() -> None:
    store = _sample_store()
    jobs = store.list_jobs()
    running = next(item for item in jobs if item.state is ResearchState.RUNNING)
    assert metric_series(running, "score")[0]["value"] == 1.5
    assert compare_job_metrics(jobs, "score")[0]["job_id"] == running.job_id


def _validate_service_lifecycle() -> None:
    with TemporaryDirectory() as directory:
        service = HedgeResearchService(Path(directory))
        row = service.submit(ResearchRequest(ResearchKind.BACKTEST, "demo", tags=("local",)))
        job_id = str(row["job_id"])
        service.begin(job_id)
        service.progress(job_id, 0.5)
        service.metric(job_id, "return", 0.1, step=1)
        final = service.complete(job_id, {"status": "PASS"})
        assert final["state"] == "SUCCEEDED" and final["progress"] == 1.0


def _validate_service_list() -> None:
    with TemporaryDirectory() as directory:
        service = HedgeResearchService(Path(directory))
        service.submit(ResearchRequest(ResearchKind.RL_EVAL, "one"))
        service.submit(ResearchRequest(ResearchKind.ML_EVAL, "two"))
        assert len(service.list_jobs(limit=1)) == 1 and service.dashboard()["total"] == 2


def _validate_workspace_paths() -> None:
    assert normalize_relative_path("a/b.json") == "a/b.json"
    for value in ("../x", "/x", "C:/x", "a/../b", "a\\..\\b", "a" + chr(0) + "b"):
        _expect_error(ValueError, lambda value=value: normalize_relative_path(value))


def _validate_workspace_artifact() -> None:
    with TemporaryDirectory() as directory:
        workspace = ResearchWorkspace(Path(directory))
        artifact = workspace.write_json("a/result.json", {"status": "PASS"})
        assert artifact.size > 0 and len(artifact.sha256) == 64
        assert workspace.read_json("a/result.json")["status"] == "PASS"
        assert sha256_bytes(b"abc") == (
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad"
        )


def _validate_job_lifecycle() -> None:
    store = ResearchJobStore()
    job = store.create(
        ResearchRequest(
            ResearchKind.OPTIMIZATION,
            "opt",
            budget=ResearchBudget(max_trials=2),
        )
    )
    store.transition(job.job_id, ResearchState.RUNNING)
    store.progress(job.job_id, 0.5)
    store.add_metric(job.job_id, ResearchMetric("score", 1.0))
    final = store.transition(job.job_id, ResearchState.SUCCEEDED)
    assert final.terminal and final.progress == 1.0


def _validate_job_guards() -> None:
    store = ResearchJobStore()
    job = store.create(ResearchRequest(ResearchKind.BACKTEST, "guard"))
    _expect_error(ValueError, lambda: store.progress(job.job_id, 0.1))
    store.transition(job.job_id, ResearchState.RUNNING)
    store.progress(job.job_id, 0.5)
    _expect_error(ValueError, lambda: store.progress(job.job_id, 0.4))
    artifact = ResearchArtifact("a", "a.json", "application/json", 2, "0" * 64)
    store.add_artifact(job.job_id, artifact)
    _expect_error(ValueError, lambda: store.add_artifact(job.job_id, artifact))


def _validate_capabilities() -> None:
    with TemporaryDirectory() as directory:
        caps = HedgeResearchService(Path(directory)).capabilities()
        assert caps["read_only_exchange"] is True and caps["live_order_write"] is False
        assert caps["rounds"] == 200


def _validate_request_contract() -> None:
    request = ResearchRequest(
        ResearchKind.BACKTEST,
        "request",
        tags=("btc", "1m"),
        budget=ResearchBudget(max_seconds=10, max_trials=2, max_workers=1),
    )
    assert request.tags == ("btc", "1m")
    _expect_error(
        ValueError,
        lambda: ResearchRequest(ResearchKind.BACKTEST, "request", tags=("x", "x")),
    )


def _validate_budget_contract() -> None:
    budget = ResearchBudget(max_seconds=1, max_trials=1, max_workers=1, max_artifact_bytes=1)
    assert budget.max_trials == 1
    _expect_error(ValueError, lambda: ResearchBudget(max_trials=0))
    _expect_error(ValueError, lambda: ResearchBudget(max_workers=0))


def _validate_artifact_limit() -> None:
    with TemporaryDirectory() as directory:
        workspace = ResearchWorkspace(Path(directory), max_artifact_bytes=4)
        workspace.write_bytes("a.bin", b"1234", media_type="application/octet-stream")
        _expect_error(
            ValueError,
            lambda: workspace.write_bytes(
                "b.bin",
                b"12345",
                media_type="application/octet-stream",
            ),
        )


def _validate_job_capacity() -> None:
    store = ResearchJobStore(capacity=2)
    first = store.create(ResearchRequest(ResearchKind.BACKTEST, "first"))
    store.transition(first.job_id, ResearchState.CANCELED)
    store.create(ResearchRequest(ResearchKind.BACKTEST, "second"))
    store.create(ResearchRequest(ResearchKind.BACKTEST, "third"))
    assert len(store.list_jobs(limit=10)) == 2
    _expect_error(KeyError, lambda: store.get(first.job_id))


def _validate_active_store_full() -> None:
    store = ResearchJobStore(capacity=1)
    store.create(ResearchRequest(ResearchKind.BACKTEST, "active"))
    _expect_error(
        RuntimeError,
        lambda: store.create(ResearchRequest(ResearchKind.BACKTEST, "blocked")),
    )


def _validate_terminal_metric() -> None:
    store = ResearchJobStore()
    job = store.create(ResearchRequest(ResearchKind.ML_EVAL, "metric-state"))
    _expect_error(
        ValueError,
        lambda: store.add_metric(job.job_id, ResearchMetric("early", 1.0)),
    )
    store.transition(job.job_id, ResearchState.RUNNING)
    store.add_metric(job.job_id, ResearchMetric("valid", 1.0))
    store.transition(job.job_id, ResearchState.SUCCEEDED)
    _expect_error(
        ValueError,
        lambda: store.add_metric(job.job_id, ResearchMetric("late", 1.0)),
    )


def _validate_command_plan() -> None:
    from .command_plan import build_command_plan

    with TemporaryDirectory() as directory:
        config = Path(directory) / "config.json"
        config.write_text("{}", encoding="utf-8")
        plan = build_command_plan(
            ResearchKind.RL_TRAIN,
            config_path=config,
            strategy="HedgeStrategy",
            python_executable="python",
        )
        assert plan.exchange_write_enabled is False
        assert "HedgeReinforcementLearner" in plan.argv


def _validate_api_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (root / "freqtrade/rpc/api_server/hedge_research.py").read_text(encoding="utf-8")
    required = (
        '/hedge/research',
        '/command-plan',
        '/analyze/backtest',
        '/analyze/optimization',
        '/analyze/ml',
        '/analyze/rl',
        '/jobs/{job_id}/artifacts/{relative_path:path}',
    )
    assert all(item in source for item in required)


def _validate_ui_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    html = (root / "freqtrade/rpc/api_server/hedge_research_ui/index.html").read_text(
        encoding="utf-8"
    )
    js = (root / "freqtrade/rpc/api_server/hedge_research_ui/app.js").read_text(
        encoding="utf-8"
    )
    assert "创建并执行研究任务" in html and "Research Dashboard" not in html
    assert "/hedge/research/jobs" in js and "setInterval(refresh, 3000)" in js


def _validate_registry_contract() -> None:
    assert len(ROUND_SPECS) == 200
    assert tuple(item.round_no for item in ROUND_SPECS) == tuple(range(1, 201))
    assert len({item.primary_feature for item in ROUND_SPECS}) == 200


def _validate_safe_surface() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "freqtrade/hedge/research").glob("*.py"))
    paths.append(root / "freqtrade/rpc/api_server/hedge_research.py")
    joined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = ("create_" + "order(", "edit_" + "order(", "cc" + "xt.")
    assert not any(token in joined for token in forbidden)



def _validate_job_artifact_budget() -> None:
    with TemporaryDirectory() as directory:
        service = HedgeResearchService(Path(directory))
        request = ResearchRequest(
            ResearchKind.BACKTEST,
            "artifact-budget",
            budget=ResearchBudget(max_artifact_bytes=220),
        )
        row = service.submit(request)
        job_id = str(row["job_id"])
        service.begin(job_id)
        _expect_error(
            ValueError,
            lambda: service.complete(job_id, {"payload": "x" * 200}),
        )
        assert service.get(job_id)["state"] == "RUNNING"



def _validate_service_recovery() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first = HedgeResearchService(root)
        row = first.submit(ResearchRequest(ResearchKind.BACKTEST, "recovery"))
        job_id = str(row["job_id"])
        first.begin(job_id)
        first.metric(job_id, "return", 0.2, step=1)
        first.complete(job_id, {"status": "PASS"})
        second = HedgeResearchService(root)
        recovered = second.get(job_id)
        assert recovered["state"] == "SUCCEEDED"
        assert recovered["metrics"][0]["value"] == 0.2


_VALIDATORS: dict[str, Callable[[], None]] = {
    name.removeprefix("_validate_"): value
    for name, value in tuple(globals().items())
    if name.startswith("_validate_") and callable(value)
}

ROUND_SPECS: tuple[RoundSpec, ...] = (
    RoundSpec(
        1,
        'backtest',
        'equity-series validation',
        'equity-series validation regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        2,
        'backtest',
        'return-series derivation',
        'return-series derivation regression guard',
        'backtest_rolling',
    ),
    RoundSpec(
        3,
        'backtest',
        'maximum-drawdown analytics',
        'maximum-drawdown analytics regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        4,
        'backtest',
        'annualized volatility',
        'annualized volatility regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        5,
        'backtest',
        'Sharpe reporting',
        'Sharpe reporting regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        6,
        'backtest',
        'Sortino reporting',
        'Sortino reporting regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        7,
        'backtest',
        'downside-deviation reporting',
        'downside-deviation reporting regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        8,
        'backtest',
        'win-rate reporting',
        'win-rate reporting regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        9,
        'backtest',
        'profit-factor reporting',
        'profit-factor reporting regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        10,
        'backtest',
        'rolling-return windows',
        'rolling-return windows regression guard',
        'backtest_rolling',
    ),
    RoundSpec(
        11,
        'backtest',
        'benchmark excess return',
        'benchmark excess return regression guard',
        'backtest_benchmark',
    ),
    RoundSpec(
        12,
        'backtest',
        'fee sensitivity matrix',
        'fee sensitivity matrix regression guard',
        'backtest_costs',
    ),
    RoundSpec(
        13,
        'backtest',
        'funding sensitivity matrix',
        'funding sensitivity matrix regression guard',
        'backtest_costs',
    ),
    RoundSpec(
        14,
        'backtest',
        'slippage stress matrix',
        'slippage stress matrix regression guard',
        'backtest_scenarios',
    ),
    RoundSpec(
        15,
        'backtest',
        'cost stress matrix',
        'cost stress matrix regression guard',
        'backtest_scenarios',
    ),
    RoundSpec(
        16,
        'backtest',
        'multi-run ranking',
        'multi-run ranking regression guard',
        'backtest_compare',
    ),
    RoundSpec(
        17,
        'backtest',
        'deterministic comparison ordering',
        'deterministic comparison ordering regression guard',
        'backtest_compare',
    ),
    RoundSpec(
        18,
        'backtest',
        'positive-equity guard',
        'positive-equity guard regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        19,
        'backtest',
        'nonfinite-series guard',
        'nonfinite-series guard regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        20,
        'backtest',
        'annualization guard',
        'annualization guard regression guard',
        'backtest_summary',
    ),
    RoundSpec(
        21,
        'backtest',
        'walk-forward split contract',
        'walk-forward split contract regression guard',
        'dataset_walkforward',
    ),
    RoundSpec(
        22,
        'backtest',
        'split embargo guard',
        'split embargo guard regression guard',
        'dataset_split',
    ),
    RoundSpec(
        23,
        'backtest',
        'time-order guard',
        'time-order guard regression guard',
        'dataset_time',
    ),
    RoundSpec(
        24,
        'backtest',
        'OHLCV integrity guard',
        'OHLCV integrity guard regression guard',
        'dataset_ohlcv',
    ),
    RoundSpec(
        25,
        'backtest',
        'dataset fingerprinting',
        'dataset fingerprinting regression guard',
        'dataset_fingerprint',
    ),
    RoundSpec(
        26,
        'backtest',
        'dataset metadata binding',
        'dataset metadata binding regression guard',
        'dataset_fingerprint',
    ),
    RoundSpec(
        27,
        'backtest',
        'fold overlap rejection',
        'fold overlap rejection regression guard',
        'dataset_split',
    ),
    RoundSpec(
        28,
        'backtest',
        'train-validation-test isolation',
        'train-validation-test isolation regression guard',
        'dataset_split',
    ),
    RoundSpec(
        29,
        'backtest',
        'multi-fold generation',
        'multi-fold generation regression guard',
        'dataset_walkforward',
    ),
    RoundSpec(
        30,
        'backtest',
        'insufficient-fold rejection',
        'insufficient-fold rejection regression guard',
        'dataset_walkforward',
    ),
    RoundSpec(
        31,
        'backtest',
        'research artifact export',
        'research artifact export regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        32,
        'backtest',
        'backtest result digest',
        'backtest result digest regression guard',
        'dataset_fingerprint',
    ),
    RoundSpec(
        33,
        'backtest',
        'job-linked backtest artifact',
        'job-linked backtest artifact regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        34,
        'backtest',
        'backtest metric timeline',
        'backtest metric timeline regression guard',
        'service_lifecycle',
    ),
    RoundSpec(
        35,
        'backtest',
        'backtest tag filtering contract',
        'backtest tag filtering contract regression guard',
        'request_contract',
    ),
    RoundSpec(
        36,
        'backtest',
        'backtest budget contract',
        'backtest budget contract regression guard',
        'budget_contract',
    ),
    RoundSpec(
        37,
        'backtest',
        'backtest cancellation state',
        'backtest cancellation state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        38,
        'backtest',
        'backtest failure state',
        'backtest failure state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        39,
        'backtest',
        'backtest success state',
        'backtest success state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        40,
        'backtest',
        'backtest dashboard projection',
        'backtest dashboard projection regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        41,
        'optimization',
        'search-dimension validation',
        'search-dimension validation regression guard',
        'opt_dimensions',
    ),
    RoundSpec(
        42,
        'optimization',
        'grid candidate generation',
        'grid candidate generation regression guard',
        'opt_grid',
    ),
    RoundSpec(
        43,
        'optimization',
        'grid limit guard',
        'grid limit guard regression guard',
        'opt_grid',
    ),
    RoundSpec(
        44,
        'optimization',
        'deterministic random sampling',
        'deterministic random sampling regression guard',
        'opt_random',
    ),
    RoundSpec(
        45,
        'optimization',
        'search seed reproducibility',
        'search seed reproducibility regression guard',
        'opt_random',
    ),
    RoundSpec(
        46,
        'optimization',
        'duplicate-dimension guard',
        'duplicate-dimension guard regression guard',
        'opt_dimensions',
    ),
    RoundSpec(
        47,
        'optimization',
        'duplicate-value guard',
        'duplicate-value guard regression guard',
        'opt_dimensions',
    ),
    RoundSpec(
        48,
        'optimization',
        'multi-objective scalarization',
        'multi-objective scalarization regression guard',
        'opt_scalar',
    ),
    RoundSpec(
        49,
        'optimization',
        'minimize-objective inversion',
        'minimize-objective inversion regression guard',
        'opt_scalar',
    ),
    RoundSpec(
        50,
        'optimization',
        'objective missing-metric guard',
        'objective missing-metric guard regression guard',
        'opt_scalar',
    ),
    RoundSpec(
        51,
        'optimization',
        'constraint minimum gate',
        'constraint minimum gate regression guard',
        'opt_constraints',
    ),
    RoundSpec(
        52,
        'optimization',
        'constraint maximum gate',
        'constraint maximum gate regression guard',
        'opt_constraints',
    ),
    RoundSpec(
        53,
        'optimization',
        'constraint missing-metric evidence',
        'constraint missing-metric evidence regression guard',
        'opt_constraints',
    ),
    RoundSpec(
        54,
        'optimization',
        'unsupported constraint guard',
        'unsupported constraint guard regression guard',
        'opt_constraints',
    ),
    RoundSpec(
        55,
        'optimization',
        'Pareto frontier extraction',
        'Pareto frontier extraction regression guard',
        'opt_pareto',
    ),
    RoundSpec(
        56,
        'optimization',
        'Pareto domination strictness',
        'Pareto domination strictness regression guard',
        'opt_pareto',
    ),
    RoundSpec(
        57,
        'optimization',
        'trial ranking',
        'trial ranking regression guard',
        'opt_rank',
    ),
    RoundSpec(
        58,
        'optimization',
        'stable ranking order',
        'stable ranking order regression guard',
        'opt_rank',
    ),
    RoundSpec(
        59,
        'optimization',
        'top-k diversity',
        'top-k diversity regression guard',
        'opt_diverse',
    ),
    RoundSpec(
        60,
        'optimization',
        'duplicate candidate suppression',
        'duplicate candidate suppression regression guard',
        'opt_diverse',
    ),
    RoundSpec(
        61,
        'optimization',
        'early-stop patience',
        'early-stop patience regression guard',
        'opt_early_stop',
    ),
    RoundSpec(
        62,
        'optimization',
        'minimum-improvement guard',
        'minimum-improvement guard regression guard',
        'opt_early_stop',
    ),
    RoundSpec(
        63,
        'optimization',
        'fold-score aggregation',
        'fold-score aggregation regression guard',
        'opt_fold_aggregate',
    ),
    RoundSpec(
        64,
        'optimization',
        'stability penalty',
        'stability penalty regression guard',
        'opt_fold_aggregate',
    ),
    RoundSpec(
        65,
        'optimization',
        'optimization job budget',
        'optimization job budget regression guard',
        'budget_contract',
    ),
    RoundSpec(
        66,
        'optimization',
        'trial-count budget',
        'trial-count budget regression guard',
        'budget_contract',
    ),
    RoundSpec(
        67,
        'optimization',
        'worker-count budget',
        'worker-count budget regression guard',
        'budget_contract',
    ),
    RoundSpec(
        68,
        'optimization',
        'optimization artifact export',
        'optimization artifact export regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        69,
        'optimization',
        'optimization metric stream',
        'optimization metric stream regression guard',
        'service_lifecycle',
    ),
    RoundSpec(
        70,
        'optimization',
        'optimization cancel state',
        'optimization cancel state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        71,
        'optimization',
        'optimization failure evidence',
        'optimization failure evidence regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        72,
        'optimization',
        'optimization success state',
        'optimization success state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        73,
        'optimization',
        'optimization dashboard comparison',
        'optimization dashboard comparison regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        74,
        'optimization',
        'optimization request persistence',
        'optimization request persistence regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        75,
        'optimization',
        'optimization capability projection',
        'optimization capability projection regression guard',
        'capabilities',
    ),
    RoundSpec(
        76,
        'ml',
        'ML experiment contract',
        'ML experiment contract regression guard',
        'ml_config',
    ),
    RoundSpec(
        77,
        'ml',
        'feature uniqueness guard',
        'feature uniqueness guard regression guard',
        'ml_config',
    ),
    RoundSpec(
        78,
        'ml',
        'target leakage guard',
        'target leakage guard regression guard',
        'ml_config',
    ),
    RoundSpec(
        79,
        'ml',
        'split-ratio guard',
        'split-ratio guard regression guard',
        'ml_config',
    ),
    RoundSpec(
        80,
        'ml',
        'regression MAE',
        'regression MAE regression guard',
        'ml_regression',
    ),
    RoundSpec(
        81,
        'ml',
        'regression RMSE',
        'regression RMSE regression guard',
        'ml_regression',
    ),
    RoundSpec(
        82,
        'ml',
        'regression R2',
        'regression R2 regression guard',
        'ml_regression',
    ),
    RoundSpec(
        83,
        'ml',
        'directional accuracy',
        'directional accuracy regression guard',
        'ml_regression',
    ),
    RoundSpec(
        84,
        'ml',
        'binary accuracy',
        'binary accuracy regression guard',
        'ml_binary',
    ),
    RoundSpec(
        85,
        'ml',
        'binary precision',
        'binary precision regression guard',
        'ml_binary',
    ),
    RoundSpec(
        86,
        'ml',
        'binary recall',
        'binary recall regression guard',
        'ml_binary',
    ),
    RoundSpec(
        87,
        'ml',
        'binary F1',
        'binary F1 regression guard',
        'ml_binary',
    ),
    RoundSpec(
        88,
        'ml',
        'Brier score',
        'Brier score regression guard',
        'ml_binary',
    ),
    RoundSpec(
        89,
        'ml',
        'classification threshold guard',
        'classification threshold guard regression guard',
        'ml_binary',
    ),
    RoundSpec(
        90,
        'ml',
        'probability bounds guard',
        'probability bounds guard regression guard',
        'ml_binary',
    ),
    RoundSpec(
        91,
        'ml',
        'calibration bins',
        'calibration bins regression guard',
        'ml_calibration',
    ),
    RoundSpec(
        92,
        'ml',
        'calibration empty-bin tolerance',
        'calibration empty-bin tolerance regression guard',
        'ml_calibration',
    ),
    RoundSpec(
        93,
        'ml',
        'population stability index',
        'population stability index regression guard',
        'ml_psi',
    ),
    RoundSpec(
        94,
        'ml',
        'constant-distribution PSI',
        'constant-distribution PSI regression guard',
        'ml_psi',
    ),
    RoundSpec(
        95,
        'ml',
        'ML promotion minimum gate',
        'ML promotion minimum gate regression guard',
        'ml_promotion',
    ),
    RoundSpec(
        96,
        'ml',
        'ML promotion maximum gate',
        'ML promotion maximum gate regression guard',
        'ml_promotion',
    ),
    RoundSpec(
        97,
        'ml',
        'ML promotion missing metric',
        'ML promotion missing metric regression guard',
        'ml_promotion',
    ),
    RoundSpec(
        98,
        'ml',
        'model-family metadata',
        'model-family metadata regression guard',
        'ml_config',
    ),
    RoundSpec(
        99,
        'ml',
        'training seed contract',
        'training seed contract regression guard',
        'ml_config',
    ),
    RoundSpec(
        100,
        'ml',
        'feature-schema persistence',
        'feature-schema persistence regression guard',
        'ml_config',
    ),
    RoundSpec(
        101,
        'ml',
        'ML request artifact',
        'ML request artifact regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        102,
        'ml',
        'ML metric stream',
        'ML metric stream regression guard',
        'service_lifecycle',
    ),
    RoundSpec(
        103,
        'ml',
        'ML success artifact',
        'ML success artifact regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        104,
        'ml',
        'ML failure state',
        'ML failure state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        105,
        'ml',
        'ML cancellation state',
        'ML cancellation state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        106,
        'ml',
        'ML dashboard projection',
        'ML dashboard projection regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        107,
        'ml',
        'ML job comparison',
        'ML job comparison regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        108,
        'ml',
        'finite-prediction guard',
        'finite-prediction guard regression guard',
        'ml_regression',
    ),
    RoundSpec(
        109,
        'ml',
        'label-shape guard',
        'label-shape guard regression guard',
        'ml_binary',
    ),
    RoundSpec(
        110,
        'ml',
        'ML capability projection',
        'ML capability projection regression guard',
        'capabilities',
    ),
    RoundSpec(
        111,
        'rl',
        'RL experiment contract',
        'RL experiment contract regression guard',
        'rl_config',
    ),
    RoundSpec(
        112,
        'rl',
        'algorithm-name guard',
        'algorithm-name guard regression guard',
        'rl_config',
    ),
    RoundSpec(
        113,
        'rl',
        'training-step guard',
        'training-step guard regression guard',
        'rl_config',
    ),
    RoundSpec(
        114,
        'rl',
        'evaluation interval guard',
        'evaluation interval guard regression guard',
        'rl_config',
    ),
    RoundSpec(
        115,
        'rl',
        'evaluation episode guard',
        'evaluation episode guard regression guard',
        'rl_config',
    ),
    RoundSpec(
        116,
        'rl',
        'evaluation schedule',
        'evaluation schedule regression guard',
        'rl_schedule',
    ),
    RoundSpec(
        117,
        'rl',
        'terminal evaluation inclusion',
        'terminal evaluation inclusion regression guard',
        'rl_schedule',
    ),
    RoundSpec(
        118,
        'rl',
        'evaluation seed schedule',
        'evaluation seed schedule regression guard',
        'rl_seeds',
    ),
    RoundSpec(
        119,
        'rl',
        'seed reproducibility',
        'seed reproducibility regression guard',
        'rl_seeds',
    ),
    RoundSpec(
        120,
        'rl',
        'episode mean reward',
        'episode mean reward regression guard',
        'rl_episode',
    ),
    RoundSpec(
        121,
        'rl',
        'episode reward volatility',
        'episode reward volatility regression guard',
        'rl_episode',
    ),
    RoundSpec(
        122,
        'rl',
        'episode worst reward',
        'episode worst reward regression guard',
        'rl_episode',
    ),
    RoundSpec(
        123,
        'rl',
        'episode best reward',
        'episode best reward regression guard',
        'rl_episode',
    ),
    RoundSpec(
        124,
        'rl',
        'episode mean drawdown',
        'episode mean drawdown regression guard',
        'rl_episode',
    ),
    RoundSpec(
        125,
        'rl',
        'episode max drawdown',
        'episode max drawdown regression guard',
        'rl_episode',
    ),
    RoundSpec(
        126,
        'rl',
        'nonfinite reward guard',
        'nonfinite reward guard regression guard',
        'rl_episode',
    ),
    RoundSpec(
        127,
        'rl',
        'drawdown bounds guard',
        'drawdown bounds guard regression guard',
        'rl_episode',
    ),
    RoundSpec(
        128,
        'rl',
        'action-mask shape guard',
        'action-mask shape guard regression guard',
        'rl_mask',
    ),
    RoundSpec(
        129,
        'rl',
        'HOLD action invariant',
        'HOLD action invariant regression guard',
        'rl_mask',
    ),
    RoundSpec(
        130,
        'rl',
        'nonempty action invariant',
        'nonempty action invariant regression guard',
        'rl_mask',
    ),
    RoundSpec(
        131,
        'rl',
        'valid-action ratio',
        'valid-action ratio regression guard',
        'rl_mask',
    ),
    RoundSpec(
        132,
        'rl',
        'reward-component aggregation',
        'reward-component aggregation regression guard',
        'rl_rewards',
    ),
    RoundSpec(
        133,
        'rl',
        'reward missing-component zero fill',
        'reward missing-component zero fill regression guard',
        'rl_rewards',
    ),
    RoundSpec(
        134,
        'rl',
        'RL promotion mean-reward gate',
        'RL promotion mean-reward gate regression guard',
        'rl_promotion',
    ),
    RoundSpec(
        135,
        'rl',
        'RL promotion drawdown gate',
        'RL promotion drawdown gate regression guard',
        'rl_promotion',
    ),
    RoundSpec(
        136,
        'rl',
        'RL promotion stability gate',
        'RL promotion stability gate regression guard',
        'rl_promotion',
    ),
    RoundSpec(
        137,
        'rl',
        'policy comparison',
        'policy comparison regression guard',
        'rl_compare',
    ),
    RoundSpec(
        138,
        'rl',
        'policy tie risk ordering',
        'policy tie risk ordering regression guard',
        'rl_compare',
    ),
    RoundSpec(
        139,
        'rl',
        'RL request artifact',
        'RL request artifact regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        140,
        'rl',
        'RL metric stream',
        'RL metric stream regression guard',
        'service_lifecycle',
    ),
    RoundSpec(
        141,
        'rl',
        'RL success artifact',
        'RL success artifact regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        142,
        'rl',
        'RL failure state',
        'RL failure state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        143,
        'rl',
        'RL cancellation state',
        'RL cancellation state regression guard',
        'job_lifecycle',
    ),
    RoundSpec(
        144,
        'rl',
        'RL dashboard projection',
        'RL dashboard projection regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        145,
        'rl',
        'RL capability projection',
        'RL capability projection regression guard',
        'capabilities',
    ),
    RoundSpec(
        146,
        'dashboard',
        'research dashboard summary',
        'research dashboard summary regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        147,
        'dashboard',
        'state counters',
        'state counters regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        148,
        'dashboard',
        'kind counters',
        'kind counters regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        149,
        'dashboard',
        'running progress summary',
        'running progress summary regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        150,
        'dashboard',
        'recent-job ordering',
        'recent-job ordering regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        151,
        'dashboard',
        'metric-series projection',
        'metric-series projection regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        152,
        'dashboard',
        'job metric comparison',
        'job metric comparison regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        153,
        'dashboard',
        'comparison ordering',
        'comparison ordering regression guard',
        'dashboard_metrics',
    ),
    RoundSpec(
        154,
        'dashboard',
        'empty-dashboard handling',
        'empty-dashboard handling regression guard',
        'dashboard_summary',
    ),
    RoundSpec(
        155,
        'dashboard',
        'job list limit',
        'job list limit regression guard',
        'service_list',
    ),
    RoundSpec(
        156,
        'dashboard',
        'job detail projection',
        'job detail projection regression guard',
        'service_list',
    ),
    RoundSpec(
        157,
        'dashboard',
        'request tags projection',
        'request tags projection regression guard',
        'service_list',
    ),
    RoundSpec(
        158,
        'dashboard',
        'artifact projection',
        'artifact projection regression guard',
        'service_list',
    ),
    RoundSpec(
        159,
        'dashboard',
        'revision projection',
        'revision projection regression guard',
        'service_list',
    ),
    RoundSpec(
        160,
        'dashboard',
        'created-time projection',
        'created-time projection regression guard',
        'service_list',
    ),
    RoundSpec(
        161,
        'dashboard',
        'started-time projection',
        'started-time projection regression guard',
        'service_list',
    ),
    RoundSpec(
        162,
        'dashboard',
        'finished-time projection',
        'finished-time projection regression guard',
        'service_list',
    ),
    RoundSpec(
        163,
        'dashboard',
        'local-only research API contract',
        'local-only research API contract regression guard',
        'api_contract',
    ),
    RoundSpec(
        164,
        'dashboard',
        'viewer capability endpoint',
        'viewer capability endpoint regression guard',
        'api_contract',
    ),
    RoundSpec(
        165,
        'dashboard',
        'viewer job-list endpoint',
        'viewer job-list endpoint regression guard',
        'api_contract',
    ),
    RoundSpec(
        166,
        'dashboard',
        'viewer job-detail endpoint',
        'viewer job-detail endpoint regression guard',
        'api_contract',
    ),
    RoundSpec(
        167,
        'dashboard',
        'operator job-create endpoint',
        'operator job-create endpoint regression guard',
        'api_contract',
    ),
    RoundSpec(
        168,
        'dashboard',
        'operator job-cancel endpoint',
        'operator job-cancel endpoint regression guard',
        'api_contract',
    ),
    RoundSpec(
        169,
        'dashboard',
        'research UI shell',
        'research UI shell regression guard',
        'ui_contract',
    ),
    RoundSpec(
        170,
        'dashboard',
        'research UI jobs table',
        'research UI jobs table regression guard',
        'ui_contract',
    ),
    RoundSpec(
        171,
        'dashboard',
        'research UI filters',
        'research UI filters regression guard',
        'ui_contract',
    ),
    RoundSpec(
        172,
        'dashboard',
        'research UI progress bars',
        'research UI progress bars regression guard',
        'ui_contract',
    ),
    RoundSpec(
        173,
        'dashboard',
        'research UI artifact links',
        'research UI artifact links regression guard',
        'ui_contract',
    ),
    RoundSpec(
        174,
        'dashboard',
        'research UI polling',
        'research UI polling regression guard',
        'ui_contract',
    ),
    RoundSpec(
        175,
        'dashboard',
        'research dashboard CSP boundary',
        'research dashboard CSP boundary regression guard',
        'ui_contract',
    ),
    RoundSpec(
        176,
        'integration',
        'research service creation',
        'research service creation regression guard',
        'service_lifecycle',
    ),
    RoundSpec(
        177,
        'integration',
        'request persistence',
        'restart recovery and persistence regression guard',
        'service_recovery',
    ),
    RoundSpec(
        178,
        'integration',
        'atomic JSON artifact write',
        'atomic JSON artifact write regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        179,
        'integration',
        'artifact SHA256',
        'artifact SHA256 regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        180,
        'integration',
        'artifact size accounting',
        'cumulative job artifact budget regression guard',
        'job_artifact_budget',
    ),
    RoundSpec(
        181,
        'integration',
        'artifact path normalization',
        'artifact path normalization regression guard',
        'workspace_paths',
    ),
    RoundSpec(
        182,
        'integration',
        'path traversal rejection',
        'path traversal rejection regression guard',
        'workspace_paths',
    ),
    RoundSpec(
        183,
        'integration',
        'Windows-drive path rejection',
        'Windows-drive path rejection regression guard',
        'workspace_paths',
    ),
    RoundSpec(
        184,
        'integration',
        'NUL path rejection',
        'NUL path rejection regression guard',
        'workspace_paths',
    ),
    RoundSpec(
        185,
        'integration',
        'workspace escape rejection',
        'workspace escape rejection regression guard',
        'workspace_paths',
    ),
    RoundSpec(
        186,
        'integration',
        'job state machine',
        'job state machine regression guard',
        'job_guards',
    ),
    RoundSpec(
        187,
        'integration',
        'invalid transition rejection',
        'invalid transition rejection regression guard',
        'job_guards',
    ),
    RoundSpec(
        188,
        'integration',
        'monotonic progress',
        'monotonic progress regression guard',
        'job_guards',
    ),
    RoundSpec(
        189,
        'integration',
        'backward-progress rejection',
        'backward-progress rejection regression guard',
        'job_guards',
    ),
    RoundSpec(
        190,
        'integration',
        'job store capacity eviction',
        'job store capacity eviction regression guard',
        'job_capacity',
    ),
    RoundSpec(
        191,
        'integration',
        'active-store full rejection',
        'active-store full rejection regression guard',
        'active_store_full',
    ),
    RoundSpec(
        192,
        'integration',
        'duplicate artifact rejection',
        'duplicate artifact rejection regression guard',
        'workspace_artifact',
    ),
    RoundSpec(
        193,
        'integration',
        'metric append contract',
        'metric append contract regression guard',
        'terminal_metric',
    ),
    RoundSpec(
        194,
        'integration',
        'failed-job metric rejection',
        'failed-job metric rejection regression guard',
        'terminal_metric',
    ),
    RoundSpec(
        195,
        'integration',
        'research capability registry',
        'research capability registry regression guard',
        'capabilities',
    ),
    RoundSpec(
        196,
        'integration',
        'exact 200-round registry',
        'exact 200-round registry regression guard',
        'registry_contract',
    ),
    RoundSpec(
        197,
        'integration',
        'round ordering invariant',
        'round ordering invariant regression guard',
        'registry_contract',
    ),
    RoundSpec(
        198,
        'integration',
        'domain round counts',
        'domain round counts regression guard',
        'registry_contract',
    ),
    RoundSpec(
        199,
        'integration',
        'live-order-write disabled',
        'live-order-write disabled regression guard',
        'safe_surface',
    ),
    RoundSpec(
        200,
        'integration',
        'exchange-readonly declaration',
        'exchange-readonly declaration regression guard',
        'safe_surface',
    ),
)

ROUND_BY_NUMBER = {item.round_no: item for item in ROUND_SPECS}


def validate_round(round_no: int) -> RoundSpec:
    try:
        spec = ROUND_BY_NUMBER[round_no]
    except KeyError as exc:
        raise ValueError(f"unknown research development round: {round_no}") from exc
    _VALIDATORS[spec.validator]()
    return spec


def validate_registry() -> None:
    assert len(ROUND_SPECS) == 200
    assert tuple(item.round_no for item in ROUND_SPECS) == tuple(range(1, 201))
    assert len({item.primary_feature for item in ROUND_SPECS}) == 200
    expected = {
        "backtest": 40,
        "optimization": 35,
        "ml": 35,
        "rl": 35,
        "dashboard": 30,
        "integration": 25,
    }
    actual = {name: sum(item.domain == name for item in ROUND_SPECS) for name in expected}
    assert actual == expected
