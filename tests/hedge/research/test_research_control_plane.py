from __future__ import annotations

from pathlib import Path

import pytest

from freqtrade.hedge.research.backtest import summarize_equity
from freqtrade.hedge.research.config import build_research_service
from freqtrade.hedge.research.contracts import ResearchKind, ResearchRequest
from freqtrade.hedge.research.datasets import chronological_split, walk_forward_folds
from freqtrade.hedge.research.service import HedgeResearchService


def test_research_service_persists_request_and_result(tmp_path: Path) -> None:
    service = HedgeResearchService(tmp_path)
    row = service.submit(ResearchRequest(ResearchKind.BACKTEST, "baseline"))
    job_id = row["job_id"]
    service.begin(job_id)
    service.metric(job_id, "return", 0.12)
    final = service.complete(job_id, {"status": "PASS"})
    assert final["state"] == "SUCCEEDED"
    assert (tmp_path / f"jobs/{job_id}/request.json").is_file()
    assert (tmp_path / f"jobs/{job_id}/result.json").is_file()


def test_build_research_service_uses_user_data_relative_workspace(tmp_path: Path) -> None:
    service = build_research_service(
        {
            "user_data_dir": tmp_path,
            "hedge": {"research": {"enabled": True, "workspace": "research"}},
        }
    )
    assert service.workspace.root == (tmp_path / "research").resolve()


def test_build_research_service_default_workspace_is_not_duplicated(tmp_path: Path) -> None:
    service = build_research_service(
        {"user_data_dir": tmp_path, "hedge": {"research": {"enabled": True}}}
    )
    assert service.workspace.root == (tmp_path / "hedge_research").resolve()


def test_backtest_summary_rejects_nonpositive_equity() -> None:
    with pytest.raises(ValueError, match="equity must remain positive"):
        summarize_equity((100.0, 0.0, 90.0))


def test_walk_forward_and_holdout_splits_are_ordered() -> None:
    split = chronological_split(100, embargo=2)
    assert split.train[1] < split.validation[0] < split.validation[1] < split.test[0]
    folds = walk_forward_folds(120, train=40, validation=10, test=10, step=10, embargo=2)
    assert len(folds) >= 4


def test_command_plan_uses_existing_safe_cli_entrypoints(tmp_path: Path) -> None:
    from freqtrade.hedge.research.command_plan import build_command_plan

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    plan = build_command_plan(
        ResearchKind.RL_TRAIN,
        config_path=config,
        strategy="HedgeStrategy",
        timerange="20250101-20250201",
        python_executable="python.exe",
    )
    assert plan.argv[:4] == ("python.exe", "-m", "freqtrade", "backtesting")
    assert "HedgeReinforcementLearner" in plan.argv
    assert plan.exchange_write_enabled is False


def test_command_plan_rejects_strategy_shell_metacharacters(tmp_path: Path) -> None:
    from freqtrade.hedge.research.command_plan import build_command_plan

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="identifier"):
        build_command_plan(
            ResearchKind.BACKTEST,
            config_path=config,
            strategy="Bad;Strategy",
        )


def test_research_service_recovers_completed_job(tmp_path: Path) -> None:
    first = HedgeResearchService(tmp_path)
    row = first.submit(ResearchRequest(ResearchKind.ML_EVAL, "recover-me"))
    job_id = row["job_id"]
    first.begin(job_id)
    first.metric(job_id, "f1", 0.8, step=1)
    first.complete(job_id, {"status": "PASS"})

    recovered = HedgeResearchService(tmp_path)
    row = recovered.get(job_id)
    assert row["state"] == "SUCCEEDED"
    assert row["metrics"] == [{"name": "f1", "value": 0.8, "step": 1}]
    assert len(row["artifacts"]) == 2


def test_research_service_fails_interrupted_running_job_closed(tmp_path: Path) -> None:
    first = HedgeResearchService(tmp_path)
    row = first.submit(ResearchRequest(ResearchKind.RL_TRAIN, "interrupted"))
    job_id = row["job_id"]
    first.begin(job_id)
    first.progress(job_id, 0.4)

    recovered = HedgeResearchService(tmp_path)
    row = recovered.get(job_id)
    assert row["state"] == "FAILED"
    assert row["progress"] == 0.4
    assert "restart" in row["message"]


def test_research_metrics_require_running_state(tmp_path: Path) -> None:
    service = HedgeResearchService(tmp_path)
    row = service.submit(ResearchRequest(ResearchKind.BACKTEST, "metric-state"))
    with pytest.raises(ValueError, match="RUNNING"):
        service.metric(row["job_id"], "return", 0.1)


def test_job_artifact_budget_is_cumulative(tmp_path: Path) -> None:
    from freqtrade.hedge.research.contracts import ResearchBudget

    service = HedgeResearchService(tmp_path)
    row = service.submit(
        ResearchRequest(
            ResearchKind.BACKTEST,
            "artifact-budget",
            budget=ResearchBudget(max_artifact_bytes=220),
        )
    )
    job_id = row["job_id"]
    service.begin(job_id)
    with pytest.raises(ValueError, match="artifact"):
        service.complete(job_id, {"payload": "x" * 200})
    assert service.get(job_id)["state"] == "RUNNING"


def test_random_candidates_handles_large_cartesian_space_without_materializing() -> None:
    from freqtrade.hedge.research.optimization import SearchDimension, random_candidates

    rows = random_candidates(
        (
            SearchDimension("a", tuple(range(1000))),
            SearchDimension("b", tuple(range(1000))),
            SearchDimension("c", tuple(range(1000))),
        ),
        trials=4,
        seed=17,
    )
    assert len(rows) == 4
    assert len({tuple(row.items()) for row in rows}) == 4


def test_binary_metrics_rejects_fractional_labels() -> None:
    from freqtrade.hedge.research.ml import binary_metrics

    with pytest.raises(ValueError, match="binary"):
        binary_metrics((0.0, 0.9), (0.1, 0.9))


def test_maximum_drawdown_rejects_nonpositive_equity() -> None:
    from freqtrade.hedge.research.backtest import maximum_drawdown

    with pytest.raises(ValueError, match="positive"):
        maximum_drawdown((100.0, 0.0, 90.0))


def test_action_mask_requires_real_booleans() -> None:
    from freqtrade.hedge.research.rl import action_mask_health

    with pytest.raises(ValueError, match="shapes"):
        action_mask_health(((1, 0, 1),))
