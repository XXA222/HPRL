"""Fail-closed model promotion gates for dry-run candidates."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    min_sharpe: float | None = None
    max_drawdown: float | None = None
    min_reward: float | None = None
    max_loss: float | None = None
    min_profit: float | None = None
    require_model_files: bool = True


def evaluate_promotion(experiment: dict[str, Any], policy: PromotionPolicy) -> dict[str, Any]:
    metrics = experiment.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    checks: list[dict[str, Any]] = []

    def check(name: str, expected: float | None, *, minimum: bool) -> None:
        if expected is None:
            return
        raw = metrics.get(name)
        if raw is None:
            checks.append({"name": name, "passed": False, "reason": "metric missing"})
            return
        value = float(raw)
        passed = value >= expected if minimum else value <= expected
        operator = ">=" if minimum else "<="
        checks.append(
            {
                "name": name,
                "value": value,
                "required": expected,
                "operator": operator,
                "passed": passed,
            }
        )

    check("sharpe", policy.min_sharpe, minimum=True)
    check("drawdown", policy.max_drawdown, minimum=False)
    check("reward", policy.min_reward, minimum=True)
    check("loss", policy.max_loss, minimum=False)
    check("profit", policy.min_profit, minimum=True)

    files = experiment.get("model_files", [])
    if policy.require_model_files:
        checks.append(
            {
                "name": "model_files",
                "value": len(files) if isinstance(files, list) else 0,
                "required": 1,
                "operator": ">=",
                "passed": isinstance(files, list) and len(files) > 0,
            }
        )
    passed = bool(checks) and all(bool(item["passed"]) for item in checks)
    return {
        "passed": passed,
        "checks": checks,
        "identifier": str(experiment.get("identifier", "")),
        "experiment_id": str(experiment.get("experiment_id", "")),
    }


def build_promotion_record(
    experiment: dict[str, Any],
    policy: PromotionPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = evaluate_promotion(experiment, policy)
    if not gate["passed"]:
        raise ValueError("experiment does not satisfy the dry-run promotion policy")
    identifier = str(experiment.get("identifier", "")).strip()
    if not identifier:
        raise ValueError("experiment has no FreqAI identifier")
    promotion_id = f"promotion-{uuid.uuid4().hex[:20]}"
    record = {
        "promotion_id": promotion_id,
        "target": "DRY_RUN_CANDIDATE",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": str(experiment.get("experiment_id", "")),
        "job_id": str(experiment.get("job_id", "")),
        "identifier": identifier,
        "gate": gate,
        "live_order_write": False,
    }
    dry_run_override = {
        "dry_run": True,
        "freqai": {
            "enabled": True,
            "identifier": identifier,
        },
    }
    return record, dry_run_override
