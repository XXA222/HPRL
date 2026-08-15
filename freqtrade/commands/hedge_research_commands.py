"""CLI entry points for the local Hedge research control plane."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from freqtrade.hedge.research.validation_matrix import (
    ROUND_SPECS,
    validate_registry,
    validate_round,
)


def _write_or_print(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def start_hedge_research_capabilities(args: dict[str, Any]) -> int:
    validate_registry()
    counts = Counter(item.domain for item in ROUND_SPECS)
    payload = {
        "schema": "freqtrade-hedge-research-capabilities-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "rounds": len(ROUND_SPECS),
        "rounds_by_domain": dict(sorted(counts.items())),
        "read_only_exchange": True,
        "live_order_write": False,
        "interfaces": {
            "backtest": "hedge-backtesting",
            "optimization": "hedge-research-optimize / hedge-hyperopt",
            "ml": "FreqAI HedgePyTorchMultiTaskRegressor + research metrics/API",
            "rl": "HedgeReinforcementLearner/MaskablePPO + research metrics/API",
            "dashboard": "/hedge-research-dashboard",
            "api": "/api/v1/hedge/research/*",
            "executor": "bounded local queue with cancel/timeout/resource telemetry",
            "logs": "live job output via /jobs/{job_id}/log",
        },
    }
    _write_or_print(payload, args.get("hedge_research_output"))
    return 0


def start_hedge_research_validate(args: dict[str, Any]) -> int:
    validate_registry()
    rows = []
    status = "PASS"
    for spec in ROUND_SPECS:
        try:
            validate_round(spec.round_no)
            row_status = "PASS"
            message = ""
        except Exception as exc:
            row_status = "FAIL"
            message = f"{type(exc).__name__}: {exc}"
            status = "FAIL"
        rows.append(
            {
                "round": spec.round_no,
                "domain": spec.domain,
                "primary_feature": spec.primary_feature,
                "secondary_feature": spec.secondary_feature,
                "status": row_status,
                "message": message,
            }
        )
        if status == "FAIL":
            break
    payload = {
        "schema": "freqtrade-hedge-research-200-round-validation-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "rounds_expected": 200,
        "rounds_executed": len(rows),
        "rounds_passed": sum(item["status"] == "PASS" for item in rows),
        "status": status,
        "results": rows,
    }
    _write_or_print(payload, args.get("hedge_research_output"))
    return 0 if status == "PASS" else 1
