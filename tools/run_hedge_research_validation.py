"""Fail-fast sequential runner for the 200 deterministic research validation checks."""

# Direct-execution bootstrap keeps this import block stable on Windows and POSIX.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_rounds = importlib.import_module("freqtrade.hedge.research.validation_matrix")
ROUND_SPECS = _rounds.ROUND_SPECS
validate_registry = _rounds.validate_registry
validate_round = _rounds.validate_round


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    validate_registry()
    results: list[dict[str, object]] = []
    started_at = datetime.now(UTC)
    for spec in ROUND_SPECS:
        started = perf_counter()
        status = "PASS"
        message = ""
        try:
            validate_round(spec.round_no)
        except Exception as exc:
            status = "FAIL"
            message = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "round": spec.round_no,
                "domain": spec.domain,
                "primary_feature": spec.primary_feature,
                "secondary_feature": spec.secondary_feature,
                "validator": spec.validator,
                "status": status,
                "duration_seconds": round(perf_counter() - started, 6),
                "message": message,
            }
        )
        if status == "FAIL":
            break
    passed = sum(item["status"] == "PASS" for item in results)
    payload = {
        "schema": "freqtrade-hedge-research-validation-v1",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "rounds_expected": 200,
        "rounds_executed": len(results),
        "rounds_passed": passed,
        "status": "PASS" if passed == 200 else "FAIL",
        "results": results,
    }
    output = args.output or root / "research-validation-result.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
