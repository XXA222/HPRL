from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path

from .contracts import BacktestEvaluation, OptimizationSummary
from .decimal_utils import json_value


def _atomic_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def _evaluation_payload(item: BacktestEvaluation) -> dict[str, object]:
    return {
        "candidate": {
            "candidate_id": item.candidate.candidate_id,
            "ordinal": item.candidate.ordinal,
            "parameters": item.candidate.parameters,
        },
        "dataset_fingerprint": item.dataset_fingerprint,
        "metrics": item.metrics,
        "objective_score": item.objective_score,
        "feasible": item.feasible,
        "violations": item.violations,
        "elapsed_seconds": item.elapsed_seconds,
        "evaluated_at": item.evaluated_at,
        "result_materialized": item.result is not None,
    }


def write_optimization_artifacts(
    summary: OptimizationSummary,
    *,
    output_dir: Path,
    report_title: str = "Freqtrade Hedge Parameter Optimization",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "optimization-summary.json"
    payload = {
        "schema_version": "hedge-optimization-summary-v1",
        "method": summary.method,
        "best_candidate_id": summary.best_candidate_id,
        "started_at": summary.started_at,
        "completed_at": summary.completed_at,
        "resumed": summary.resumed,
        "feasible_count": summary.feasible_count,
        "evaluation_count": len(summary.evaluations),
        "evaluations": [_evaluation_payload(item) for item in summary.evaluations],
    }
    _atomic_text(
        summary_path,
        json.dumps(json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
    )

    parameter_names = sorted(
        {name for item in summary.evaluations for name in item.candidate.parameters}
    )
    metric_names = sorted(
        {name for item in summary.evaluations for name in item.metrics}
    )
    leaderboard_path = output_dir / "leaderboard.csv"
    temporary = leaderboard_path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "rank",
            "candidate_id",
            "ordinal",
            "feasible",
            "objective_score",
            "violations",
            *[f"parameter.{name}" for name in parameter_names],
            *[f"metric.{name}" for name in metric_names],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        ranked = sorted(
            summary.evaluations,
            key=lambda item: (item.feasible, item.objective_score, -item.candidate.ordinal),
            reverse=True,
        )
        for rank, item in enumerate(ranked, 1):
            row: dict[str, object] = {
                "rank": rank,
                "candidate_id": item.candidate.candidate_id,
                "ordinal": item.candidate.ordinal,
                "feasible": item.feasible,
                "objective_score": item.objective_score,
                "violations": " | ".join(item.violations),
            }
            row.update(
                {
                    f"parameter.{key}": value
                    for key, value in item.candidate.parameters.items()
                }
            )
            row.update({f"metric.{key}": value for key, value in item.metrics.items()})
            writer.writerow(json_value(row))
    temporary.replace(leaderboard_path)

    best = next(
        (
            item
            for item in summary.evaluations
            if item.candidate.candidate_id == summary.best_candidate_id
        ),
        None,
    )
    report_path = output_dir / "REPORT.md"
    lines = [
        f"# {report_title}",
        "",
        f"- Method: `{summary.method.value}`",
        f"- Evaluations: `{len(summary.evaluations)}`",
        f"- Feasible: `{summary.feasible_count}`",
        f"- Resumed: `{summary.resumed}`",
        f"- Best candidate: `{summary.best_candidate_id or 'NONE'}`",
        "",
    ]
    if best is not None:
        lines.extend(["## Best parameters", ""])
        for name, value in sorted(best.candidate.parameters.items()):
            lines.append(f"- `{name}`: `{value}`")
        lines.extend(["", "## Best metrics", ""])
        for name in (
            "total_return_ratio",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_ratio",
            "fees",
            "funding",
            "liquidation_count",
        ):
            if name in best.metrics:
                lines.append(f"- `{name}`: `{best.metrics[name]}`")
    _atomic_text(report_path, "\n".join(lines) + "\n")

    manifest_path = output_dir / "SHA256SUMS.txt"
    files = (summary_path, leaderboard_path, report_path)
    manifest = "".join(
        f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files
    )
    _atomic_text(manifest_path, manifest, encoding="ascii")
    return {
        "summary": summary_path,
        "leaderboard": leaderboard_path,
        "report": report_path,
        "manifest": manifest_path,
    }
