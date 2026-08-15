from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from freqtrade.hedge.acceptance.models import RuntimeAcceptanceReport


def write_report(
    report: RuntimeAcceptanceReport, output_directory: str | Path
) -> tuple[Path, Path]:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "ACCEPT-runtime-acceptance.json"
    md_path = output / "ACCEPT-runtime-acceptance.md"
    payload = report.to_dict()
    payload["report_sha256"] = report.sha256()
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Runtime Acceptance",
        "",
        f"- Passed: **{report.passed}**",
        f"- Live evidence: **{report.live_evidence}**",
        f"- Baseline: `{report.baseline_version}`",
        f"- Report SHA256: `{report.sha256()}`",
        "",
        "## 20 rounds",
        "",
        "| Round | Status | Title |",
        "|---|---|---|",
    ]
    for item in report.rounds:
        lines.append(f"| {item.round_id} | {item.status.value} | {item.title} |")
    lines.extend(("", "## Hard metrics", "", "```json"))
    lines.append(
        json.dumps(
            report.to_dict()["hard_metrics"], ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    lines.extend(("```", ""))
    if report.notes:
        lines.extend(("## Notes", ""))
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def print_summary(report: RuntimeAcceptanceReport) -> dict[str, Any]:
    return {
        "status": "PASS" if report.passed else "FAIL",
        "rounds": len(report.rounds),
        "passed": sum(1 for item in report.rounds if item.passed),
        "failed": sum(1 for item in report.rounds if not item.passed),
        "live_evidence": report.live_evidence,
        "hard_metrics": report.to_dict()["hard_metrics"],
        "report_sha256": report.sha256(),
    }
