"""Runtime Acceptance command handlers for the Hedge mainline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from freqtrade.hedge.acceptance.evidence import print_summary, write_report
from freqtrade.hedge.acceptance.scenario import run_deterministic_acceptance
from freqtrade.hedge.acceptance.session import RuntimeAcceptanceRoundFailure


def _project_root(value: Any) -> Path:
    return Path(str(value or ".")).expanduser().resolve()


def _output_directory(value: Any, project_root: Path) -> Path:
    if value:
        return Path(str(value)).expanduser().resolve()
    return project_root / "artifacts" / "runtime-acceptance"


def start_hedge_runtime_acceptance(args: dict[str, Any]) -> int:
    mode = str(args.get("hedge_acceptance_mode") or "deterministic").strip().lower()
    project_root = _project_root(args.get("project_root"))
    output_directory = _output_directory(
        args.get("hedge_acceptance_output_directory"),
        project_root,
    )
    database_path = Path(
        str(
            args.get("hedge_acceptance_database")
            or output_directory / "runtime-acceptance.sqlite"
        )
    ).expanduser().resolve()

    if mode == "deterministic":
        report = run_deterministic_acceptance(
            project_root=project_root,
            output_db=database_path,
        )
    elif mode == "live-readonly":
        from freqtrade.commands.hedge_readonly_commands import _assert_readonly_config
        from freqtrade.configuration import Configuration
        from freqtrade.hedge.acceptance.live import run_live_acceptance

        config = Configuration(args, None).get_config()
        _assert_readonly_config(config)
        try:
            report = asyncio.run(
                run_live_acceptance(
                    config=config,
                    project_root=project_root,
                    database_path=database_path,
                    observe_seconds=float(
                        args.get("hedge_acceptance_observe_seconds") or 60.0
                    ),
                    target_soak_stage=str(
                        args.get("hedge_acceptance_target_soak_stage") or "smoke"
                    ),
                )
            )
        except RuntimeAcceptanceRoundFailure as exc:
            evidence = exc.evidence
            output_directory.mkdir(parents=True, exist_ok=True)
            failure_path = output_directory / "runtime-acceptance-partial-failure.json"
            payload = {
                "schema": "hedge-runtime-acceptance-partial-failure-v1",
                "status": "FAIL",
                "failed_round": {
                    "round_id": evidence.round_id,
                    "title": evidence.title,
                    "checks": list(evidence.checks),
                    "metrics": dict(evidence.metrics),
                    "detail": evidence.detail,
                    "started_at": evidence.started_at.isoformat(),
                    "completed_at": evidence.completed_at.isoformat(),
                },
            }
            failure_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            payload["failure_report"] = str(failure_path)
            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return 1
    else:
        raise ValueError("acceptance mode must be deterministic or live-readonly")

    json_path, md_path = write_report(report, output_directory)
    summary = print_summary(report)
    summary["json_report"] = str(json_path)
    summary["markdown_report"] = str(md_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1
