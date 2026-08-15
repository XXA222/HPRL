"""Command-line entrypoint for the external Hedge deployment supervisor."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .config import HedgeDeploymentConfig
from .readiness import validate_security_readiness_report
from .state import RuntimeStateStore
from .supervisor import HedgeProcessSupervisor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freqtrade-hedge-supervisor")
    parser.add_argument("--deployment-config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("run")
    subparsers.add_parser("status")
    subparsers.add_parser("stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = HedgeDeploymentConfig.from_file(args.deployment_config)
    if args.command == "preflight":
        readiness = validate_security_readiness_report(config)
        print(json.dumps(asdict(readiness), sort_keys=True, default=str))
        return 0
    supervisor = HedgeProcessSupervisor(config)
    if args.command == "run":
        result = supervisor.run()
        print(json.dumps(asdict(result), sort_keys=True, default=str))
        return 0 if result.phase.value in {"STOPPED"} else 2
    if args.command == "stop":
        supervisor.request_stop()
        print(json.dumps({"status": "STOP_REQUESTED"}, sort_keys=True))
        return 0
    if args.command == "status":
        state = RuntimeStateStore(config.state_dir / "runtime-state.json").read()
        if state is None:
            print(json.dumps({"status": "NOT_STARTED"}, sort_keys=True))
            return 3
        payload = asdict(state)
        payload["phase"] = state.phase.value
        heartbeat = datetime.fromisoformat(state.heartbeat_at_utc)
        age = (datetime.now(UTC) - heartbeat).total_seconds()
        payload["heartbeat_age_seconds"] = age
        payload["heartbeat_fresh"] = age <= config.heartbeat_stale_seconds
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["heartbeat_fresh"] else 4
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
