"""Fail-closed prerequisite validation for supervised startup."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import DeploymentMode, HedgeDeploymentConfig


class DeploymentReadinessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeploymentReadiness:
    status: str
    report_sha256: str
    checks: tuple[tuple[str, str], ...]


def validate_security_readiness_report(config: HedgeDeploymentConfig) -> DeploymentReadiness:
    raw = config.security_readiness_report.read_bytes()
    payload: Any = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise DeploymentReadinessError("security readiness report must be a JSON object")
    checks: dict[str, str] = {
        "status": "PASS" if payload.get("status") == "SECURITY_AND_DEPLOYMENT_HARDENING_COMPLETE_MAINNET_LOCKED" else "FAIL",
        "complete": "PASS" if payload.get("security_and_deployment_hardening_complete") is True else "FAIL",
        "mainnet_locked": "PASS" if payload.get("mainnet_live_exchange_write") == "LOCKED" else "FAIL",
        "mainnet_requests_zero": "PASS" if int(payload.get("real_mainnet_order_requests_sent", -1)) == 0 else "FAIL",
        "installer_network_none": "PASS" if payload.get("installer_network_access") == "NONE" else "FAIL",
    }
    if any(value != "PASS" for value in checks.values()):
        raise DeploymentReadinessError(f"security readiness prerequisite failed: {checks}")
    if config.mode is DeploymentMode.PRODUCTION_LOCKED:
        checks["production_mode_locked"] = "PASS"
    elif config.mode is DeploymentMode.PAPER:
        checks["paper_mode"] = "PASS"
    elif config.mode is DeploymentMode.TESTNET_VALIDATION:
        checks["testnet_validation_only"] = "PASS"
    return DeploymentReadiness(
        status="READY_LIVE_LOCKED",
        report_sha256=hashlib.sha256(raw).hexdigest(),
        checks=tuple(sorted(checks.items())),
    )
