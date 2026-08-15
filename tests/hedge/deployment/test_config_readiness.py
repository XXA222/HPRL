from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from freqtrade.hedge.deployment.config import DeploymentConfigError, HedgeDeploymentConfig
from freqtrade.hedge.deployment.readiness import validate_security_readiness_report


def _payload(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")).mkdir(parents=True)
    python = project / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.write_text("", encoding="utf-8")
    config = project / "user_data" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    report = project / "user_data" / "audit" / "security-readiness" / "SECURITY-DEPLOYMENT-READINESS.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "status": "SECURITY_AND_DEPLOYMENT_HARDENING_COMPLETE_MAINNET_LOCKED",
                "security_and_deployment_hardening_complete": True,
                "mainnet_live_exchange_write": "LOCKED",
                "real_mainnet_order_requests_sent": 0,
                "installer_network_access": "NONE",
            }
        ),
        encoding="utf-8",
    )
    return {
        "project_root": str(project),
        "freqtrade_config": str(config),
        "python_executable": str(python),
        "security_readiness_report": str(report),
        "mode": "HEDGE_SIMULATED",
    }


def test_deployment_uses_current_security_readiness_contract(tmp_path: Path) -> None:
    config = HedgeDeploymentConfig.from_mapping(_payload(tmp_path))
    assert config.state_dir == config.project_root / "user_data" / "hedge" / "runtime"
    assert validate_security_readiness_report(config).status == "READY_LIVE_LOCKED"


def test_historical_readiness_key_is_not_part_of_clean_mainline(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["r37_report"] = payload.pop("security_readiness_report")
    with pytest.raises(DeploymentConfigError, match="unknown deployment config keys"):
        HedgeDeploymentConfig.from_mapping(payload)
