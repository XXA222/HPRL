"""Fail-closed aggregate acceptance for HPRL runtime closure R2."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


HPRL_RUNTIME_CLOSURE_API_VERSION = "2.0"
HPRL_RUNTIME_CLOSURE_RELEASE = "freqtrade-hedge-hprl-v3-runtime-closure-r2"


class EvidenceState(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class RuntimeClosureEvidence:
    name: str
    state: EvidenceState
    digest: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("evidence name is required")
        if self.digest and (len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest.lower())):
            raise ValueError("evidence digest must be SHA-256 when supplied")


@dataclass(frozen=True, slots=True)
class RuntimeClosurePolicy:
    require_container_pytest: bool = True
    require_postgres_core: bool = True
    require_postgres_failover: bool = True
    require_postgres_restore: bool = True
    require_binance_real_market_dryrun: bool = True
    require_fault_campaign: bool = True
    require_shadow_24h: bool = True
    require_shadow_72h: bool = True
    require_two_year_backtest: bool = True
    require_position_behavior: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeClosureAcceptance:
    state: EvidenceState
    evidence: tuple[RuntimeClosureEvidence, ...]
    blocking_failures: tuple[str, ...]
    pending_requirements: tuple[str, ...]
    acceptance_sha256: str

    @property
    def passed(self) -> bool:
        return self.state is EvidenceState.PASS


_REQUIRED_BY_POLICY = {
    "container_pytest": "require_container_pytest",
    "postgres_core": "require_postgres_core",
    "postgres_failover": "require_postgres_failover",
    "postgres_restore": "require_postgres_restore",
    "binance_real_market_dryrun": "require_binance_real_market_dryrun",
    "fault_campaign": "require_fault_campaign",
    "shadow_24h": "require_shadow_24h",
    "shadow_72h": "require_shadow_72h",
    "two_year_backtest": "require_two_year_backtest",
    "position_behavior": "require_position_behavior",
}


def evaluate_runtime_closure_acceptance(
    evidence: Mapping[str, RuntimeClosureEvidence],
    *,
    policy: RuntimeClosurePolicy | None = None,
) -> RuntimeClosureAcceptance:
    effective = policy or RuntimeClosurePolicy()
    failures: list[str] = []
    pending: list[str] = []
    selected: list[RuntimeClosureEvidence] = []
    for name, policy_field in _REQUIRED_BY_POLICY.items():
        if not bool(getattr(effective, policy_field)):
            continue
        item = evidence.get(name)
        if item is None:
            pending.append(name + ":MISSING")
            selected.append(RuntimeClosureEvidence(name, EvidenceState.PENDING, detail="missing evidence"))
            continue
        selected.append(item)
        if item.state is EvidenceState.FAIL:
            failures.append(name)
        elif item.state is not EvidenceState.PASS:
            pending.append(name)
    selected.sort(key=lambda item: item.name)
    if failures:
        state = EvidenceState.FAIL
    elif pending:
        state = EvidenceState.PENDING
    else:
        state = EvidenceState.PASS
    payload = [
        {"name": x.name, "state": x.state.value, "digest": x.digest, "detail": x.detail}
        for x in selected
    ]
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return RuntimeClosureAcceptance(state, tuple(selected), tuple(failures), tuple(pending), digest)


RUNTIME_CLOSURE_EVIDENCE_REGISTRY_SCHEMA = "hprl-runtime-closure-r2-evidence-registry-v1"


def required_runtime_closure_evidence_names() -> tuple[str, ...]:
    return tuple(_REQUIRED_BY_POLICY)


def _evidence_payload(evidence: Mapping[str, RuntimeClosureEvidence]) -> dict[str, dict[str, str]]:
    return {
        name: {
            "state": item.state.value,
            "digest": item.digest,
            "detail": item.detail,
        }
        for name, item in sorted(evidence.items())
    }


def _evidence_registry_digest(payload: Mapping[str, object]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".hprl-r2-evidence-", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_runtime_closure_evidence_registry(
    path: str | Path,
    evidence: Mapping[str, RuntimeClosureEvidence],
) -> str:
    selected = {
        name: evidence.get(name, RuntimeClosureEvidence(name, EvidenceState.PENDING, detail="not collected"))
        for name in required_runtime_closure_evidence_names()
    }
    body: dict[str, object] = {
        "schema": RUNTIME_CLOSURE_EVIDENCE_REGISTRY_SCHEMA,
        "release": HPRL_RUNTIME_CLOSURE_RELEASE,
        "evidence": _evidence_payload(selected),
    }
    registry_sha256 = _evidence_registry_digest(body)
    body["registry_sha256"] = registry_sha256
    _atomic_json_write(Path(path), body)
    return registry_sha256


def load_runtime_closure_evidence_registry(path: str | Path) -> dict[str, RuntimeClosureEvidence]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != RUNTIME_CLOSURE_EVIDENCE_REGISTRY_SCHEMA:
        raise ValueError("invalid runtime closure evidence registry schema")
    claimed = str(raw.get("registry_sha256", ""))
    body = {key: value for key, value in raw.items() if key != "registry_sha256"}
    if len(claimed) != 64 or _evidence_registry_digest(body) != claimed:
        raise ValueError("runtime closure evidence registry digest mismatch")
    rows = raw.get("evidence")
    if not isinstance(rows, dict):
        raise ValueError("runtime closure evidence registry is missing evidence map")
    result: dict[str, RuntimeClosureEvidence] = {}
    for name in required_runtime_closure_evidence_names():
        item = rows.get(name)
        if not isinstance(item, dict):
            result[name] = RuntimeClosureEvidence(name, EvidenceState.PENDING, detail="not collected")
            continue
        result[name] = RuntimeClosureEvidence(
            name=name,
            state=EvidenceState(str(item.get("state", "PENDING"))),
            digest=str(item.get("digest", "")),
            detail=str(item.get("detail", "")),
        )
    return result


def initialize_runtime_closure_evidence_registry(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        existing = load_runtime_closure_evidence_registry(target)
    else:
        existing = {
            name: RuntimeClosureEvidence(name, EvidenceState.PENDING, detail="not collected")
            for name in required_runtime_closure_evidence_names()
        }
    return write_runtime_closure_evidence_registry(target, existing)


def record_runtime_closure_evidence(
    path: str | Path,
    *,
    name: str,
    state: EvidenceState,
    digest: str = "",
    detail: str = "",
) -> str:
    if name not in _REQUIRED_BY_POLICY:
        raise ValueError(f"unknown runtime closure evidence name: {name}")
    target = Path(path)
    if target.is_file():
        current = load_runtime_closure_evidence_registry(target)
    else:
        current = {
            item: RuntimeClosureEvidence(item, EvidenceState.PENDING, detail="not collected")
            for item in required_runtime_closure_evidence_names()
        }
    current[name] = RuntimeClosureEvidence(name=name, state=state, digest=digest, detail=detail)
    return write_runtime_closure_evidence_registry(target, current)
