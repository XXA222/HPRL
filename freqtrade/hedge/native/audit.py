"""Runtime/source audit contracts used by the clean-mainline runtime verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from json import dumps
from typing import Any, Callable, Iterable, Mapping

from .models import utc_datetime


class AuditStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class AuditCheckResult:
    check_id: str
    title: str
    status: AuditStatus
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=utc_datetime)

    def __post_init__(self) -> None:
        if not self.check_id.strip() or not self.title.strip():
            raise ValueError("audit check id/title are required")
        object.__setattr__(self, "status", AuditStatus(self.status))
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "checked_at", utc_datetime(self.checked_at))


@dataclass(frozen=True, slots=True)
class AuditReport:
    name: str
    results: tuple[AuditCheckResult, ...]
    schema: str = "hedge-native-audit-v1"

    @property
    def passed(self) -> bool:
        return all(item.status not in {AuditStatus.FAIL} for item in self.results)

    @property
    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(item.status is status for item in self.results)
            for status in AuditStatus
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "name": self.name,
            "passed": self.passed,
            "counts": self.counts,
            "results": [
                {
                    "check_id": item.check_id,
                    "title": item.title,
                    "status": item.status.value,
                    "detail": item.detail,
                    "evidence": dict(item.evidence),
                    "checked_at": item.checked_at.isoformat(),
                }
                for item in self.results
            ],
        }
        canonical = dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        payload["sha256"] = sha256(canonical.encode()).hexdigest()
        return payload


AuditCallable = Callable[[], AuditCheckResult | bool | tuple[bool, str]]


class NativeAuditRunner:
    def __init__(self) -> None:
        self._checks: list[tuple[str, str, AuditCallable]] = []

    def add(self, check_id: str, title: str, check: AuditCallable) -> None:
        if any(existing == check_id for existing, _, _ in self._checks):
            raise ValueError(f"duplicate audit check id: {check_id}")
        if not callable(check):
            raise TypeError("audit check must be callable")
        self._checks.append((check_id, title, check))

    def run(self, *, name: str = "Hedge native convergence") -> AuditReport:
        rows: list[AuditCheckResult] = []
        for check_id, title, check in self._checks:
            try:
                value = check()
            except Exception as exc:
                rows.append(
                    AuditCheckResult(
                        check_id,
                        title,
                        AuditStatus.FAIL,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if isinstance(value, AuditCheckResult):
                if value.check_id != check_id:
                    rows.append(
                        AuditCheckResult(
                            check_id,
                            title,
                            AuditStatus.FAIL,
                            f"check returned mismatched id {value.check_id}",
                        )
                    )
                else:
                    rows.append(value)
            elif isinstance(value, tuple):
                passed, detail = value
                rows.append(
                    AuditCheckResult(
                        check_id,
                        title,
                        AuditStatus.PASS if passed else AuditStatus.FAIL,
                        str(detail),
                    )
                )
            elif isinstance(value, bool):
                rows.append(
                    AuditCheckResult(
                        check_id,
                        title,
                        AuditStatus.PASS if value else AuditStatus.FAIL,
                    )
                )
            else:
                rows.append(
                    AuditCheckResult(
                        check_id,
                        title,
                        AuditStatus.FAIL,
                        "unsupported audit return type",
                    )
                )
        return AuditReport(name, tuple(rows))

    @property
    def check_count(self) -> int:
        return len(self._checks)
