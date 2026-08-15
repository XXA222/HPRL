from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

from freqtrade.hedge.acceptance.models import (
    AcceptanceStatus,
    HardMetrics,
    RoundEvidence,
    RuntimeAcceptanceReport,
    utc_now,
)


ROUND_TITLES: tuple[tuple[str, str], ...] = (
    ("ACCEPT-01", "真实环境基线和配置审计"),
    ("ACCEPT-02", "Binance Server Time/Clock Skew"),
    ("ACCEPT-03", "Account资产事实"),
    ("ACCEPT-04", "Hedge Mode/Cross/Leverage事实"),
    ("ACCEPT-05", "LONG/SHORT身份"),
    ("ACCEPT-06", "Open Orders全量事实"),
    ("ACCEPT-07", "Trades分页与去重"),
    ("ACCEPT-08", "Income/Funding事实"),
    ("ACCEPT-09", "REST Snapshot模型"),
    ("ACCEPT-10", "User Stream生命周期"),
    ("ACCEPT-11", "ACCOUNT_UPDATE"),
    ("ACCEPT-12", "ORDER_TRADE_UPDATE"),
    ("ACCEPT-13", "重复事件"),
    ("ACCEPT-14", "乱序事件"),
    ("ACCEPT-15", "Event Gap恢复"),
    ("ACCEPT-16", "FAST Reconciliation"),
    ("ACCEPT-17", "DEEP Reconciliation"),
    ("ACCEPT-18", "SQLite Crash Recovery"),
    ("ACCEPT-19", "网络/429/5xx故障注入"),
    ("ACCEPT-20", "长时间运行与Production Readiness"),
)


class RuntimeAcceptanceRoundFailure(RuntimeError):
    """Raised when live acceptance reaches a round that records FAIL."""

    def __init__(self, evidence: RoundEvidence) -> None:
        self.evidence = evidence
        super().__init__(
            f"{evidence.round_id} FAIL: checks={list(evidence.checks)!r}; "
            f"metrics={dict(evidence.metrics)!r}; detail={evidence.detail!r}"
        )


class RuntimeAcceptanceSession:
    def __init__(self, *, baseline_version: str = "clean-mainline", live_evidence: bool) -> None:
        self.baseline_version = baseline_version
        self.live_evidence = live_evidence
        self._rounds: list[RoundEvidence] = []
        self._hard_metrics = HardMetrics()
        self._notes: list[str] = []

    @property
    def rounds(self) -> tuple[RoundEvidence, ...]:
        return tuple(self._rounds)

    @property
    def hard_metrics(self) -> HardMetrics:
        return self._hard_metrics

    def require_last_passed(self) -> RoundEvidence:
        if not self._rounds:
            raise RuntimeError("no runtime acceptance round has been recorded")
        evidence = self._rounds[-1]
        if not evidence.passed:
            raise RuntimeAcceptanceRoundFailure(evidence)
        return evidence

    def note(self, value: str) -> None:
        if value and value not in self._notes:
            self._notes.append(value)

    def set_metric(self, name: str, value: int) -> None:
        if name not in self._hard_metrics.__dataclass_fields__:
            raise KeyError(name)
        self._hard_metrics = replace(self._hard_metrics, **{name: int(value)})

    def record(
        self,
        round_id: str,
        *,
        passed: bool,
        checks: tuple[str, ...] = (),
        metrics: Mapping[str, Any] | None = None,
        detail: str = "",
        started_at: datetime | None = None,
    ) -> RoundEvidence:
        expected_index = len(self._rounds)
        if expected_index >= len(ROUND_TITLES):
            raise RuntimeError("all runtime acceptance rounds are already recorded")
        expected_id, title = ROUND_TITLES[expected_index]
        if round_id != expected_id:
            raise RuntimeError(f"round order violation: expected {expected_id}, got {round_id}")
        if self._rounds and not self._rounds[-1].passed:
            raise RuntimeError("previous round did not PASS; later rounds are gated")
        completed = utc_now()
        evidence = RoundEvidence(
            round_id=round_id,
            title=title,
            status=AcceptanceStatus.PASS if passed else AcceptanceStatus.FAIL,
            started_at=started_at or completed,
            completed_at=completed,
            checks=checks,
            metrics=dict(metrics or {}),
            detail=detail,
        )
        self._rounds.append(evidence)
        return evidence

    def finalize(self) -> RuntimeAcceptanceReport:
        if len(self._rounds) != len(ROUND_TITLES):
            missing = [round_id for round_id, _ in ROUND_TITLES[len(self._rounds) :]]
            raise RuntimeError("incomplete runtime acceptance: " + ",".join(missing))
        return RuntimeAcceptanceReport(
            schema="hedge-runtime-acceptance-v1",
            generated_at=utc_now(),
            baseline_version=self.baseline_version,
            rounds=tuple(self._rounds),
            hard_metrics=self._hard_metrics,
            live_evidence=self.live_evidence,
            notes=tuple(self._notes),
        )
