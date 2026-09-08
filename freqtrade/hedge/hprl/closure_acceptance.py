"""Unified release gate for HPRL risk learning and two-year scalability evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .risk_acceptance import RiskLearningAcceptanceReport
from .two_year_acceptance import RuntimeScaleReport, TwoYearCapacityPlan


@dataclass(frozen=True, slots=True)
class HPRLClosureAcceptanceReport:
    schema: str
    verdict: str
    reasons: tuple[str, ...]
    risk_learning_verdict: str
    capacity_host_fit: bool
    capacity_cuda_fit: bool
    two_year_runtime_verdict: str
    final_release_ready: bool


def evaluate_hprl_closure(
    risk_learning: RiskLearningAcceptanceReport,
    capacity: TwoYearCapacityPlan,
    runtime: RuntimeScaleReport,
) -> HPRLClosureAcceptanceReport:
    """Combine evidence without upgrading provisional/inconclusive results to PASS.

    Final release readiness is deliberately strict:

    * risk/position learning must be demonstrated out-of-sample (``PASS``),
    * the planned two-year dataset/replay layout must fit both host and accelerator budgets,
    * the full historical run itself must complete with a ``PASS`` runtime verdict.

    ``INCONCLUSIVE`` risk evidence or a ``PROVISIONAL`` scale projection therefore remains
    incomplete evidence, even if every other gate is green.
    """

    reasons: list[str] = []
    if risk_learning.verdict != "PASS":
        reasons.append(f"risk_learning_{risk_learning.verdict.lower()}")
    if not capacity.host_fit:
        reasons.append("host_capacity_exceeded")
    if not capacity.cuda_fit:
        reasons.append("cuda_capacity_exceeded")
    if runtime.verdict != "PASS":
        reasons.append(f"two_year_runtime_{runtime.verdict.lower()}")

    ready = not reasons
    return HPRLClosureAcceptanceReport(
        schema="hprl-risk-two-year-closure-v1",
        verdict="PASS" if ready else "BLOCKED",
        reasons=tuple(reasons),
        risk_learning_verdict=risk_learning.verdict,
        capacity_host_fit=bool(capacity.host_fit),
        capacity_cuda_fit=bool(capacity.cuda_fit),
        two_year_runtime_verdict=runtime.verdict,
        final_release_ready=ready,
    )
