"""Aggregate acceptance decision for HPRL V3 Production Integration R2."""
from __future__ import annotations

from dataclasses import dataclass

from .binance_dryrun import BinanceDryRunAcceptanceReport
from .hprl_replay_backtest import HprlReplayBuildReport, HprlReplayParityReport
from .postgres_acceptance import PostgresR2AcceptanceReport
from .recovery_checkpoint import RecoveryBarrierReport
from .release import ProductionReadinessReport
from .source_convergence import CanonicalSourceSnapshot

HPRL_V3_PRODUCTION_API_VERSION = "3.0"
HPRL_V3_PRODUCTION_RELEASE = "freqtrade-hedge-hprl-v3-production-integration-r2"


@dataclass(frozen=True, slots=True)
class HprlV3ProductionAcceptanceReport:
    source_converged: bool
    hedge_adapter_ready: bool
    replay_backtest_ready: bool
    crash_recovery_ready: bool
    postgres_ready: bool
    binance_dryrun_ready: bool
    production_spine_ready: bool
    offline_source_passed: bool
    environment_passed: bool
    live_ready: bool
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.live_ready


def evaluate_hprl_v3_production_r2(
    *,
    source: CanonicalSourceSnapshot,
    replay_build: HprlReplayBuildReport,
    replay_parity: HprlReplayParityReport,
    recovery: RecoveryBarrierReport,
    hedge_adapter_ready: bool,
    postgres: PostgresR2AcceptanceReport | None = None,
    binance_dryrun: BinanceDryRunAcceptanceReport | None = None,
    production: ProductionReadinessReport | None = None,
) -> HprlV3ProductionAcceptanceReport:
    blockers: list[str] = []
    source_ok = bool(source.passed)
    if not source_ok:
        blockers.append("R2_SOURCE_NOT_CONVERGED")
    adapter_ok = bool(hedge_adapter_ready)
    if not adapter_ok:
        blockers.append("R2_HPRL_HEDGE_ADAPTER_NOT_READY")
    replay_ok = bool(replay_build.passed and replay_parity.passed)
    if not replay_ok:
        blockers.append("R2_DUAL_LEG_REPLAY_BACKTEST_NOT_READY")
    recovery_ok = bool(recovery.passed and recovery.allow_new_risk)
    if not recovery_ok:
        blockers.append("R2_CRASH_RECOVERY_NOT_CONVERGED")
    postgres_ok = bool(postgres and postgres.passed)
    if not postgres_ok:
        blockers.append("R2_REAL_POSTGRES_EVIDENCE_REQUIRED")
    dryrun_ok = bool(binance_dryrun and binance_dryrun.passed)
    if not dryrun_ok:
        blockers.append("R2_BINANCE_DRYRUN_EVIDENCE_REQUIRED")
    production_ok = bool(production and production.passed and production.target_stage.value == "LIVE_READY")
    if not production_ok:
        blockers.append("R2_EXISTING_PRODUCTION_LIVE_READY_GATE_REQUIRED")
    offline = source_ok and adapter_ok and replay_ok and recovery_ok
    environment = offline and postgres_ok and dryrun_ok
    live = environment and production_ok
    return HprlV3ProductionAcceptanceReport(
        source_converged=source_ok,
        hedge_adapter_ready=adapter_ok,
        replay_backtest_ready=replay_ok,
        crash_recovery_ready=recovery_ok,
        postgres_ready=postgres_ok,
        binance_dryrun_ready=dryrun_ok,
        production_spine_ready=production_ok,
        offline_source_passed=offline,
        environment_passed=environment,
        live_ready=live,
        blockers=tuple(blockers),
    )
