"""Final staged acceptance for HPRL V3 closed-loop production integration."""
from __future__ import annotations

from dataclasses import dataclass

from .closed_loop_dryrun import ClosedLoopDryRunAcceptanceReport
from .closed_loop_recovery import ClosedLoopRecoveryReport
from .hprl_replay_backtest import HprlReplayBuildReport, HprlReplayParityReport
from .postgres_acceptance import PostgresR2AcceptanceReport
from .release import ProductionReadinessReport
from .shadow import ShadowQualification
from .source_convergence import CanonicalSourceSnapshot

HPRL_V3_CLOSED_LOOP_API_VERSION = "3.1"
HPRL_V3_CLOSED_LOOP_RELEASE = "freqtrade-hedge-hprl-v3-closed-loop-r1"


@dataclass(frozen=True, slots=True)
class HprlV3ClosedLoopAcceptanceReport:
    source_converged: bool
    dual_leg_adapter_ready: bool
    replay_parity_ready: bool
    closed_loop_recovery_ready: bool
    postgres_ready: bool
    binance_dryrun_ready: bool
    shadow_24h_ready: bool
    shadow_72h_ready: bool
    production_spine_ready: bool
    offline_source_passed: bool
    environment_passed: bool
    live_candidate_ready: bool
    live_ready: bool
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.live_ready


def evaluate_hprl_v3_closed_loop(
    *,
    source: CanonicalSourceSnapshot,
    replay_build: HprlReplayBuildReport,
    replay_parity: HprlReplayParityReport,
    recovery: ClosedLoopRecoveryReport,
    dual_leg_adapter_ready: bool,
    postgres: PostgresR2AcceptanceReport | None = None,
    binance_dryrun: ClosedLoopDryRunAcceptanceReport | None = None,
    shadow_24h: ShadowQualification | None = None,
    shadow_72h: ShadowQualification | None = None,
    production: ProductionReadinessReport | None = None,
) -> HprlV3ClosedLoopAcceptanceReport:
    blockers: list[str] = []
    source_ok = source.passed
    adapter_ok = bool(dual_leg_adapter_ready)
    replay_ok = replay_build.passed and replay_parity.passed
    recovery_ok = recovery.passed and recovery.allow_new_risk
    postgres_ok = bool(postgres and postgres.passed)
    dryrun_ok = bool(binance_dryrun and binance_dryrun.passed)
    shadow24_ok = bool(shadow_24h and shadow_24h.target == "24h" and shadow_24h.passed)
    shadow72_ok = bool(shadow_72h and shadow_72h.target == "72h" and shadow_72h.passed)
    production_ok = bool(production and production.passed)

    checks = (
        (source_ok, "CLOSED_LOOP_SOURCE_NOT_CONVERGED"),
        (adapter_ok, "CLOSED_LOOP_DUAL_LEG_ADAPTER_NOT_READY"),
        (replay_ok, "CLOSED_LOOP_REPLAY_PARITY_NOT_READY"),
        (recovery_ok, "CLOSED_LOOP_RECOVERY_NOT_CONVERGED"),
        (postgres_ok, "CLOSED_LOOP_REAL_POSTGRES_EVIDENCE_REQUIRED"),
        (dryrun_ok, "CLOSED_LOOP_BINANCE_DRYRUN_EVIDENCE_REQUIRED"),
        (shadow24_ok, "CLOSED_LOOP_SHADOW_24H_REQUIRED"),
        (shadow72_ok, "CLOSED_LOOP_SHADOW_72H_REQUIRED"),
        (production_ok, "CLOSED_LOOP_PRODUCTION_SPINE_NOT_READY"),
    )
    blockers.extend(reason for passed, reason in checks if not passed)
    offline = source_ok and adapter_ok and replay_ok and recovery_ok
    environment = offline and postgres_ok and dryrun_ok
    candidate = environment and shadow24_ok and shadow72_ok
    live = candidate and production_ok
    return HprlV3ClosedLoopAcceptanceReport(
        source_converged=source_ok,
        dual_leg_adapter_ready=adapter_ok,
        replay_parity_ready=replay_ok,
        closed_loop_recovery_ready=recovery_ok,
        postgres_ready=postgres_ok,
        binance_dryrun_ready=dryrun_ok,
        shadow_24h_ready=shadow24_ok,
        shadow_72h_ready=shadow72_ok,
        production_spine_ready=production_ok,
        offline_source_passed=offline,
        environment_passed=environment,
        live_candidate_ready=candidate,
        live_ready=live,
        blockers=tuple(blockers),
    )
