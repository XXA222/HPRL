from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from freqtrade.hedge.backtesting.contracts import Candidate, EngineConfig
from freqtrade.hedge.backtesting.dataset import build_dataset
from freqtrade.hedge.hprl.config import HPRLActionConfig
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.production.acceptance_r2 import evaluate_hprl_v3_production_r2
from freqtrade.hedge.production.binance_dryrun import (
    BinanceDryRunPolicy,
    BinanceDryRunSafetyContext,
    evaluate_binance_dryrun,
)
from freqtrade.hedge.production.hprl_hedge_adapter import (
    HprlHedgeAdapter,
    HprlHedgeAdapterPolicy,
    HprlTargetUnit,
)
from freqtrade.hedge.production.hprl_replay_backtest import (
    HprlReplayBacktestRunner,
    HprlReplayDatasetBuilder,
    TimedHprlTarget,
)
from freqtrade.hedge.production.recovery_checkpoint import (
    DurableRecoveryCheckpoint,
    RecoveryBarrierPolicy,
    RecoveryCheckpointStore,
    RecoveryConvergenceBarrier,
)
from freqtrade.hedge.production.source_convergence import CanonicalSourceSnapshot, build_canonical_source_snapshot
from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry, StrategyTelemetry

NOW = datetime(2026, 8, 15, 0, 0, tzinfo=UTC)
ZERO_HASH = "0" * 64


def _passing_source_snapshot_for_acceptance() -> CanonicalSourceSnapshot:
    """Return a policy-neutral passing source fixture for acceptance-only tests.

    Source convergence itself has dedicated package/live-policy coverage below.
    Acceptance gating must not depend on whether pytest happens to run from a
    canonical package tree or from a dirty-but-attested live workspace.
    """
    return CanonicalSourceSnapshot(
        manifest_schema="test-manifest-v1",
        manifest_version="test",
        manifest_file_count=1,
        manifest_sha256="1" * 64,
        tree_sha256="2" * 64,
        hprl_api_version="test",
        hprl_release="test",
        production_api_version="test",
        production_release="test",
        closed_loop_api_version="test",
        closed_loop_release="test",
        github_baseline_repository="XXA222/HPRL",
        github_baseline_commit="c7411179744a38b3af91a11a91985db2327c77a4",
        required_paths_present=True,
        manifest_matches_workspace=True,
        missing_paths=(),
        manifest_missing_files=(),
        manifest_unexpected_files=(),
        manifest_mismatched_files=(),
        validation_policy="package",
        managed_attestation_verified=False,
        managed_attestation_path="",
        managed_attestation_count=0,
        managed_attestation_overlay_sha256="",
        managed_attestation_target_release="",
        workspace_missing_files=(),
        workspace_unexpected_files=(),
        workspace_mismatched_files=(),
    )


def _intent(long: float, short: float, confidence: float = 1.0) -> PlannedExecutionIntent:
    return PlannedExecutionIntent(
        symbol="BTC/USDT:USDT",
        target_long_exposure=long,
        target_short_exposure=short,
        confidence=confidence,
        model_id="hprl-xqc-r2",
        metadata={"source": "hprl"},
    )


def _adapter() -> HprlHedgeAdapter:
    cfg = HPRLActionConfig(
        leverage=3.0,
        max_leg_margin_ratio=0.40,
        max_gross_margin_ratio=0.80,
        max_abs_net_margin_ratio=0.40,
    )
    return HprlHedgeAdapter(
        HprlHedgeAdapterPolicy.from_hprl_action_config(
            cfg,
            target_unit=HprlTargetUnit.NOTIONAL_EQUITY_RATIO,
        )
    )


def _base_dataset():
    prices = [100, 101, 102, 101, 103, 104, 102, 105]
    events = []
    for index, price in enumerate(prices):
        ts = NOW + timedelta(minutes=index)
        p = Decimal(price)
        events.append(
            BarEvent(
                timestamp=ts,
                symbol="BTC/USDT:USDT",
                open=p,
                high=p + Decimal("1"),
                low=p - Decimal("1"),
                close=p + Decimal("0.5"),
                volume=Decimal("1000"),
            )
        )
        if index == 4:
            events.append(
                FundingEvent(
                    timestamp=ts,
                    symbol="BTC/USDT:USDT",
                    rate=Decimal("0.0001"),
                    mark_price=p,
                )
            )
    events.sort(key=lambda item: item.timestamp)
    return build_dataset(
        events=events,
        dataset_id="r2-base",
        timeframe="1m",
        metadata={"fixture": "r2"},
    )


def _bundle():
    adapter = _adapter()
    targets = (
        TimedHprlTarget(NOW, 1, _intent(0.15, 0.0)),       # 5% margin long
        TimedHprlTarget(NOW + timedelta(minutes=1), 2, _intent(0.36, 0.15)),
        TimedHprlTarget(NOW + timedelta(minutes=2), 3, _intent(0.75, 0.36)),
        TimedHprlTarget(NOW + timedelta(minutes=3), 4, _intent(1.20, 0.75)),
        TimedHprlTarget(NOW + timedelta(minutes=4), 5, _intent(0.75, 0.75)),
        TimedHprlTarget(NOW + timedelta(minutes=5), 6, _intent(0.36, 0.36)),
    )
    return adapter, HprlReplayDatasetBuilder(adapter).build(
        base_dataset=_base_dataset(), targets=targets
    )


def test_hprl_adapter_derives_production_envelope_from_action_config() -> None:
    adapter = _adapter()
    assert adapter.policy.leverage == Decimal("3.0")
    assert adapter.policy.max_leg_notional_ratio == Decimal("1.200")
    assert adapter.policy.max_gross_notional_ratio == Decimal("2.400")
    assert adapter.policy.max_increase_margin_delta == Decimal("0.15")


def test_hprl_adapter_preserves_margin_notional_separation() -> None:
    adapter = _adapter()
    projection = adapter.adapt(_intent(1.2, 0.36), sequence=1, observed_at=NOW, now=NOW)
    assert projection.accepted
    assert projection.long_margin_ratio == Decimal("0.4")
    assert projection.short_margin_ratio == Decimal("0.12")
    assert projection.long_notional_ratio == Decimal("1.2")
    assert projection.short_notional_ratio == Decimal("0.36")


def test_hprl_adapter_rejects_target_outside_margin_envelope() -> None:
    adapter = _adapter()
    projection = adapter.adapt(_intent(1.5, 0.0), sequence=1, observed_at=NOW, now=NOW)
    assert not projection.accepted
    assert projection.long_margin_ratio == 0
    assert "MODEL_TARGET_LONG_LIMIT" in projection.reasons


def test_hprl_adapter_rejects_large_scale_in_but_allows_fast_derisk() -> None:
    adapter = _adapter()
    previous = adapter.adapt(_intent(0.15, 0.0), sequence=1, observed_at=NOW, now=NOW)
    rejected = adapter.adapt(
        _intent(1.2, 0.0),
        sequence=2,
        observed_at=NOW + timedelta(seconds=1),
        now=NOW + timedelta(seconds=1),
        previous=previous,
    )
    assert not rejected.accepted
    high = adapter.adapt(_intent(1.2, 0.0), sequence=10, observed_at=NOW, now=NOW)
    reduced = adapter.adapt(
        _intent(0.15, 0.0),
        sequence=11,
        observed_at=NOW + timedelta(seconds=1),
        now=NOW + timedelta(seconds=1),
        previous=high,
    )
    assert reduced.accepted
    assert reduced.long_margin_ratio == Decimal("0.05")


def test_hprl_adapter_signal_uses_exact_independent_leg_scales_without_net_collapse() -> None:
    adapter = _adapter()
    projection = adapter.adapt(_intent(0.75, 0.36), sequence=1, observed_at=NOW, now=NOW)
    signal = adapter.to_signal_event(projection)
    assert signal.target_net_ratio is None
    assert signal.long_exposure_scale == Decimal("0.625")
    assert signal.short_exposure_scale == Decimal("0.3")
    assert signal.allow_new_risk
    assert signal.regime == "HPRL"


def test_hprl_adapter_live_snapshot_uses_same_semantics_as_replay_signal() -> None:
    adapter = _adapter()
    projection = adapter.adapt(_intent(0.75, 0.36), sequence=1, observed_at=NOW, now=NOW)
    event = adapter.to_signal_event(projection)
    snapshot = adapter.signal_snapshot_kwargs(
        projection,
        timeframe="1m",
        candle_close_time=NOW,
        feature_timestamp=NOW,
    )
    assert snapshot["long_exposure_scale"] == event.long_exposure_scale
    assert snapshot["short_exposure_scale"] == event.short_exposure_scale
    assert snapshot["target_net_ratio"] is event.target_net_ratio is None
    assert snapshot["model_version"] == event.model_version


def test_hprl_planner_profile_turns_model_targets_into_core_targets_not_tactical_noise() -> None:
    profile = _adapter().planner_profile()
    cfg = profile.planner_config
    assert cfg.core_wallet_exposure_long == Decimal("1.200")
    assert cfg.core_wallet_exposure_short == Decimal("1.200")
    assert cfg.tactical_wallet_exposure_long == 0
    assert cfg.tactical_wallet_exposure_short == 0
    assert cfg.max_gross_wallet_exposure == Decimal("2.400")


def test_hprl_replay_replaces_strategy_signals_and_builds_deterministic_dataset() -> None:
    _, bundle = _bundle()
    assert bundle.report.passed
    assert bundle.report.target_count == 6
    assert bundle.dataset.signal_count == 6
    assert bundle.dataset.metadata["hprl_v3"] == "true"
    assert len(bundle.report.projection_chain_sha256) == 64


def test_hprl_replay_backtest_exercises_dual_leg_engine_and_is_deterministic() -> None:
    adapter, bundle = _bundle()
    runner = HprlReplayBacktestRunner(
        bundle=bundle,
        adapter=adapter,
        engine_config=EngineConfig(
            initial_balance=Decimal("1000"),
            leverage=Decimal("3"),
            volume_participation=Decimal("1"),
            max_fill_ratio_per_order=Decimal("1"),
        ),
    )
    evaluation = runner.evaluate(Candidate("r2", {}))
    assert evaluation.result is not None
    assert evaluation.result.report["long_add_count"] > 0
    assert evaluation.result.report["short_add_count"] > 0
    parity = runner.parity(Candidate("r2-parity", {}))
    assert parity.passed
    assert parity.first_state_sha256 == parity.second_state_sha256
    assert parity.first_event_sha256 == parity.second_event_sha256


def test_recovery_checkpoint_is_hash_bound_atomic_and_generation_monotonic(tmp_path: Path) -> None:
    store = RecoveryCheckpointStore(tmp_path / "recovery.json")
    first = DurableRecoveryCheckpoint(
        generation=1,
        created_at=NOW,
        source_release="r2",
        model_id="hprl-xqc-r2",
        evidence_digest="a" * 64,
        reconciliation_digest="b" * 64,
        projection_chain_sha256="c" * 64,
        last_market_sequence=10,
        last_user_sequence=20,
    )
    store.save_atomic(first)
    assert store.load() == first
    with pytest.raises(ValueError):
        store.save_atomic(first)
    raw = (tmp_path / "recovery.json").read_text(encoding="utf-8").replace("hprl-xqc-r2", "hprl-xqc-rX")
    (tmp_path / "recovery.json").write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.load()


def test_recovery_barrier_opens_only_when_checkpoint_and_current_digests_converge() -> None:
    checkpoint = DurableRecoveryCheckpoint(
        generation=1,
        created_at=NOW,
        source_release="r2",
        model_id="hprl-xqc-r2",
        evidence_digest="a" * 64,
        reconciliation_digest="b" * 64,
        projection_chain_sha256="c" * 64,
        last_market_sequence=10,
        last_user_sequence=20,
    )
    barrier = RecoveryConvergenceBarrier(RecoveryBarrierPolicy(max_checkpoint_age=timedelta(minutes=5)))
    good = barrier.evaluate(
        checkpoint,
        orders=(),
        now=NOW + timedelta(seconds=30),
        current_evidence_digest="a" * 64,
        current_reconciliation_digest="b" * 64,
    )
    assert good.passed and good.allow_new_risk
    changed = barrier.evaluate(
        checkpoint,
        orders=(),
        now=NOW + timedelta(seconds=30),
        current_evidence_digest="d" * 64,
        current_reconciliation_digest="b" * 64,
    )
    assert not changed.passed
    assert not changed.allow_new_risk
    assert "RECOVERY_EVIDENCE_DIGEST_CHANGED" in changed.reasons


def _dryrun_cycles() -> tuple[DryRunCycleTelemetry, ...]:
    strategy = StrategyTelemetry(
        long_score=Decimal("0.5"),
        short_score=Decimal("0.25"),
        long_exposure_scale=Decimal("0.5"),
        short_exposure_scale=Decimal("0.25"),
        model_version="hprl-xqc-r2",
        regime="HPRL",
    )
    return tuple(
        DryRunCycleTelemetry(
            cycle_id=f"cycle-{index}",
            account_id="dryrun:binance",
            symbol="BTCUSDT",
            timestamp=NOW + timedelta(minutes=index),
            mark_price=Decimal("100") + index,
            equity=Decimal("1000") + index,
            available_balance=Decimal("700"),
            gross_notional=Decimal("300"),
            net_quantity=Decimal("1"),
            long_quantity=Decimal("2"),
            short_quantity=Decimal("1"),
            long_target_quantity=Decimal("2"),
            short_target_quantity=Decimal("1"),
            strategy=strategy,
        )
        for index in range(4)
    )


def test_binance_dryrun_acceptance_requires_real_market_plus_zero_write_capability() -> None:
    safety = BinanceDryRunSafetyContext(
        exchange="binance",
        operation_mode="dry_run",
        real_market_data=True,
        exchange_write_capability=False,
        simulated_execution=True,
        hedge_mode_semantics=True,
        cross_margin_semantics=True,
        source_release="r2",
        account_namespace="dryrun",
    )
    report = evaluate_binance_dryrun(
        _dryrun_cycles(),
        safety=safety,
        policy=BinanceDryRunPolicy(
            minimum_cycles=4,
            minimum_duration=timedelta(minutes=3),
            maximum_cycle_gap=timedelta(minutes=2),
        ),
    )
    assert report.passed
    assert report.dual_leg_target_observed
    assert report.dual_leg_position_observed
    assert len(report.telemetry_sha256) == 64


def test_binance_dryrun_rejects_any_exchange_write_capability() -> None:
    unsafe = BinanceDryRunSafetyContext(
        exchange="binance",
        operation_mode="dry_run",
        real_market_data=True,
        exchange_write_capability=True,
        simulated_execution=True,
        hedge_mode_semantics=True,
        cross_margin_semantics=True,
        source_release="r2",
        account_namespace="dryrun",
    )
    report = evaluate_binance_dryrun(
        _dryrun_cycles(),
        safety=unsafe,
        policy=BinanceDryRunPolicy(minimum_cycles=1, minimum_duration=timedelta(0)),
    )
    assert not report.passed
    assert "BINANCE_DRYRUN_SAFETY_CONTEXT_INVALID" in report.reasons


def test_r2_acceptance_keeps_environment_and_live_locked_without_real_evidence() -> None:
    adapter, bundle = _bundle()
    runner = HprlReplayBacktestRunner(bundle=bundle, adapter=adapter)
    parity = runner.parity()
    checkpoint = DurableRecoveryCheckpoint(
        generation=1,
        created_at=NOW,
        source_release="r2",
        model_id="hprl-xqc-r2",
        evidence_digest="a" * 64,
        reconciliation_digest="b" * 64,
        projection_chain_sha256=bundle.report.projection_chain_sha256,
        last_market_sequence=10,
        last_user_sequence=20,
    )
    recovery = RecoveryConvergenceBarrier().evaluate(
        checkpoint,
        orders=(),
        now=NOW,
        current_evidence_digest="a" * 64,
        current_reconciliation_digest="b" * 64,
    )
    source = _passing_source_snapshot_for_acceptance()
    report = evaluate_hprl_v3_production_r2(
        source=source,
        replay_build=bundle.report,
        replay_parity=parity,
        recovery=recovery,
        hedge_adapter_ready=True,
    )
    assert report.offline_source_passed
    assert not report.environment_passed
    assert not report.live_ready
    assert "R2_REAL_POSTGRES_EVIDENCE_REQUIRED" in report.blockers
    assert "R2_BINANCE_DRYRUN_EVIDENCE_REQUIRED" in report.blockers


def test_live_source_policy_treats_clean_version_metadata_as_package_only(tmp_path, monkeypatch):
    """Live convergence blocks executable drift, not package-only version metadata."""
    import json
    import shutil

    from freqtrade.hedge.production import source_convergence as source_module

    project = tmp_path / "project"
    source_root = Path(__file__).resolve().parents[3]
    shutil.copytree(source_root, project, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    version_file = project / "CLEAN-MAINLINE-VERSION.txt"
    version_file.write_text(version_file.read_text(encoding="utf-8") + "live-metadata-drift\n", encoding="utf-8")

    manifest = json.loads((project / "CLEAN-MAINLINE-MANIFEST.json").read_text(encoding="utf-8"))
    managed = []
    for row in manifest["files"]:
        rel = row["path"]
        if rel.startswith("freqtrade/hedge/production/") or rel in {
            "freqtrade/hedge/execution/orchestrator.py",
            "freqtrade/hedge/execution/unknown_supervisor.py",
            "freqtrade/hedge/execution/production_runtime.py",
            "freqtrade/hedge/execution/binance_usdm_adapter.py",
            "freqtrade/hedge/integration/production_main_loop.py",
            "tools/validate_hprl_clean_mainline_200.py",
            "CLEAN-MAINLINE-MANIFEST.json",
        }:
            target = project / rel
            if target.is_file():
                managed.append({"path": rel, "sha256": source_module._sha_file(target)})
    # Live attestation size is release-specific.  Exercise a deliberately small
    # managed set so generic source convergence never embeds a historical count.
    managed = managed[:21]
    overlay_sha256 = "a" * 64
    target_release = "freqtrade-hedge-hprl-v3-runtime-closure-r2"
    attestation = tmp_path / "managed.json"
    attestation.write_text(json.dumps({
        "schema": "hprl-v3-closed-loop-managed-attestation-v1",
        "project_root": str(project.resolve()),
        "target_release": target_release,
        "overlay_sha256": overlay_sha256,
        "managed_count": len(managed),
        "files": managed,
    }), encoding="utf-8")

    monkeypatch.setenv("HPRL_SOURCE_VALIDATION_POLICY", "live")
    monkeypatch.setenv("HPRL_MANAGED_ATTESTATION", str(attestation))
    monkeypatch.setenv("HPRL_EXPECTED_MANAGED_OVERLAY_SHA256", overlay_sha256)
    monkeypatch.setenv("HPRL_EXPECTED_MANAGED_TARGET_RELEASE", target_release)
    snapshot = source_module.build_canonical_source_snapshot(project)
    assert snapshot.passed
    assert "CLEAN-MAINLINE-VERSION.txt" in snapshot.workspace_mismatched_files
    assert "CLEAN-MAINLINE-VERSION.txt" not in snapshot.manifest_mismatched_files
    assert snapshot.managed_attestation_count == len(managed)
    assert snapshot.managed_attestation_overlay_sha256 == overlay_sha256
    assert snapshot.managed_attestation_target_release == target_release

    monkeypatch.setenv("HPRL_EXPECTED_MANAGED_OVERLAY_SHA256", "b" * 64)
    stale = source_module.build_canonical_source_snapshot(project)
    assert not stale.passed
    assert not stale.managed_attestation_verified
    monkeypatch.setenv("HPRL_EXPECTED_MANAGED_OVERLAY_SHA256", overlay_sha256)

    critical = project / "freqtrade/hedge/production/closed_loop.py"
    critical.write_text(critical.read_text(encoding="utf-8") + "\n# critical drift\n", encoding="utf-8")
    snapshot = source_module.build_canonical_source_snapshot(project)
    assert not snapshot.passed
    assert "freqtrade/hedge/production/closed_loop.py" in snapshot.manifest_mismatched_files
