#!/usr/bin/env python3
"""Deterministic 200-check Clean Mainline regression matrix.

This gate is intentionally independent from the large pytest suite.  It catches
architectural regressions that previously caused thousands of secondary test
failures, especially configuration-schema/default interactions and the return
of versioned runtime branches.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable


sys.dont_write_bytecode = True

# Reuse the canonical Clean Mainline workspace classifier instead of maintaining
# a second, subtly different exclusion list in this 200-point matrix.
from validate_clean_mainline import manifest_files, should_ignore_workspace_path, source_files

SCHEMA = "freqtrade-hedge-clean-mainline-200-v2"


class Matrix:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.rows: list[dict[str, Any]] = []

    def add(self, category: str, name: str, fn: Callable[[], Any]) -> None:
        number = len(self.rows) + 1
        try:
            detail = fn()
            passed = detail is not False
            if detail is True:
                detail = "ok"
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        self.rows.append(
            {
                "round": number,
                "category": category,
                "name": name,
                "passed": passed,
                "detail": detail,
            }
        )

    def path(self, relative: str) -> Path:
        return self.root / relative

    def exists(self, relative: str) -> str:
        path = self.path(relative)
        if not path.is_file():
            raise AssertionError(f"missing file: {relative}")
        return relative

    def dir_absent(self, relative: str) -> str:
        path = self.path(relative)
        if path.exists():
            raise AssertionError(f"forbidden path exists: {relative}")
        return relative

    def contains(self, relative: str, marker: str) -> str:
        path = self.path(relative)
        text = path.read_text(encoding="utf-8-sig")
        if marker not in text:
            raise AssertionError(f"{relative} missing marker: {marker}")
        return marker

    def not_contains(self, relative: str, marker: str) -> str:
        path = self.path(relative)
        text = path.read_text(encoding="utf-8-sig")
        if marker in text:
            raise AssertionError(f"{relative} contains forbidden marker: {marker}")
        return marker

    def symbol(self, relative: str, symbol: str) -> str:
        path = self.path(relative)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if symbol not in names:
            raise AssertionError(f"{relative} missing top-level symbol: {symbol}")
        return symbol


def _load_schema_extension(root: Path) -> dict[str, Any]:
    path = root / "freqtrade/hedge/config_schema_extension.py"
    spec = importlib.util.spec_from_file_location("_cm_schema_extension_200", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load config schema extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema: dict[str, Any] = {"properties": {}}
    module.extend_config_schema(schema)
    return schema


def _config_dynamic(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    try:
        from freqtrade.exceptions import OperationalException
        from freqtrade.hedge.config import normalize_hedge_config
        from freqtrade.hedge.operations.config import operations_config, validate_operations_config

        ordinary = {
            "dry_run": True,
            "trading_mode": "spot",
            "margin_mode": "",
            "exchange": {"name": "binance", "pair_whitelist": ["ETH/BTC"]},
        }
        first = normalize_hedge_config(ordinary)
        snapshot = deepcopy(ordinary)
        second = normalize_hedge_config(ordinary)

        legacy = {
            "dry_run": True,
            "trading_mode": "spot",
            "margin_mode": "",
            "exchange": {"name": "binance", "pair_whitelist": ["ETH/BTC"]},
            "hedge": {"r56": {"enabled": False}},
        }
        legacy_before_runtime = validate_operations_config(legacy)
        migrated = normalize_hedge_config(legacy)

        conflict = {
            "dry_run": True,
            "trading_mode": "spot",
            "margin_mode": "",
            "exchange": {"name": "binance", "pair_whitelist": ["ETH/BTC"]},
            "hedge": {
                "operations": {"enabled": False},
                "r56": {"enabled": False},
            },
        }
        conflict_failed = False
        try:
            normalize_hedge_config(conflict)
        except OperationalException:
            conflict_failed = True

        return {
            "ordinary_fixed_point": first == second and snapshot == ordinary,
            "ordinary_has_retired": "r56" in ordinary.get("hedge", {}),
            "legacy_pre_runtime": legacy_before_runtime,
            "legacy_migrated": "r56" not in legacy["hedge"]
            and legacy["hedge"].get("operations") == {"enabled": False},
            "legacy_runtime": dict(operations_config(legacy)),
            "migration_result_equal": migrated == normalize_hedge_config(legacy),
            "conflict_failed": conflict_failed,
        }
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def _no_forbidden_imports(root: Path, workspace_mode: bool) -> str:
    prefixes = (
        "freqtrade.hedge.r4",
        "freqtrade.hedge.r5",
        "freqtrade.hedge.r54",
        "freqtrade.hedge.r55",
        "freqtrade.hedge.r56",
        "freqtrade.hedge.r561",
        "freqtrade.hedge.r58",
        "integrate_h3_full",
    )
    findings: list[str] = []
    for path in source_files(root, workspace_mode):
        if path.suffix != ".py":
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("user_data/"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                    findings.append(
                        f"{relative}:{getattr(node, 'lineno', 0)}:{name}"
                    )
    if findings:
        raise AssertionError(findings[:10])
    return "no forbidden imports"


def _python_compile(root: Path, workspace_mode: bool) -> str:
    count = 0
    for path in source_files(root, workspace_mode):
        if path.suffix != ".py":
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("user_data/"):
            continue
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
        count += 1
    return f"{count} Python files compiled"


def _no_generated_payload(root: Path, workspace_mode: bool) -> str:
    """Enforce clean package payload while accepting known workspace state.

    Package mode is intentionally strict: generated metadata, caches, a local
    virtual environment, or runtime artifacts are release-candidate failures.
    Workspace mode ignores exactly the same paths as the canonical
    validate_clean_mainline workspace classifier.
    """

    findings: list[str] = []
    for path in root.rglob("*"):
        if workspace_mode and should_ignore_workspace_path(root, path):
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if any(
            part == "__pycache__"
            or part.endswith(".egg-info")
            or part in {".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
            for part in parts
        ):
            findings.append(rel.as_posix())
        if path.is_file() and path.suffix == ".pyc":
            findings.append(rel.as_posix())
    if findings:
        raise AssertionError(findings[:10])
    return (
        "workspace generated/runtime paths ignored by canonical classifier"
        if workspace_mode
        else "no generated payload"
    )


def _manifest_exact(root: Path, workspace_mode: bool) -> str:
    path = root / "CLEAN-MAINLINE-MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("files", [])
    if not isinstance(rows, list):
        raise AssertionError("manifest files is not a list")
    declared = {row["path"] for row in rows}
    actual = {
        p.relative_to(root).as_posix()
        for p in manifest_files(root, workspace_mode)
    }
    if declared != actual:
        missing = sorted(actual - declared)[:5]
        extra = sorted(declared - actual)[:5]
        raise AssertionError(f"manifest mismatch missing={missing} extra={extra}")
    return f"{len(rows)} manifest rows"


def _windows_matrix_uses_workspace_mode(root: Path) -> str:
    script = (
        root / "scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1"
    ).read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"validate_clean_mainline_200\.py(?:(?!Run-Step).){0,400}--workspace-mode",
        re.DOTALL,
    )
    if not pattern.search(script):
        raise AssertionError(
            "Windows Clean Mainline validation must invoke the 200-point matrix "
            "with --workspace-mode"
        )
    return "200-point matrix uses --workspace-mode"


def build_matrix(root: Path, workspace_mode: bool) -> Matrix:
    m = Matrix(root)
    schema = _load_schema_extension(root)
    hedge_props = schema["properties"]["hedge"]["properties"]
    dynamic = _config_dynamic(root)

    # 001-020: configuration single-authority invariants.
    m.add("config", "main config module exists", lambda: m.exists("freqtrade/hedge/config.py"))
    m.add("config", "migration boundary module exists", lambda: m.exists("freqtrade/hedge/config_migration.py"))
    m.add("config", "operations config module exists", lambda: m.exists("freqtrade/hedge/operations/config.py"))
    m.add("config", "schema extension exists", lambda: m.exists("freqtrade/hedge/config_schema_extension.py"))
    m.add("config", "schema exposes operations", lambda: "operations" in hedge_props or (_ for _ in ()).throw(AssertionError()))
    m.add("config", "schema excludes retired operations key", lambda: "r56" not in hedge_props or (_ for _ in ()).throw(AssertionError()))
    m.add("config", "operations container has no schema default", lambda: "default" not in hedge_props["operations"] or (_ for _ in ()).throw(AssertionError()))
    m.add("config", "main config contains canonical operations key", lambda: m.contains("freqtrade/hedge/config.py", '"operations"'))
    m.add("config", "main config has no retired key literal", lambda: m.not_contains("freqtrade/hedge/config.py", '"r56"'))
    m.add("config", "operations runtime has no retired key literal", lambda: m.not_contains("freqtrade/hedge/operations/config.py", '"r56"'))
    m.add("config", "migration helper defines retired key", lambda: m.contains("freqtrade/hedge/config_migration.py", 'LEGACY_OPERATIONS_KEY = "r56"'))
    m.add("config", "migration helper defines canonical key", lambda: m.contains("freqtrade/hedge/config_migration.py", 'CURRENT_OPERATIONS_KEY = "operations"'))
    m.add("config", "ordinary normalization is fixed point", lambda: dynamic["ordinary_fixed_point"] or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "ordinary config has no retired key", lambda: not dynamic["ordinary_has_retired"] or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "legacy config blocked before normalization", lambda: dynamic["legacy_pre_runtime"] == ("OPERATIONS_CONFIG_NOT_NORMALIZED",) or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "legacy input migrates to operations", lambda: dynamic["legacy_migrated"] or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "migrated runtime reads canonical operations", lambda: dynamic["legacy_runtime"] == {"enabled": False} or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "migration normalization remains idempotent", lambda: dynamic["migration_result_equal"] or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "dual current and retired keys fail closed", lambda: dynamic["conflict_failed"] or (_ for _ in ()).throw(AssertionError(dynamic)))
    m.add("config", "config migration utility exists", lambda: m.exists("tools/migrate_clean_mainline_config.py"))

    # 021-040: clean layout and branch removal.
    for relative in (
        "merge_history",
        "release",
        "verification",
        "project_docs",
        "hedge_port",
        "freqtrade/hedge/r54",
        "freqtrade/hedge/r55",
        "freqtrade/hedge/r56",
        "freqtrade/hedge/r561",
        "freqtrade/hedge/r58",
        "freqtrade/hedge/p2_h2",
    ):
        m.add("layout", f"forbidden path absent: {relative}", lambda relative=relative: m.dir_absent(relative))
    m.add("layout", "no forbidden versioned imports", lambda: _no_forbidden_imports(root, workspace_mode))
    m.add("layout", "clean architecture document exists", lambda: m.exists("CLEAN-MAINLINE-ARCHITECTURE.md"))
    m.add("layout", "clean migration report exists", lambda: m.exists("CLEAN-MAINLINE-MIGRATION-REPORT.md"))
    m.add("layout", "clean version file exists", lambda: m.exists("CLEAN-MAINLINE-VERSION.txt"))
    m.add("layout", "clean manifest exists", lambda: m.exists("CLEAN-MAINLINE-MANIFEST.json"))
    m.add("layout", "clean mainline validator exists", lambda: m.exists("tools/validate_clean_mainline.py"))
    m.add("layout", "200-check validator exists", lambda: m.exists("tools/validate_clean_mainline_200.py"))
    m.add("layout", "local runner exists", lambda: m.exists("scripts/Run-Freqtrade-Hedge-Local.ps1"))
    m.add("layout", "local source registration exists", lambda: m.exists("scripts/Configure-Freqtrade-Hedge-LocalSource.ps1"))

    # 041-060: exchange / readonly / reconciliation surface.
    exchange_specs = (
        ("freqtrade/hedge/exchange/binance_readonly.py", "BinanceReadonlyClient"),
        ("freqtrade/hedge/exchange/binance_readonly.py", "PermissionPolicy"),
        ("freqtrade/hedge/exchange/binance_readonly.py", "BinanceAccountBundle"),
        ("freqtrade/hedge/exchange/binance_user_stream.py", "BinanceUserStream"),
        ("freqtrade/hedge/exchange/binance_user_stream.py", "BinanceEventSequencer"),
        ("freqtrade/hedge/exchange/binance_user_stream.py", "EventDeduplicator"),
        ("freqtrade/hedge/exchange/listen_key.py", "ListenKeyManager"),
        ("freqtrade/hedge/exchange/clock_sync.py", "ClockSynchronizer"),
        ("freqtrade/hedge/exchange/rate_limit.py", "AdaptiveWeightLimiter"),
        ("freqtrade/hedge/exchange/symbol_codec.py", "BinanceSymbol"),
        ("freqtrade/hedge/readonly/service.py", "BinanceReadonlyService"),
        ("freqtrade/hedge/readonly/runtime.py", "BinanceReadonlyRuntime"),
        ("freqtrade/hedge/readonly/calibration.py", "ReadonlyCalibration"),
        ("freqtrade/hedge/readonly/freshness.py", "FreshnessPolicy"),
        ("freqtrade/hedge/readonly/freshness.py", "UserStreamFreshness"),
        ("freqtrade/hedge/readiness/gate.py", "ReadinessGate"),
        ("freqtrade/hedge/readiness/monitor.py", "ReadinessMonitor"),
        ("freqtrade/hedge/readiness/state.py", "ReadinessReport"),
        ("freqtrade/hedge/contracts/events.py", "PositionSnapshot"),
        ("freqtrade/hedge/contracts/types.py", "AccountRiskSnapshot"),
    )
    for relative, symbol in exchange_specs:
        m.add("exchange-readonly", f"{symbol} is canonical", lambda relative=relative, symbol=symbol: m.symbol(relative, symbol))

    # 061-080: execution and persistence safety.
    execution_specs = (
        ("freqtrade/hedge/execution/service.py", "OrderIntent"),
        ("freqtrade/hedge/execution/service.py", "ApprovedOrderIntent"),
        ("freqtrade/hedge/execution/service.py", "ExecutionResult"),
        ("freqtrade/hedge/execution/service.py", "ExecutionService"),
        ("freqtrade/hedge/execution/state_machine.py", "OrderState"),
        ("freqtrade/hedge/execution/state_machine.py", "OrderLifecycle"),
        ("freqtrade/hedge/execution/client_order_id.py", "build_client_order_id"),
        ("freqtrade/hedge/execution/idempotency.py", "InMemoryIdempotencyStore"),
        ("freqtrade/hedge/execution/unknown_resolver.py", "UnknownOrderResolver"),
        ("freqtrade/hedge/execution/unknown_supervisor.py", "UnknownOrderSupervisor"),
        ("freqtrade/hedge/execution/kill_switch.py", "KillSwitch"),
        ("freqtrade/hedge/execution/outbox_dispatcher.py", "OutboxDispatcher"),
        ("freqtrade/hedge/execution/production_gate.py", "ProductionExecutionGate"),
        ("freqtrade/hedge/execution/binance_usdm_adapter.py", "BinanceUSDMExecutionAdapter"),
        ("freqtrade/hedge/integration/repository.py", "PersistenceMirroringReadonlyRepository"),
        ("freqtrade/hedge/integration/central_source.py", "IntegrationSafetyError"),
        ("freqtrade/hedge/integration/central_source.py", "IntegrationReport"),
        ("freqtrade/hedge/concurrency/single_writer.py", "SingleWriterGuard"),
        ("freqtrade/hedge/concurrency/position_lock.py", "PositionLockManager"),
        ("freqtrade/hedge/execution/ledger.py", "InMemoryExecutionLedger"),
    )
    for relative, symbol in execution_specs:
        m.add("execution-persistence", f"{symbol} is canonical", lambda relative=relative, symbol=symbol: m.symbol(relative, symbol))

    # 081-100: planning / risk / simulation.
    prs_specs = (
        ("freqtrade/hedge/planning/ideal_orders.py", "PureHedgePlanner"),
        ("freqtrade/hedge/planning/context.py", "PlanningContext"),
        ("freqtrade/hedge/planning/target.py", "TargetPosition"),
        ("freqtrade/hedge/planning/core_tactical.py", "plan_leg"),
        ("freqtrade/hedge/planning/grid.py", "build_entry_grid"),
        ("freqtrade/hedge/planning/trailing.py", "trailing_confirmed"),
        ("freqtrade/hedge/planning/unstuck.py", "build_unstuck_intent"),
        ("freqtrade/hedge/risk/engine.py", "HedgeRiskEngine"),
        ("freqtrade/hedge/risk/models.py", "RiskDecision"),
        ("freqtrade/hedge/risk/portfolio.py", "RiskPortfolioSnapshot"),
        ("freqtrade/hedge/risk/liquidation.py", "LegLiquidationBuffer"),
        ("freqtrade/hedge/risk/emergency.py", "EmergencyReduceOnlyController"),
        ("freqtrade/hedge/simulation/exchange.py", "BarEvent"),
        ("freqtrade/hedge/simulation/exchange.py", "FillEvent"),
        ("freqtrade/hedge/simulation/exchange.py", "SimulationResult"),
        ("freqtrade/hedge/simulation/cross_wallet.py", "CrossWallet"),
        ("freqtrade/hedge/simulation/matcher.py", "ConservativeMatcher"),
        ("freqtrade/hedge/simulation/funding.py", "FundingEngine"),
        ("freqtrade/hedge/simulation/replay.py", "EventReplayEngine"),
        ("freqtrade/hedge/simulation/reports.py", "build_report"),
    )
    for relative, symbol in prs_specs:
        m.add("planning-risk-simulation", f"{symbol} is canonical", lambda relative=relative, symbol=symbol: m.symbol(relative, symbol))

    # 101-120: integration / paper runtime / state projection.
    integration_specs = (
        ("freqtrade/hedge/integration/paper_runtime.py", "IntegratedPaperHedgeApplication"),
        ("freqtrade/hedge/integration/paper_runtime.py", "PaperCycleResult"),
        ("freqtrade/hedge/integration/controller.py", "HedgeController"),
        ("freqtrade/hedge/integration/controller.py", "HedgeControllerCycle"),
        ("freqtrade/hedge/integration/projection.py", "CentralRuntimeProjection"),
        ("freqtrade/hedge/integration/projection.py", "build_central_projection"),
        ("freqtrade/hedge/integration/candle_cursor.py", "bar_fingerprint"),
        ("freqtrade/hedge/integration/paper_events.py", "PaperAccountEventSink"),
        ("freqtrade/hedge/integration/paper_state.py", "PaperStateStore"),
        ("freqtrade/hedge/integration/paper_risk_gate.py", "apply_new_risk_gate"),
        ("freqtrade/hedge/integration/production_controller.py", "ProductionHedgeController"),
        ("freqtrade/hedge/integration/production_main_loop.py", "ProductionEquivalentHedgeMainLoop"),
        ("freqtrade/hedge/integration/market_data.py", "PaperMarketInput"),
        ("freqtrade/hedge/integration/signal_provider.py", "FreqtradeStrategySignalProvider"),
        ("freqtrade/hedge/integration/strategy_state.py", "StrategyStateStorePort"),
        ("freqtrade/hedge/operations/runtime.py", "DryRunOperationsRuntime"),
        ("freqtrade/hedge/operations/readiness.py", "DryRunReadinessBuilder"),
        ("freqtrade/hedge/operations/checkpoint.py", "RuntimeCheckpointManager"),
        ("freqtrade/hedge/operations/ledger.py", "FeeFundingLedger"),
        ("freqtrade/hedge/operations/risk.py", "PortfolioRiskMonitor"),
    )
    for relative, symbol in integration_specs:
        m.add("integration-paper", f"{symbol} is canonical", lambda relative=relative, symbol=symbol: m.symbol(relative, symbol))

    # 121-140: backtest / optimization / research.
    research_specs = (
        ("freqtrade/hedge/backtesting/runner.py", "HedgeBacktestRunner"),
        ("freqtrade/hedge/backtesting/contracts.py", "BacktestDataset"),
        ("freqtrade/hedge/backtesting/dataset.py", "build_dataset"),
        ("freqtrade/hedge/backtesting/execution_realism.py", "QueueFill"),
        ("freqtrade/hedge/backtesting/walkforward.py", "run_walk_forward"),
        ("freqtrade/hedge/optimization/engine.py", "OptimizationEngine"),
        ("freqtrade/hedge/optimization/pareto.py", "pareto_front"),
        ("freqtrade/hedge/optimization/robust_selection.py", "rank_candidates"),
        ("freqtrade/hedge/optimization/store.py", "StudyStore"),
        ("freqtrade/hedge/optimization/stress.py", "StressScenario"),
        ("freqtrade/hedge/research/service.py", "HedgeResearchService"),
        ("freqtrade/hedge/research/pipeline.py", "ResearchPipelineManager"),
        ("freqtrade/hedge/research/pipeline.py", "ResearchPipelineSpec"),
        ("freqtrade/hedge/research/jobs.py", "ResearchJobStore"),
        ("freqtrade/hedge/research/workspace.py", "ResearchWorkspace"),
        ("freqtrade/hedge/research/walkforward.py", "build_walk_forward_folds"),
        ("freqtrade/hedge/research/promotion.py", "PromotionPolicy"),
        ("freqtrade/hedge/research/validation_matrix.py", "RoundSpec"),
        ("freqtrade/hedge/research/training.py", "build_freqai_override"),
        ("freqtrade/hedge/research/execution.py", "ResearchExecutionManager"),
    )
    for relative, symbol in research_specs:
        m.add("research-backtest", f"{symbol} is canonical", lambda relative=relative, symbol=symbol: m.symbol(relative, symbol))

    # 141-160: current ML/RL subsystem.
    mlrl_specs = (
        ("freqtrade/freqai/hedge_rl/environment.py", "HedgeTradingEnv"),
        ("freqtrade/freqai/hedge_rl/contracts.py", "SeedLedger"),
        ("freqtrade/freqai/hedge_rl/contracts.py", "ActionRiskTier"),
        ("freqtrade/freqai/hedge_rl/features.py", "FeatureSchema"),
        ("freqtrade/freqai/hedge_rl/features.py", "RobustFeatureScaler"),
        ("freqtrade/freqai/hedge_rl/market_data.py", "dataset_fingerprint"),
        ("freqtrade/freqai/hedge_rl/accounting.py", "IdempotentFillLedger"),
        ("freqtrade/freqai/hedge_rl/accounting.py", "FundingLedger"),
        ("freqtrade/freqai/hedge_rl/execution_models.py", "ExecutionAuditTrail"),
        ("freqtrade/freqai/hedge_rl/reward_extensions.py", "RewardExplainer"),
        ("freqtrade/freqai/hedge_rl/env_extensions.py", "HedgeEnvSnapshot"),
        ("freqtrade/freqai/hedge_rl/training_extensions.py", "RecurrentStateManager"),
        ("freqtrade/freqai/hedge_rl/training_extensions.py", "DistributionalValueHead"),
        ("freqtrade/freqai/hedge_rl/training_extensions.py", "AuxiliaryRiskHead"),
        ("freqtrade/freqai/hedge_rl/registry.py", "HedgeModelRegistry"),
        ("freqtrade/freqai/hedge_rl/inference.py", "HedgeInferenceGuard"),
        ("freqtrade/freqai/hedge_rl/planner_adapter.py", "HedgeRLPlannerAdapter"),
        ("freqtrade/freqai/hedge_rl/observation.py", "HedgeObservationBuilder"),
        ("freqtrade/freqai/hedge_rl/actions.py", "HedgeActionCatalog"),
        ("freqtrade/freqai/hedge_rl/reward_extensions.py", "safe_log_equity_return"),
    )
    for relative, symbol in mlrl_specs:
        m.add("mlrl", f"{symbol} is canonical", lambda relative=relative, symbol=symbol: m.symbol(relative, symbol))

    # 161-180: regression tests and quality gates.
    test_specs = (
        "tests/hedge/test_clean_mainline_config_isolation.py",
        "tests/hedge/operations/test_config.py",
        "tests/hedge/persistence/test_central_integration.py",
        "tests/hedge/integration/test_integrated_main_path.py",
        "tests/hedge/integration/test_local_safety_composition.py",
        "tests/hedge/exchange/test_binance_readonly.py",
        "tests/hedge/exchange/test_user_stream.py",
        "tests/hedge/execution/test_service.py",
        "tests/hedge/execution/test_state_machine.py",
        "tests/hedge/planning/test_planner.py",
        "tests/hedge/risk/test_engine.py",
        "tests/hedge/simulation/test_wallet_matcher.py",
        "tests/hedge/research/test_research_control_plane.py",
        "tests/hedge/research/test_validation_matrix.py",
        "tests/hedge/mlrl/test_contracts_matrix.py",
        "tests/hedge/mlrl/test_training_extensions_matrix.py",
        "tools/validate_hedge_research_quality.py",
        "tools/run_hedge_research_validation.py",
        "tools/validate_hedge_mlrl_code_quality.py",
        "tools/run_hedge_mlrl_validation.py",
    )
    for relative in test_specs:
        m.add("tests-quality", f"regression surface exists: {relative}", lambda relative=relative: m.exists(relative))

    # 181-200: packaging / Windows local-venv authority.
    m.add("package-windows", "all project Python compiles", lambda: _python_compile(root, workspace_mode))
    m.add("package-windows", "no generated package payload", lambda: _no_generated_payload(root, workspace_mode))
    m.add("package-windows", "manifest path set is exact", lambda: _manifest_exact(root, workspace_mode))
    m.add("package-windows", "focused config isolation gate is wired", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "Clean config isolation pytest"))
    m.add("package-windows", "pyproject has only current freqtrade console script", lambda: m.contains("pyproject.toml", 'freqtrade = "freqtrade.main:main"'))
    m.add("package-windows", "local source script writes pth", lambda: m.contains("scripts/Configure-Freqtrade-Hedge-LocalSource.ps1", "_freqtrade_hedge_local_source.pth"))
    m.add("package-windows", "local source script includes ft_client", lambda: m.contains("scripts/Configure-Freqtrade-Hedge-LocalSource.ps1", 'ft_client'))
    m.add("package-windows", "local source script does not editable install", lambda: m.not_contains("scripts/Configure-Freqtrade-Hedge-LocalSource.ps1", "pip install -e"))
    m.add("package-windows", "local runner uses project venv", lambda: m.contains("scripts/Run-Freqtrade-Hedge-Local.ps1", ".venv\\Scripts\\python.exe"))
    m.add("package-windows", "Windows validation uses project venv", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", ".venv\\Scripts\\python.exe"))
    m.add("package-windows", "Windows validation runs full Hedge tests", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "tests\\hedge"))
    m.add("package-windows", "Windows validation excludes online exchange suite", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "--ignore=tests\\exchange_online"))
    m.add("package-windows", "Windows validation excludes pip audit from offline", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "--ignore=tests\\test_pip_audit.py"))
    m.add("package-windows", "Windows validation isolates basetemp", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "--basetemp"))
    m.add("package-windows", "Windows validation runs paper smoke", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "hedge_integrated_smoke.py"))
    m.add("package-windows", "Windows validation runs research 200", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "run_hedge_research_validation.py"))
    m.add("package-windows", "Windows validation runs MLRL matrix", lambda: m.contains("scripts/Run-Freqtrade-Hedge-CleanMainline-Validation.ps1", "run_hedge_mlrl_validation.py"))
    m.add("package-windows", "Windows validation runs clean 200 matrix in workspace mode", lambda: _windows_matrix_uses_workspace_mode(root))
    m.add("package-windows", "config migration tool creates backup", lambda: m.contains("tools/migrate_clean_mainline_config.py", "pre-clean-mainline"))
    m.add("package-windows", "clean version identifies v1.2.1", lambda: m.contains("CLEAN-MAINLINE-VERSION.txt", "v1.2.1"))

    if len(m.rows) != 200:
        raise AssertionError(f"expected 200 checks, got {len(m.rows)}")
    return m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workspace-mode",
        action="store_true",
        help="Ignore only canonical installed-workspace/runtime paths; package mode remains strict.",
    )
    args = parser.parse_args(argv)

    matrix = build_matrix(args.project_root, args.workspace_mode)
    passed = sum(row["passed"] for row in matrix.rows)
    payload = {
        "schema": SCHEMA,
        "project_root": str(args.project_root.resolve()),
        "mode": "workspace" if args.workspace_mode else "package",
        "expected": 200,
        "executed": len(matrix.rows),
        "passed": passed,
        "failed": len(matrix.rows) - passed,
        "status": "PASS" if passed == len(matrix.rows) else "FAIL",
        "rounds": matrix.rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
