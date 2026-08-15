"""Hedge CLI registration isolated from Freqtrade's upstream parser layout."""

from __future__ import annotations

from argparse import ArgumentParser, _SubParsersAction
from typing import Any


def _start_hedge_db(args: dict[str, Any]) -> Any:
    """Load the Hedge database command only when it is executed."""
    from freqtrade.commands.hedge_db_commands import start_hedge_db

    return start_hedge_db(args)


def _start_hedge_backtesting(args: dict[str, Any]) -> Any:
    """Load the Hedge backtesting runtime only when it is executed."""
    from freqtrade.commands.hedge_runtime_commands import start_hedge_backtesting

    return start_hedge_backtesting(args)


def _start_hedge_paper(args: dict[str, Any]) -> Any:
    """Load the durable Hedge Paper runtime only when it is executed."""
    from freqtrade.commands.hedge_runtime_commands import start_hedge_paper

    return start_hedge_paper(args)


def _start_hedge_readonly_check(args: dict[str, Any]) -> Any:
    """Load the Binance readonly preflight only when it is executed."""
    from freqtrade.commands.hedge_readonly_commands import start_hedge_readonly_check

    return start_hedge_readonly_check(args)


def _start_hedge_native_audit(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_native_audit
    return start_hedge_native_audit(args)


def _start_hedge_model_check(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_model_check
    return start_hedge_model_check(args)


def _start_hedge_contracts(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_contracts
    return start_hedge_contracts(args)


def _start_hedge_result_analysis(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_result_analysis
    return start_hedge_result_analysis(args)


def _start_hedge_lookahead_file_analysis(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_lookahead_file_analysis
    return start_hedge_lookahead_file_analysis(args)


def _start_hedge_recursive_file_analysis(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_recursive_file_analysis
    return start_hedge_recursive_file_analysis(args)


def _start_hedge_native_hyperopt(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_native_commands import start_hedge_native_hyperopt
    return start_hedge_native_hyperopt(args)


def _start_hedge_research_optimize(args: dict[str, Any]) -> Any:
    """Load the resumable research optimizer without replacing native hyperopt."""
    from freqtrade.commands.hedge_runtime_commands import start_hedge_research_optimize

    return start_hedge_research_optimize(args)


def _start_hedge_research_capabilities(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_research_commands import start_hedge_research_capabilities

    return start_hedge_research_capabilities(args)


def _start_hedge_research_validate(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_research_commands import start_hedge_research_validate

    return start_hedge_research_validate(args)


def _start_hedge_runtime_acceptance(args: dict[str, Any]) -> Any:
    from freqtrade.commands.hedge_acceptance_commands import start_hedge_runtime_acceptance

    return start_hedge_runtime_acceptance(args)


def _register_runtime_acceptance_command(
    subparsers: _SubParsersAction,
    common_parser: ArgumentParser,
    choices: dict[str, Any],
) -> None:
    if "hedge-runtime-acceptance" in choices:
        return
    command = subparsers.add_parser(
        "hedge-runtime-acceptance",
        help="Run the 20-round state-integrity Runtime Acceptance.",
        parents=[common_parser],
    )
    command.set_defaults(func=_start_hedge_runtime_acceptance)
    command.add_argument(
        "--mode",
        dest="hedge_acceptance_mode",
        choices=("deterministic", "live-readonly"),
        default="deterministic",
    )
    command.add_argument("--project-root", dest="project_root")
    command.add_argument("--output-directory", dest="hedge_acceptance_output_directory")
    command.add_argument("--acceptance-db", dest="hedge_acceptance_database")
    command.add_argument(
        "--observe-seconds", dest="hedge_acceptance_observe_seconds", type=float, default=60.0
    )
    command.add_argument(
        "--target-soak-stage",
        dest="hedge_acceptance_target_soak_stage",
        choices=("smoke", "1h", "6h", "24h", "72h"),
        default="smoke",
    )


def register_hedge_subcommands(  # noqa: C901
    manager: Any,
    subparsers: _SubParsersAction,
    common_parser: ArgumentParser,
    strategy_parser: ArgumentParser,
    *,
    trade_options: list[str],
    backtest_options: list[str],
) -> None:
    """Idempotently register Hedge-only commands without importing runtimes."""

    choices = subparsers.choices
    if "hedge-paper" not in choices:
        hedge_paper_cmd = subparsers.add_parser(
            "hedge-paper",
            help="Run SQL-durable Hedge Paper with real DataProvider OHLCV.",
            parents=[common_parser, strategy_parser],
        )
        hedge_paper_cmd.set_defaults(func=_start_hedge_paper)
        manager._build_args(optionlist=trade_options, parser=hedge_paper_cmd)

    if "hedge-backtesting" not in choices:
        hedge_backtesting_cmd = subparsers.add_parser(
            "hedge-backtesting",
            help="Backtest dual-leg Hedge planning with next-bar execution.",
            parents=[common_parser, strategy_parser],
        )
        hedge_backtesting_cmd.set_defaults(func=_start_hedge_backtesting)
        manager._build_args(optionlist=backtest_options, parser=hedge_backtesting_cmd)
        hedge_backtesting_cmd.add_argument(
            "--hedge-export-filename",
            dest="hedge_export_filename",
            help="Write the Hedge result JSON to this path.",
        )
        hedge_backtesting_cmd.add_argument(
            "--hedge-export-events",
            dest="hedge_export_events",
            action="store_true",
            default=False,
            help="Include the full event ledger in the result JSON.",
        )

    if "hedge-db" not in choices:
        hedge_db = subparsers.add_parser(
            "hedge-db",
            help="Plan, migrate, or verify the explicitly gated Hedge schema.",
            parents=[common_parser],
        )
        hedge_db.set_defaults(func=_start_hedge_db)
        hedge_db.add_argument(
            "--action",
            dest="hedge_db_action",
            choices=("status", "plan", "migrate", "verify"),
            default="status",
        )
        hedge_db.add_argument("--db-url", dest="db_url")
        hedge_db.add_argument(
            "--backup-directory",
            dest="hedge_backup_directory",
        )

    if "hedge-readonly-check" not in choices:
        readonly_check = subparsers.add_parser(
            "hedge-readonly-check",
            help="Run a fail-closed Binance REST-only account preflight.",
            parents=[common_parser],
        )
        readonly_check.set_defaults(func=_start_hedge_readonly_check)
        readonly_check.add_argument(
            "--output",
            dest="hedge_readonly_output",
            help="Write the sanitized preflight JSON report to this path.",
        )
        readonly_check.add_argument(
            "--include-history",
            dest="hedge_readonly_include_history",
            action="store_true",
            default=False,
            help="Also collect order, fill and income history during preflight.",
        )
    if "hedge-native-audit" not in choices:
        command = subparsers.add_parser(
            "hedge-native-audit",
            help="Run fail-closed Hedge native convergence source checks.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_native_audit)
        command.add_argument("--project-root", dest="project_root")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-model-check" not in choices:
        command = subparsers.add_parser(
            "hedge-model-check",
            help="Validate a Hedge FreqAI model manifest and expiry.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_model_check)
        command.add_argument("--manifest", dest="hedge_model_manifest", required=True)
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-native-contracts" not in choices:
        command = subparsers.add_parser(
            "hedge-native-contracts",
            help="Print Hedge Hyperopt and dual-leg RL contracts.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_contracts)
        command.add_argument("--output", dest="hedge_native_output")
    if "hedge-result-analysis" not in choices:
        command = subparsers.add_parser(
            "hedge-result-analysis",
            help="Rank Hedge v4 result artifacts.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_result_analysis)
        command.add_argument("--result", dest="hedge_result_files", action="append", required=True)
        command.add_argument("--metric", dest="hedge_result_metric", default="total_return")
        command.add_argument("--ascending", dest="hedge_result_ascending", action="store_true")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-lookahead-analysis" not in choices:
        command = subparsers.add_parser(
            "hedge-lookahead-analysis",
            help="Compare full and truncated Hedge result prefixes.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_lookahead_file_analysis)
        command.add_argument("--baseline", dest="hedge_baseline_result", required=True)
        command.add_argument(
            "--candidate",
            dest="hedge_candidate_results",
            action="append",
            required=True,
            help="CUTOFF=PATH",
        )
        command.add_argument("--field", dest="hedge_analysis_fields", action="append")
        command.add_argument("--tolerance", dest="hedge_analysis_tolerance", default="0")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-recursive-analysis" not in choices:
        command = subparsers.add_parser(
            "hedge-recursive-analysis",
            help="Compare Hedge terminal outputs across startup windows.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_recursive_file_analysis)
        command.add_argument(
            "--result",
            dest="hedge_recursive_results",
            action="append",
            required=True,
            help="STARTUP_CANDLES=PATH",
        )
        command.add_argument("--compare-tail", dest="hedge_compare_tail", type=int, default=1)
        command.add_argument("--tolerance", dest="hedge_analysis_tolerance", default="0")
        command.add_argument("--output", dest="hedge_native_output")

    if "hedge-research-optimize" not in choices:
        command = subparsers.add_parser(
            "hedge-research-optimize",
            help="Run resumable research-grade Hedge parameter optimization.",
            parents=[common_parser, strategy_parser],
        )
        command.set_defaults(func=_start_hedge_research_optimize)
        manager._build_args(optionlist=backtest_options, parser=command)
        command.add_argument("--hedge-study-name", dest="hedge_study_name")
        command.add_argument("--hedge-trials", dest="hedge_trials", type=int)
        command.add_argument("--hedge-workers", dest="hedge_workers", type=int)
        command.add_argument(
            "--hedge-sampler",
            dest="hedge_sampler",
            choices=("grid", "random"),
        )
        command.add_argument(
            "--hedge-optimization-output",
            dest="hedge_optimization_output",
        )

    if "hedge-hyperopt" not in choices:
        command = subparsers.add_parser(
            "hedge-hyperopt",
            help="Run Hedge-native parameter search.",
            parents=[common_parser, strategy_parser],
        )
        command.set_defaults(func=_start_hedge_native_hyperopt)
        manager._build_args(optionlist=backtest_options, parser=command)
        command.add_argument("--hedge-epochs", dest="hedge_epochs", type=int, default=10)
        command.add_argument(
            "--hedge-random-state", dest="hedge_random_state", type=int, default=42
        )
        command.add_argument("--hedge-hyperopt-directory", dest="hedge_hyperopt_directory")
        command.add_argument("--output", dest="hedge_native_output")
    if "hedge-research-capabilities" not in choices:
        command = subparsers.add_parser(
            "hedge-research-capabilities",
            help="Show the 200-round Hedge research capability surface.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_research_capabilities)
        command.add_argument("--output", dest="hedge_research_output")

    if "hedge-research-validate" not in choices:
        command = subparsers.add_parser(
            "hedge-research-validate",
            help="Run the fail-fast 200-round Hedge research validation suite.",
            parents=[common_parser],
        )
        command.set_defaults(func=_start_hedge_research_validate)
        command.add_argument("--output", dest="hedge_research_output")

    _register_runtime_acceptance_command(subparsers, common_parser, choices)
