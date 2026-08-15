# Clean Mainline Consolidation Report

## Evidence baseline

The pre-consolidation Windows deep audit inventoried 82,222 files. 77,018 were inside the project-local virtual environment, 3,123 were generated/runtime files, and 219 files were versioned source/test candidates. It found only 13 direct imports from non-versioned production code into versioned Hedge implementations, making a controlled migration practical.

The Windows baseline also established strong behavior evidence before consolidation: full offline pytest passed 6009 tests with 26 environment/platform skips; Research quality and Research validation passed; the integrated closed-Bar Paper smoke passed; and runtime source loading resolved to the local project tree.

## Consolidation performed

- Migrated strategy contract from the old version namespace to `freqtrade/hedge/strategies/contract.py`.
- Migrated Dry-run control and telemetry to `freqtrade/hedge/control/` and `freqtrade/hedge/telemetry/`.
- Migrated durable operational runtime to `freqtrade/hedge/operations/`.
- Migrated runtime acceptance to `freqtrade/hedge/acceptance/` and generic acceptance commands.
- Removed all versioned Hedge implementation directories from the active tree.
- Removed versioned Hedge CLI entry points from `pyproject.toml`.
- Removed deprecated flat Binance/persistence/target-position compatibility modules after current callers were migrated.
- Moved versioned tests into current semantic test domains, or archived tests whose only purpose was historical release verification.
- Archived versioned verifier/tool/release trees outside the active project.
- Removed historical release reports/manifests/installers from the project root.
- Replaced the versioned Docker entrypoint with a generic clean-mainline entrypoint.
- Renamed current optimization/backtesting quality modules and removed historical development-round registries from runtime authority.
- Renamed Research and ML/RL validation surfaces to semantic current-mainline names while preserving deterministic validation coverage.
- Retained only a one-way raw-input migration for the retired operations token. It is canonicalized to `hedge.operations` before schema validation and removed from memory; no versioned runtime implementation remains behind it.

## Safety invariants retained

- Closed DataProvider BarEvent remains mandatory for production-equivalent Paper cycles.
- LONG and SHORT remain independent identities.
- Paper/Backtest/Research do not write authoritative exchange projections.
- UNKNOWN order state remains fail-closed and must be recovered rather than blindly resubmitted.
- Writer fencing, readiness, reconciliation and risk gates remain explicit authorities.
- Binance network/live-write evidence is not fabricated by offline source tests.

## Archive policy

Everything removed for historical reasons is stored outside the active project in the separately delivered history archive. That archive is for provenance only and is never installed into the clean mainline.

## Final consolidation refinements

- Renamed code-facing execution budget/income ORM classes to semantic current names. The physical `hedge_r5_*` SQL table names and historical migration IDs remain unchanged because they are durable database ABI, not parallel runtime branches. Renaming those tables would require a separate data migration and is intentionally outside source-tree cleanup.
- Replaced historical adoption seed labels with `CLEAN_MAINLINE_ADOPTION_SEED` / `clean-mainline`.
- Genericized Testnet E2E recovery reasons and user-visible messages; no release-number runtime vocabulary remains there.
- Replaced the deployment supervisor's historical `r37_report` contract with `security_readiness_report`, moved AUTO readiness/runtime evidence under `user_data/`, and made AUTO Python resolution prefer only the project-local `.venv`.
- Added fail-closed detection when current `hedge.operations` and the retired raw-input token are supplied together.

## Deliberately retained compatibility identifiers

The clean mainline does **not** retain historical implementation packages. Two narrow compatibility surfaces remain deliberately:

1. `hedge.operations` is the sole runtime authority. The retired `hedge.r56` token exists only in the raw-input migration helper, is absent from the JSON schema/runtime, and dual-key input fails closed.
2. Physical SQL identifiers such as `hedge_r5_daily_budgets` and historical migration step IDs remain immutable persistence ABI so existing SQLite/PostgreSQL databases can be opened without destructive schema renaming. Code-facing class/function names are current and semantic.

These are migration/data-compatibility contracts, not active release branches.
