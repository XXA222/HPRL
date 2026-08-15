# Freqtrade-Hedge Clean Mainline Architecture

This tree is the long-lived Freqtrade-Hedge development and runtime mainline.
Historical release packages, versioned implementation namespaces and audit evidence are not part of this source tree.

## Runtime model

- Run from the project root with the project-local `.venv` Python: `.venv\Scripts\python.exe -m freqtrade`.
- An editable install is not required for local source authority.
- `user_data/` is runtime-owned and is preserved during clean-mainline upgrades.
- Binance live writes remain outside offline source acceptance. Read-only and simulation paths stay fail-closed.

## Canonical Hedge packages

- `freqtrade/hedge/contracts/` — stable domain contracts and ports.
- `freqtrade/hedge/exchange/` — Binance read-only facts, normalizers and user stream.
- `freqtrade/hedge/execution/` — intent approval, lifecycle, UNKNOWN recovery and execution safety.
- `freqtrade/hedge/planning/` — pure target/ideal-order planning.
- `freqtrade/hedge/risk/` — account/portfolio risk authority.
- `freqtrade/hedge/integration/` — runtime composition and Paper integration.
- `freqtrade/hedge/operations/` — durable Dry-run operational health/readiness.
- `freqtrade/hedge/acceptance/` — deterministic runtime state-integrity acceptance.
- `freqtrade/hedge/simulation/` and `backtesting/` — authoritative event simulation and backtest support.
- `freqtrade/hedge/optimization/` — current parameter optimization engine and quality gates.
- `freqtrade/hedge/research/` — local research control plane and deterministic validation matrix.
- `freqtrade/freqai/hedge_rl/` — current Hedge ML/RL subsystem.
- `freqtrade/hedge/hprl/` — independent CPU/CUDA-selectable high-performance RL research subsystem; parallel to existing RL and integrated only through canonical signal/planning contracts.
  Device authority is HPRL configuration (`auto`, `cpu`, `cuda`, or `cuda:N`); CUDA mode supports device-resident environment/replay tensors, AMP and TF32 without changing the legacy RL subsystem.
- `freqtrade/rpc/api_server/hedge_*` — current read/control API and local UI surfaces.

## Compatibility policy

There are no versioned runtime packages such as `r54`, `r55`, `r56`, `r58` or `p2_h2` in the mainline.
`hedge.operations` is the only runtime authority. The retired `hedge.r56` token is accepted only by the one-way raw-input migration boundary, moved to `hedge.operations` before JSON-schema validation, and removed from the in-memory configuration. It is absent from the JSON schema and from the operations runtime. Explicit dual-key input fails closed. Historical SQL table/migration identifiers are retained only as immutable database ABI.

## Source hygiene

`tools/validate_clean_mainline.py` is the package/workspace authority. It rejects versioned Hedge namespaces, removed compatibility modules, historical release directories, generated Python metadata and source-package runtime artifacts.
`CLEAN-MAINLINE-MANIFEST.json` is the only current source hash authority.


## Local environment authority

The clean mainline is designed for a project-local `.venv` without an editable install. The supported launcher changes directory to the project root and runs `.venv\Scripts\python.exe -m freqtrade`, so Python resolves `freqtrade/` directly from this source tree. Generated `*.egg-info`, build caches and runtime artifacts are forbidden in the source package.
