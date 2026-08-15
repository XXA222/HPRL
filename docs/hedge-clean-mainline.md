# Freqtrade-Hedge Clean Mainline

This tree is the long-term-maintenance mainline.  Historical release packages,
version-numbered implementation directories, merge evidence, and deprecated
compatibility facades are intentionally kept outside the runtime source tree.

## Runtime model

- Project source is executed directly from the repository root with the local
  `.venv` Python: `.venv\Scripts\python.exe -m freqtrade ...`.
- The source tree is not installed with `pip install -e`.
- The project-local `.venv` contains one `_freqtrade_hedge_local_source.pth` file that points to the project root and `ft_client`; it contains no package copy and creates no `egg-info` in the source tree.
- `user_data` is runtime state and is preserved independently from source
  replacement.
- Binance account validation stays fail-closed and readonly until an explicit
  execution stage is enabled.

## Canonical Hedge packages

- `freqtrade.hedge.exchange`: Binance readonly facts and user-stream adapters.
- `freqtrade.hedge.execution`: order intent lifecycle and write safety.
- `freqtrade.hedge.persistence` is represented by the shared
  `freqtrade.persistence` Hedge models/repositories; no parallel legacy Hedge
  persistence facade is retained.
- `freqtrade.hedge.planning`: target and ideal-order planning.
- `freqtrade.hedge.risk`: account/portfolio risk gates.
- `freqtrade.hedge.integration`: production and Paper composition.
- `freqtrade.hedge.operations`: durable Dry-run operational health.
- `freqtrade.hedge.acceptance`: 20-round state-integrity runtime acceptance.
- `freqtrade.hedge.simulation`: deterministic dual-leg simulation.
- `freqtrade.hedge.optimization`: optimization and analysis.
- `freqtrade.hedge.research`: research lifecycle/control plane.
- `freqtrade.freqai.hedge_rl`: current Hedge ML/RL implementation.

## Compatibility policy

A compatibility alias may be accepted at a configuration boundary when it is
needed to migrate an existing local config.  A compatibility alias must map
into the one current implementation; duplicate implementations and versioned
runtime packages are not permitted.

## Source hygiene

Run:

```powershell
& ".\.venv\Scripts\python.exe" tools\validate_clean_mainline.py
```

The validator rejects versioned Hedge source packages, historical release
payloads, generated install metadata, stale merge artifacts, and imports that
point back to archived implementation branches.

## Local source registration

After moving or replacing the project directory, run:

```powershell
& ".\scripts\Configure-Freqtrade-Hedge-LocalSource.ps1"
```

The registration is intentionally limited to two local source roots:

1. the project root for `freqtrade`;
2. `ft_client` for `freqtrade_client`.

This is not an editable package install. Dependencies remain installed in `.venv`,
while both first-party packages are imported directly from the current source tree.

## V1.2 configuration isolation hardening

Clean Mainline V1.2 removes the retired operations key from the JSON schema entirely. Generic Freqtrade schema-default application cannot materialize a legacy Hedge branch. A raw legacy value, when present, is migrated once before schema validation; the operations runtime consumes only `hedge.operations`.
