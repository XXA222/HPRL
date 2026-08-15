# Clean Mainline V1.2.1 workspace matrix fix

V1.2 introduced a deterministic 200-point matrix but called it against an installed
workspace while two package-only checks still treated `.venv` and runtime artifacts
as forbidden payload. The matrix therefore could fail immediately after a correct
Clean Mainline installation.

V1.2.1 gives the matrix explicit package/workspace semantics and reuses the canonical
workspace classifier from `tools/validate_clean_mainline.py`.

- package mode stays strict;
- workspace mode ignores `.venv`, `user_data`, `artifacts`, caches and generated
  metadata exactly as the main workspace validator does;
- source, import and manifest checks still cover all active mainline files;
- the Windows validation runner invokes the 200-point matrix with `--workspace-mode`.

This change does not alter trading, execution, exchange, persistence, risk, planning,
Research, MLRL, or Binance behavior.
