# Freqtrade-Hedge Production Readiness R1

Production Readiness R1 is the fail-closed promotion spine for the canonical Hedge runtime.
It does **not** replace the existing execution, risk, simulation, reconciliation, HPRL, or
FreqAI implementations.  It defines when those components are allowed to progress from
source validation to paper/shadow, Testnet, live reduce-only, and finally live new-risk.

## Stage ladder

1. `SOURCE_READY`
2. `DATABASE_READY`
3. `REPLAY_READY`
4. `SHADOW_24H`
5. `SHADOW_72H`
6. `TESTNET_READY`
7. `LIVE_CANDIDATE`
8. `LIVE_READY`

`LIVE_CANDIDATE` and `LIVE_READY` are intentionally distinct.  Candidate promotion requires an explicit `LIVE_CANDIDATE_APPROVAL` artifact and can obtain `LIVE_REDUCE` plus an explicitly configured, tightly bounded `LIVE_CANARY_RISK` lease.  `LIVE_CANARY` evidence is deliberately **not** a Candidate prerequisite because the canary has not happened yet.  Only after real canary evidence is appended can `LIVE_NEW_RISK` be leased at `LIVE_READY`.

Requirements are cumulative.  A later stage cannot be reached by supplying only the newest
artifact.  Each artifact is stored as an immutable hash-chained `EvidenceRecord` with an
explicit TTL.  A stale, failed, missing, reordered or tampered record fails closed.

## Production invariants

- An ambiguous exchange submit is queried by `clientOrderId`; it is never blindly resent.
- Any position/order/balance/mode reconciliation drift blocks new risk; an unknown exchange
  order blocks even controlled reduction until resolved.
- Risk admission simulates the candidate order against Cross Margin gross/net exposure,
  initial/maintenance margin, available balance, funding reserve and fee/slippage reserve.
- Controlled reductions are preferred while new-risk actions are clipped/rejected.
- Resume from HALT requires both readiness and reconciliation convergence.
- Live and Testnet write capability is short-lived and evidence-digest bound.
- Candidate live new-risk is never equivalent to production live new-risk: only `MICRO`/`SMALL` canary orders may use `LIVE_CANARY_RISK`, must pass the runtime canary envelope, and ordinary `LIVE_NEW_RISK` remains locked until `LIVE_CANARY` evidence promotes the ledger to `LIVE_READY`.
- ML/RL/HPRL is advisory: an unapproved, drifting, slow or non-finite model falls back to a
  deterministic profile and cannot bypass Planner/Risk/Execution.
- 24h/72h Shadow evidence must include restart recovery and funding-cycle coverage.
- PostgreSQL migration/concurrency/fencing/outbox/deadlock/backup/restore evidence is
  mandatory before live database readiness. SQLite remains valid for tests/paper/replay.
- A live canary degrades to reduce-only on incident, daily-loss, drawdown, gross-exposure or
  open-order limit breach.

## Operational CLI

```bash
python tools/hedge_production_readiness_r1.py init --ledger user_data/production-evidence.json
python tools/hedge_production_readiness_r1.py requirements --stage LIVE_READY
python tools/hedge_production_readiness_r1.py record \
  --ledger user_data/production-evidence.json \
  --kind SOURCE_GATES --status PASS --artifact report.json \
  --producer source-gate --ttl-hours 168
python tools/hedge_production_readiness_r1.py verify --ledger user_data/production-evidence.json
python tools/hedge_production_readiness_r1.py status \
  --ledger user_data/production-evidence.json --stage DATABASE_READY
```

The R1 installer records only **source/offline evidence** it actually produced.  It never
fabricates PostgreSQL, 24/72h Shadow, Testnet or Live Canary evidence and never unlocks
mainnet writes.

## 800-point matrix

`tools/validate_hedge_production_readiness_800.py` executes 16 domains × 50 scenarios:

- stage/capability monotonicity
- evidence hash chain and persistence
- PostgreSQL readiness
- Cross Margin risk envelope
- reconciliation
- crash recovery
- ambiguous submission/idempotency
- control plane/incidents
- recorded replay/canonical state
- 24/72h shadow and SLO
- observability
- fault injection
- model governance
- security/live canary
- final admission/golden strategy
- release/source hygiene

This matrix is a scenario/invariant gate.  It is not a substitute for the natural-time
24/72h soak, a real PostgreSQL instance, Binance Testnet, or small live-canary evidence.
