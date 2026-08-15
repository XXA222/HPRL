"""Durable, resumable Binance USD-M Testnet E2E orchestration.

The orchestrator owns one guarded canary run from read-only preflight through
``/order/test`` validation and, when explicitly requested, one passive GTX
submit/cancel cycle.  Mainnet endpoints are never accepted.  A SQLite journal
makes the workflow idempotent and allows recovery after a process interruption
without blindly resubmitting a possibly successful write.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol
from uuid import UUID, uuid4

from .binance_usdm_adapter import HttpTransport
from .client_order_id import build_client_order_id
from .service import (
    ExecutionOrder,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from .state_machine import OrderLifecycle, OrderState
from .testnet import (
    DEFAULT_TESTNET_MAX_NOTIONAL,
    TESTNET_ALLOWED_SYMBOLS,
    BinanceTestnetCredentials,
    GuardedTestnetRuntime,
    build_guarded_testnet_runtime,
    build_testnet_readonly_runtime,
    evidence_from_ready_testnet_readonly,
    make_testnet_limit_intent,
)
from .testnet_market import BinanceTestnetMarketProbe, TestnetCanaryOrder


class TestnetScenarioMode(StrEnum):
    VALIDATION_ONLY = "VALIDATION_ONLY"
    SUBMIT_CANCEL = "SUBMIT_CANCEL"


class TestnetScenarioState(StrEnum):
    NEW = "NEW"
    PREFLIGHT = "PREFLIGHT"
    READONLY_READY = "READONLY_READY"
    ARMED = "ARMED"
    VALIDATED = "VALIDATED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    CANCELING = "CANCELING"
    VERIFYING = "VERIFYING"
    COMPENSATING = "COMPENSATING"
    COMPLETED = "COMPLETED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    FAILED = "FAILED"


_TERMINAL_STATES = {
    TestnetScenarioState.COMPLETED,
    TestnetScenarioState.RECOVERY_REQUIRED,
    TestnetScenarioState.FAILED,
}
_ALLOWED_TRANSITIONS: dict[TestnetScenarioState, frozenset[TestnetScenarioState]] = {
    TestnetScenarioState.NEW: frozenset({TestnetScenarioState.PREFLIGHT, TestnetScenarioState.FAILED}),
    TestnetScenarioState.PREFLIGHT: frozenset(
        {TestnetScenarioState.READONLY_READY, TestnetScenarioState.FAILED}
    ),
    TestnetScenarioState.READONLY_READY: frozenset(
        {TestnetScenarioState.ARMED, TestnetScenarioState.FAILED}
    ),
    TestnetScenarioState.ARMED: frozenset(
        {TestnetScenarioState.VALIDATED, TestnetScenarioState.FAILED}
    ),
    TestnetScenarioState.VALIDATED: frozenset(
        {
            TestnetScenarioState.SUBMITTING,
            TestnetScenarioState.VERIFYING,
            TestnetScenarioState.FAILED,
        }
    ),
    TestnetScenarioState.SUBMITTING: frozenset(
        {
            TestnetScenarioState.SUBMITTED,
            TestnetScenarioState.COMPENSATING,
            TestnetScenarioState.RECOVERY_REQUIRED,
            TestnetScenarioState.FAILED,
        }
    ),
    TestnetScenarioState.SUBMITTED: frozenset(
        {
            TestnetScenarioState.CANCELING,
            TestnetScenarioState.VERIFYING,
            TestnetScenarioState.COMPENSATING,
            TestnetScenarioState.FAILED,
        }
    ),
    TestnetScenarioState.CANCELING: frozenset(
        {
            TestnetScenarioState.VERIFYING,
            TestnetScenarioState.COMPENSATING,
            TestnetScenarioState.FAILED,
        }
    ),
    TestnetScenarioState.VERIFYING: frozenset(
        {
            TestnetScenarioState.COMPLETED,
            TestnetScenarioState.COMPENSATING,
            TestnetScenarioState.FAILED,
        }
    ),
    TestnetScenarioState.COMPENSATING: frozenset(
        {
            TestnetScenarioState.VERIFYING,
            TestnetScenarioState.RECOVERY_REQUIRED,
            TestnetScenarioState.FAILED,
        }
    ),
    TestnetScenarioState.COMPLETED: frozenset(),
    TestnetScenarioState.RECOVERY_REQUIRED: frozenset(),
    TestnetScenarioState.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class TestnetE2EConfig:
    mode: TestnetScenarioMode = TestnetScenarioMode.VALIDATION_ONLY
    symbol: str = "ETHUSDT"
    position_side: PositionSide = PositionSide.LONG
    target_notional: Decimal = Decimal("10")
    max_order_notional: Decimal = DEFAULT_TESTNET_MAX_NOTIONAL
    passive_offset_bps: int = 1000
    ready_timeout_seconds: float = 120.0
    arm_ttl_seconds: int = 300
    max_used_weight_1m: int = 1800
    max_order_count_1m: int = 900

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, TestnetScenarioMode) else TestnetScenarioMode(self.mode)
        symbol = str(self.symbol).strip().upper().replace("/", "").split(":", 1)[0]
        if symbol not in TESTNET_ALLOWED_SYMBOLS:
            raise ValueError("R3.6 only supports BTCUSDT and ETHUSDT perpetual")
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(self.position_side)
        )
        target = _notional(self.target_notional, "target_notional")
        maximum = _notional(self.max_order_notional, "max_order_notional")
        if target > maximum:
            raise ValueError("target_notional exceeds max_order_notional")
        if not isinstance(self.passive_offset_bps, int) or isinstance(
            self.passive_offset_bps, bool
        ):
            raise TypeError("passive_offset_bps must be an integer")
        if not 100 <= self.passive_offset_bps <= 3000:
            raise ValueError("passive_offset_bps must be in [100, 3000]")
        if not 5 <= float(self.ready_timeout_seconds) <= 600:
            raise ValueError("ready_timeout_seconds must be in [5, 600]")
        if not isinstance(self.arm_ttl_seconds, int) or not 30 <= self.arm_ttl_seconds <= 600:
            raise ValueError("arm_ttl_seconds must be in [30, 600]")
        if not isinstance(self.max_used_weight_1m, int) or not 1 <= self.max_used_weight_1m <= 2400:
            raise ValueError("max_used_weight_1m must be in [1, 2400]")
        if not isinstance(self.max_order_count_1m, int) or not 1 <= self.max_order_count_1m <= 1200:
            raise ValueError("max_order_count_1m must be in [1, 1200]")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "target_notional", target)
        object.__setattr__(self, "max_order_notional", maximum)
        object.__setattr__(self, "ready_timeout_seconds", float(self.ready_timeout_seconds))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "symbol": self.symbol,
            "position_side": self.position_side.value,
            "target_notional": str(self.target_notional),
            "max_order_notional": str(self.max_order_notional),
            "passive_offset_bps": self.passive_offset_bps,
            "ready_timeout_seconds": self.ready_timeout_seconds,
            "arm_ttl_seconds": self.arm_ttl_seconds,
            "max_used_weight_1m": self.max_used_weight_1m,
            "max_order_count_1m": self.max_order_count_1m,
        }

    @property
    def request_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TestnetRunRecord:
    run_id: str
    request_hash: str
    state: TestnetScenarioState
    account_id: str
    actor: str
    config: Mapping[str, Any]
    client_order_id: str | None
    intent: Mapping[str, Any] | None
    last_order_status: str | None
    report: Mapping[str, Any] | None
    error: str | None
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TestnetE2EReport:
    run_id: str
    status: str
    final_state: TestnetScenarioState
    account_id: str
    actor: str
    mode: TestnetScenarioMode
    symbol: str
    position_side: PositionSide
    client_order_id: str | None
    validation_accepted: bool
    submitted_status: str | None
    final_order_status: str | None
    unexpected_fill: bool
    manual_cleanup_required: bool
    logical_requests: int
    write_requests: int
    read_requests: int
    used_weight_1m: int | None
    order_count_1m: int | None
    started_at: datetime
    completed_at: datetime
    resumed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "toolkit_version": "3.7.0",
            "run_id": self.run_id,
            "status": self.status,
            "final_state": self.final_state.value,
            "account_id": self.account_id,
            "actor": self.actor,
            "mode": self.mode.value,
            "symbol": self.symbol,
            "position_side": self.position_side.value,
            "client_order_id": self.client_order_id,
            "validation_accepted": self.validation_accepted,
            "submitted_status": self.submitted_status,
            "final_order_status": self.final_order_status,
            "unexpected_fill": self.unexpected_fill,
            "manual_cleanup_required": self.manual_cleanup_required,
            "logical_requests": self.logical_requests,
            "write_requests": self.write_requests,
            "read_requests": self.read_requests,
            "used_weight_1m": self.used_weight_1m,
            "order_count_1m": self.order_count_1m,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": (self.completed_at - self.started_at).total_seconds(),
            "resumed": self.resumed,
            "error": self.error,
            "mainnet_live_exchange_write": "LOCKED",
            "allowed_symbols": list(TESTNET_ALLOWED_SYMBOLS),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["report_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload


class ReadonlyRuntimeProtocol(Protocol):
    config: Any

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def wait_until_ready(self, *, timeout_seconds: float) -> Any: ...

    def snapshot(self) -> Any: ...


class TestnetRunJournal:
    """SQLite state journal with compare-and-swap transitions and replay safety."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._initialize()

    def __enter__(self) -> "TestnetRunJournal":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Release SQLite resources deterministically.

        Every operation already uses a short-lived connection.  ``close`` also
        checkpoints WAL state and marks this journal instance unusable, which
        makes Windows temporary-file cleanup deterministic.
        """

        if self._closed:
            return
        if self.path.exists():
            with self._connect(allow_closed=True) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("R3.6 journal is closed")

    def claim(
        self,
        *,
        run_id: str,
        request_hash: str,
        account_id: str,
        actor: str,
        config: Mapping[str, Any],
    ) -> tuple[TestnetRunRecord, bool]:
        now = datetime.now(UTC).isoformat()
        config_json = _json(config)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM r36_testnet_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is not None:
                record = self._record(row)
                if record.request_hash != request_hash or record.account_id != account_id:
                    raise RuntimeError("testnet E2E run_id is already bound to different input or account")
                if record.actor != actor:
                    if record.actor == "" and record.state is TestnetScenarioState.COMPLETED:
                        connection.commit()
                        return record, True
                    if record.actor == "":
                        raise RuntimeError(
                            "testnet E2E active run has no actor binding; manual recovery is required"
                        )
                    raise RuntimeError("testnet E2E run_id is already bound to a different actor")
                connection.commit()
                return record, True
            connection.execute(
                """
                INSERT INTO r36_testnet_runs(
                    run_id,request_hash,state,account_id,actor,config_json,version,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    request_hash,
                    TestnetScenarioState.NEW.value,
                    account_id,
                    actor,
                    config_json,
                    0,
                    now,
                    now,
                ),
            )
            self._event(
                connection,
                run_id,
                TestnetScenarioState.NEW,
                {"claimed": True, "actor": actor},
            )
            connection.commit()
        return self.require(run_id), False

    def require(self, run_id: str) -> TestnetRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM r36_testnet_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._record(row)

    def transition(
        self,
        *,
        run_id: str,
        target: TestnetScenarioState,
        details: Mapping[str, Any] | None = None,
        client_order_id: str | None = None,
        intent: Mapping[str, Any] | None = None,
        last_order_status: str | None = None,
        report: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> TestnetRunRecord:
        target_state = target if isinstance(target, TestnetScenarioState) else TestnetScenarioState(target)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM r36_testnet_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            current = self._record(row)
            if target_state not in _ALLOWED_TRANSITIONS[current.state]:
                raise RuntimeError(f"invalid R3.6 transition: {current.state}->{target_state}")
            now = datetime.now(UTC).isoformat()
            next_client = client_order_id if client_order_id is not None else current.client_order_id
            next_intent = intent if intent is not None else current.intent
            next_status = (
                last_order_status if last_order_status is not None else current.last_order_status
            )
            next_report = report if report is not None else current.report
            next_error = error if error is not None else current.error
            connection.execute(
                """
                UPDATE r36_testnet_runs
                SET state=?,client_order_id=?,intent_json=?,last_order_status=?,report_json=?,
                    error=?,version=version+1,updated_at=?
                WHERE run_id=? AND version=?
                """,
                (
                    target_state.value,
                    next_client,
                    None if next_intent is None else _json(next_intent),
                    next_status,
                    None if next_report is None else _json(next_report),
                    next_error,
                    now,
                    run_id,
                    current.version,
                ),
            )
            if connection.total_changes != 1:
                raise RuntimeError("R3.6 journal compare-and-swap failed")
            self._event(connection, run_id, target_state, details or {})
            connection.commit()
        return self.require(run_id)

    def events(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence,state,details_json,occurred_at FROM r36_testnet_events "
                "WHERE run_id=? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return tuple(
            {
                "sequence": int(row["sequence"]),
                "state": str(row["state"]),
                "details": json.loads(str(row["details_json"])),
                "occurred_at": str(row["occurred_at"]),
            }
            for row in rows
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS r36_testnet_runs(
                    run_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    client_order_id TEXT,
                    intent_json TEXT,
                    last_order_status TEXT,
                    report_json TEXT,
                    error TEXT,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS r36_testnet_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES r36_testnet_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS ix_r36_testnet_events_run
                    ON r36_testnet_events(run_id,sequence);
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(r36_testnet_runs)")
            }
            if "actor" not in columns:
                connection.execute(
                    "ALTER TABLE r36_testnet_runs ADD COLUMN actor TEXT NOT NULL DEFAULT ''"
                )

    @contextmanager
    def _connect(
        self,
        *,
        allow_closed: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        if not allow_closed:
            self._ensure_open()
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        run_id: str,
        state: TestnetScenarioState,
        details: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO r36_testnet_events(run_id,state,details_json,occurred_at) VALUES(?,?,?,?)",
            (run_id, state.value, _json(details), datetime.now(UTC).isoformat()),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> TestnetRunRecord:
        return TestnetRunRecord(
            run_id=str(row["run_id"]),
            request_hash=str(row["request_hash"]),
            state=TestnetScenarioState(str(row["state"])),
            account_id=str(row["account_id"]),
            actor=str(row["actor"]),
            config=json.loads(str(row["config_json"])),
            client_order_id=None if row["client_order_id"] is None else str(row["client_order_id"]),
            intent=None if row["intent_json"] is None else json.loads(str(row["intent_json"])),
            last_order_status=(
                None if row["last_order_status"] is None else str(row["last_order_status"])
            ),
            report=None if row["report_json"] is None else json.loads(str(row["report_json"])),
            error=None if row["error"] is None else str(row["error"]),
            version=int(row["version"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )


class TestnetE2EOrchestrator:
    def __init__(
        self,
        *,
        journal: TestnetRunJournal,
        readonly_factory: Callable[..., ReadonlyRuntimeProtocol] = build_testnet_readonly_runtime,
        guarded_factory: Callable[..., GuardedTestnetRuntime] = build_guarded_testnet_runtime,
        market_probe_factory: Callable[..., BinanceTestnetMarketProbe] = BinanceTestnetMarketProbe,
    ) -> None:
        self.journal = journal
        self._readonly_factory = readonly_factory
        self._guarded_factory = guarded_factory
        self._market_probe_factory = market_probe_factory

    async def run(
        self,
        *,
        credentials: BinanceTestnetCredentials,
        config: TestnetE2EConfig,
        arm_token: str,
        expected_arm_token_sha256: str,
        actor: str,
        proxy_url: str | None = None,
        run_id: str | None = None,
        execution_transport: HttpTransport | None = None,
        market_transport: HttpTransport | None = None,
    ) -> TestnetE2EReport:
        if not isinstance(config, TestnetE2EConfig):
            raise TypeError("config must be TestnetE2EConfig")
        token = str(arm_token).strip()
        if len(token) < 16:
            raise ValueError("arm_token must contain at least 16 characters")
        expected_hash = str(expected_arm_token_sha256).strip().lower()
        if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
            raise ValueError("expected_arm_token_sha256 must be a SHA256 hex digest")
        if not hashlib.sha256(token.encode("utf-8")).hexdigest() == expected_hash:
            raise PermissionError("arm token does not match independently configured hash")
        actor_text = str(actor).strip()
        if not actor_text or len(actor_text) > 128:
            raise ValueError("actor is invalid")
        selected_run_id = str(run_id or f"r36-{uuid4().hex}").strip()
        if not selected_run_id or len(selected_run_id) > 128:
            raise ValueError("run_id is invalid")
        record, resumed = self.journal.claim(
            run_id=selected_run_id,
            request_hash=config.request_hash,
            account_id=credentials.account_id,
            actor=actor_text,
            config=config.canonical_payload(),
        )
        if record.state is TestnetScenarioState.COMPLETED and record.report is not None:
            return _report_from_payload(record.report)
        if record.state in {TestnetScenarioState.FAILED, TestnetScenarioState.RECOVERY_REQUIRED}:
            raise RuntimeError(
                f"testnet E2E run is terminal {record.state.value}; inspect evidence and use a new run_id"
            )

        started = record.created_at
        readonly: ReadonlyRuntimeProtocol | None = None
        guarded: GuardedTestnetRuntime | None = None
        market_order: TestnetCanaryOrder | None = None
        submitted_status: str | None = record.last_order_status
        validation_accepted = record.state in {
            TestnetScenarioState.VALIDATED,
            TestnetScenarioState.SUBMITTING,
            TestnetScenarioState.SUBMITTED,
            TestnetScenarioState.CANCELING,
            TestnetScenarioState.VERIFYING,
        }
        unexpected_fill = False
        manual_cleanup_required = False
        try:
            if record.state is TestnetScenarioState.NEW:
                record = self.journal.transition(
                    run_id=selected_run_id,
                    target=TestnetScenarioState.PREFLIGHT,
                    details={"mainnet_write": "LOCKED", "account_id": credentials.account_id},
                )
            readonly = self._readonly_factory(
                credentials=credentials,
                managed_symbols=(config.symbol,),
                proxy_url=proxy_url,
            )
            await readonly.start()
            await readonly.wait_until_ready(timeout_seconds=config.ready_timeout_seconds)
            if record.state in {TestnetScenarioState.PREFLIGHT, TestnetScenarioState.NEW}:
                record = self.journal.transition(
                    run_id=selected_run_id,
                    target=TestnetScenarioState.READONLY_READY,
                    details={"readonly": "READY", "symbol": config.symbol},
                )
            evidence = evidence_from_ready_testnet_readonly(
                readonly=readonly,
                credentials=credentials,
                expected_arm_token_sha256=expected_hash,
                max_order_notional=config.max_order_notional,
                futures_trading_permission=True,
                allow_market_orders=False,
            )
            probe = self._market_probe_factory(
                transport=market_transport,
                proxy_url=proxy_url,
            )
            market_order = probe.passive_order(
                symbol=config.symbol,
                position_side=config.position_side,
                target_notional=config.target_notional,
                max_notional=config.max_order_notional,
                passive_offset_bps=config.passive_offset_bps,
            )
            store = InMemoryExecutionStore()
            guarded = self._guarded_factory(
                credentials=credentials,
                evidence=evidence,
                readonly=readonly,
                proxy_url=proxy_url,
                transport=execution_transport,
                store=store,
            )
            guarded.runtime.exchange.synchronize_clock()
            guarded.arm(
                token=token,
                actor=actor_text,
                confirmed=True,
                ttl_seconds=config.arm_ttl_seconds,
            )
            if record.state is TestnetScenarioState.READONLY_READY:
                record = self.journal.transition(
                    run_id=selected_run_id,
                    target=TestnetScenarioState.ARMED,
                    details={
                        "arm_ttl_seconds": config.arm_ttl_seconds,
                        "actor": actor_text,
                        "arm_policy": "INDEPENDENT_EXPECTED_HASH",
                    },
                )

            if record.intent is None:
                intent = make_testnet_limit_intent(
                    credentials=credentials,
                    symbol=market_order.symbol,
                    position_side=market_order.position_side,
                    quantity=market_order.quantity,
                    limit_price=market_order.limit_price,
                    idempotency_key=f"r36:{selected_run_id}",
                    time_in_force=market_order.time_in_force,
                )
                client_order_id = build_client_order_id(
                    account_id=intent.account_id,
                    symbol=intent.symbol,
                    position_side=intent.position_side.value,
                    idempotency_key=intent.idempotency_key,
                )
                intent_payload = _intent_payload(intent)
            else:
                intent = _intent_from_payload(record.intent)
                client_order_id = record.client_order_id or build_client_order_id(
                    account_id=intent.account_id,
                    symbol=intent.symbol,
                    position_side=intent.position_side.value,
                    idempotency_key=intent.idempotency_key,
                )
                intent_payload = dict(record.intent)
                _seed_store(store, intent, client_order_id, record.last_order_status)

            if record.state is TestnetScenarioState.ARMED:
                validation = guarded.validate_order(intent)
                validation_accepted = validation.accepted
                record = self.journal.transition(
                    run_id=selected_run_id,
                    target=TestnetScenarioState.VALIDATED,
                    client_order_id=client_order_id,
                    intent=intent_payload,
                    details={
                        "accepted": validation.accepted,
                        "price": str(market_order.limit_price),
                        "quantity": str(market_order.quantity),
                        "notional": str(market_order.notional),
                        "time_in_force": market_order.time_in_force,
                    },
                )

            if config.mode is TestnetScenarioMode.VALIDATION_ONLY:
                if record.state is TestnetScenarioState.VALIDATED:
                    record = self.journal.transition(
                        run_id=selected_run_id,
                        target=TestnetScenarioState.VERIFYING,
                        details={"mode": "VALIDATION_ONLY"},
                    )
                final_status = None
            else:
                submit_in_this_process = False
                if record.state is TestnetScenarioState.VALIDATED:
                    # Persist the deterministic identity before any order write.  A process
                    # interruption after this point is recovered by query only and never by
                    # blind resubmission.
                    record = self.journal.transition(
                        run_id=selected_run_id,
                        target=TestnetScenarioState.SUBMITTING,
                        client_order_id=client_order_id,
                        intent=intent_payload,
                        last_order_status=OrderState.UNKNOWN.value,
                        details={
                            "client_order_id": client_order_id,
                            "write_may_start": True,
                            "recovery_policy": "QUERY_ONLY_NO_RESUBMIT",
                        },
                    )
                    submit_in_this_process = True

                if record.state is TestnetScenarioState.SUBMITTING:
                    if submit_in_this_process:
                        result = guarded.submit(intent)
                        submitted_status = result.order.lifecycle.status.value
                        unexpected_fill = result.order.lifecycle.filled_quantity > 0
                        record = self.journal.transition(
                            run_id=selected_run_id,
                            target=TestnetScenarioState.SUBMITTED,
                            client_order_id=result.order.client_order_id,
                            intent=intent_payload,
                            last_order_status=submitted_status,
                            details={
                                "status": submitted_status,
                                "exchange_order_id": result.order.lifecycle.exchange_order_id,
                            },
                        )
                    else:
                        _seed_store(
                            store,
                            intent,
                            client_order_id,
                            OrderState.UNKNOWN.value,
                        )
                        guarded.idempotency.reserve(intent.idempotency_key)
                        recovered = None
                        for delay in (0.0, 0.25, 0.75):
                            if delay:
                                await asyncio.sleep(delay)
                            recovered = guarded.runtime.engine.refresh_order(client_order_id)
                            if recovered.order.lifecycle.status is not OrderState.UNKNOWN:
                                break
                        if recovered is None or recovered.order.lifecycle.status is OrderState.UNKNOWN:
                            telemetry = guarded.runtime.exchange.telemetry()
                            completed = datetime.now(UTC)
                            report = TestnetE2EReport(
                                run_id=selected_run_id,
                                status="TESTNET_E2E_RECOVERY_REQUIRED_MAINNET_LOCKED",
                                final_state=TestnetScenarioState.RECOVERY_REQUIRED,
                                account_id=credentials.account_id,
                                actor=actor_text,
                                mode=config.mode,
                                symbol=config.symbol,
                                position_side=config.position_side,
                                client_order_id=client_order_id,
                                validation_accepted=True,
                                submitted_status=OrderState.UNKNOWN.value,
                                final_order_status=OrderState.UNKNOWN.value,
                                unexpected_fill=False,
                                manual_cleanup_required=True,
                                logical_requests=telemetry.logical_requests,
                                write_requests=telemetry.write_requests,
                                read_requests=telemetry.read_requests,
                                used_weight_1m=telemetry.used_weight,
                                order_count_1m=telemetry.order_count,
                                started_at=started,
                                completed_at=completed,
                                resumed=True,
                                error="SUBMIT_OUTCOME_UNRESOLVED_NO_RESUBMIT",
                            )
                            payload = report.to_dict()
                            self.journal.transition(
                                run_id=selected_run_id,
                                target=TestnetScenarioState.RECOVERY_REQUIRED,
                                report=payload,
                                last_order_status=OrderState.UNKNOWN.value,
                                error=report.error,
                                details={
                                    "manual_cleanup_required": True,
                                    "blind_resubmit": False,
                                },
                            )
                            return report
                        submitted_status = recovered.order.lifecycle.status.value
                        unexpected_fill = recovered.order.lifecycle.filled_quantity > 0
                        record = self.journal.transition(
                            run_id=selected_run_id,
                            target=TestnetScenarioState.SUBMITTED,
                            client_order_id=client_order_id,
                            intent=intent_payload,
                            last_order_status=submitted_status,
                            details={
                                "status": submitted_status,
                                "recovered_from": "SUBMITTING",
                                "blind_resubmit": False,
                            },
                        )
                else:
                    _seed_store(store, intent, client_order_id, record.last_order_status)
                    guarded.idempotency.reserve(intent.idempotency_key)

                current = guarded.runtime.engine.refresh_order(client_order_id)
                current_status = current.order.lifecycle.status
                submitted_status = submitted_status or current_status.value
                unexpected_fill = current.order.lifecycle.filled_quantity > 0
                if not current.order.lifecycle.terminal:
                    if record.state is TestnetScenarioState.SUBMITTED:
                        record = self.journal.transition(
                            run_id=selected_run_id,
                            target=TestnetScenarioState.CANCELING,
                            last_order_status=current_status.value,
                            details={"client_order_id": client_order_id},
                        )
                    canceled = guarded.cancel(client_order_id)
                    current_status = canceled.order.lifecycle.status
                    unexpected_fill = (
                        unexpected_fill or canceled.order.lifecycle.filled_quantity > 0
                    )
                if record.state in {
                    TestnetScenarioState.SUBMITTING,
                    TestnetScenarioState.SUBMITTED,
                    TestnetScenarioState.CANCELING,
                    TestnetScenarioState.COMPENSATING,
                }:
                    record = self.journal.transition(
                        run_id=selected_run_id,
                        target=TestnetScenarioState.VERIFYING,
                        last_order_status=current_status.value,
                        details={"status": current_status.value},
                    )
                verified = guarded.runtime.engine.refresh_order(client_order_id)
                final_status = verified.order.lifecycle.status.value
                unexpected_fill = unexpected_fill or verified.order.lifecycle.filled_quantity > 0
                manual_cleanup_required = not verified.order.lifecycle.terminal
                if final_status not in {OrderState.CANCELED.value, OrderState.FILLED.value}:
                    manual_cleanup_required = True
                    raise RuntimeError("TESTNET_CANARY_NOT_TERMINAL")
                if unexpected_fill:
                    raise RuntimeError("TESTNET_CANARY_UNEXPECTED_FILL")

            telemetry = guarded.runtime.exchange.telemetry()
            _assert_rate_budget(telemetry, config)
            completed = datetime.now(UTC)
            report = TestnetE2EReport(
                run_id=selected_run_id,
                status="TESTNET_E2E_COMPLETE_MAINNET_LOCKED",
                final_state=TestnetScenarioState.COMPLETED,
                account_id=credentials.account_id,
                actor=actor_text,
                mode=config.mode,
                symbol=config.symbol,
                position_side=config.position_side,
                client_order_id=record.client_order_id,
                validation_accepted=validation_accepted,
                submitted_status=submitted_status,
                final_order_status=final_status,
                unexpected_fill=unexpected_fill,
                manual_cleanup_required=manual_cleanup_required,
                logical_requests=telemetry.logical_requests,
                write_requests=telemetry.write_requests,
                read_requests=telemetry.read_requests,
                used_weight_1m=telemetry.used_weight,
                order_count_1m=telemetry.order_count,
                started_at=started,
                completed_at=completed,
                resumed=resumed,
            )
            payload = report.to_dict()
            self.journal.transition(
                run_id=selected_run_id,
                target=TestnetScenarioState.COMPLETED,
                report=payload,
                last_order_status=final_status,
                details={"status": payload["status"]},
            )
            return report
        except Exception as exc:
            error = f"{type(exc).__name__}:{str(exc)[:1000]}"
            current = self.journal.require(selected_run_id)
            if (
                config.mode is TestnetScenarioMode.SUBMIT_CANCEL
                and guarded is not None
                and current.client_order_id
                and current.state in {
                    TestnetScenarioState.SUBMITTED,
                    TestnetScenarioState.CANCELING,
                    TestnetScenarioState.VERIFYING,
                }
            ):
                try:
                    if current.state is not TestnetScenarioState.COMPENSATING:
                        current = self.journal.transition(
                            run_id=selected_run_id,
                            target=TestnetScenarioState.COMPENSATING,
                            details={"reason": error},
                        )
                    refreshed = guarded.runtime.engine.refresh_order(current.client_order_id)
                    if not refreshed.order.lifecycle.terminal:
                        guarded.cancel(current.client_order_id)
                    verified = guarded.runtime.engine.refresh_order(current.client_order_id)
                    manual_cleanup_required = not verified.order.lifecycle.terminal
                except Exception:
                    manual_cleanup_required = True
            current = self.journal.require(selected_run_id)
            if current.state not in _TERMINAL_STATES:
                target = (
                    TestnetScenarioState.RECOVERY_REQUIRED
                    if manual_cleanup_required
                    and current.client_order_id is not None
                    and current.state in {
                        TestnetScenarioState.SUBMITTING,
                        TestnetScenarioState.COMPENSATING,
                    }
                    else TestnetScenarioState.FAILED
                )
                self.journal.transition(
                    run_id=selected_run_id,
                    target=target,
                    error=error,
                    last_order_status=current.last_order_status,
                    details={
                        "manual_cleanup_required": manual_cleanup_required,
                        "blind_resubmit": False,
                    },
                )
            raise
        finally:
            if readonly is not None:
                await readonly.stop()


# Prevent pytest from treating imported public types as test classes.
for _public_type in (
    TestnetScenarioMode,
    TestnetScenarioState,
    TestnetE2EConfig,
    TestnetRunRecord,
    TestnetE2EReport,
    TestnetRunJournal,
    TestnetE2EOrchestrator,
):
    setattr(_public_type, "__test__", False)


def _assert_rate_budget(telemetry: Any, config: TestnetE2EConfig) -> None:
    if telemetry.used_weight is not None and telemetry.used_weight > config.max_used_weight_1m:
        raise RuntimeError("TESTNET_REQUEST_WEIGHT_BUDGET_EXCEEDED")
    if telemetry.order_count is not None and telemetry.order_count > config.max_order_count_1m:
        raise RuntimeError("TESTNET_ORDER_COUNT_BUDGET_EXCEEDED")
    expected_writes = 1 if config.mode is TestnetScenarioMode.VALIDATION_ONLY else 3
    if telemetry.write_requests > expected_writes:
        raise RuntimeError("TESTNET_WRITE_BUDGET_EXCEEDED")


def _seed_store(
    store: InMemoryExecutionStore,
    intent: OrderIntent,
    client_order_id: str,
    status: str | None,
) -> None:
    if store.get_by_client_order_id(client_order_id) is not None:
        return
    state = OrderState.ACKNOWLEDGED
    if status:
        try:
            state = OrderState(status)
        except ValueError:
            state = OrderState.UNKNOWN
    if state in {OrderState.PREPARED, OrderState.SUBMITTING, OrderState.REJECTED}:
        state = OrderState.UNKNOWN
    lifecycle = OrderLifecycle(
        status=state,
        filled_quantity=(intent.quantity if state is OrderState.FILLED else Decimal("0")),
        average_price=(intent.limit_price if state is OrderState.FILLED else None),
        version=1,
        updated_at=datetime.now(UTC),
        reason="TESTNET_RESTART_RECOVERY",
    )
    store.put(
        ExecutionOrder(
            intent=intent,
            client_order_id=client_order_id,
            approved_quantity=intent.quantity,
            lifecycle=lifecycle,
            created_at=datetime.now(UTC),
        )
    )


def _intent_payload(intent: OrderIntent) -> dict[str, Any]:
    return {
        "account_id": intent.account_id,
        "symbol": intent.symbol,
        "position_side": intent.position_side.value,
        "action": intent.action.value,
        "quantity": str(intent.quantity),
        "idempotency_key": intent.idempotency_key,
        "order_type": intent.order_type.value,
        "limit_price": None if intent.limit_price is None else str(intent.limit_price),
        "reduce_only": intent.reduce_only,
        "intent_id": str(intent.intent_id),
        "action_group_id": None if intent.action_group_id is None else str(intent.action_group_id),
        "metadata": dict(intent.metadata),
    }


def _intent_from_payload(payload: Mapping[str, Any]) -> OrderIntent:
    return OrderIntent(
        account_id=str(payload["account_id"]),
        symbol=str(payload["symbol"]),
        position_side=PositionSide(str(payload["position_side"])),
        action=IntentAction(str(payload["action"])),
        quantity=Decimal(str(payload["quantity"])),
        idempotency_key=str(payload["idempotency_key"]),
        order_type=OrderType(str(payload["order_type"])),
        limit_price=(
            None if payload.get("limit_price") is None else Decimal(str(payload["limit_price"]))
        ),
        reduce_only=bool(payload.get("reduce_only", False)),
        intent_id=UUID(str(payload["intent_id"])),
        action_group_id=(
            None
            if payload.get("action_group_id") is None
            else UUID(str(payload["action_group_id"]))
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def _report_from_payload(payload: Mapping[str, Any]) -> TestnetE2EReport:
    return TestnetE2EReport(
        run_id=str(payload["run_id"]),
        status=str(payload["status"]),
        final_state=TestnetScenarioState(str(payload["final_state"])),
        account_id=str(payload["account_id"]),
        actor=str(payload.get("actor", "legacy-unknown")),
        mode=TestnetScenarioMode(str(payload["mode"])),
        symbol=str(payload["symbol"]),
        position_side=PositionSide(str(payload["position_side"])),
        client_order_id=(
            None if payload.get("client_order_id") is None else str(payload["client_order_id"])
        ),
        validation_accepted=bool(payload["validation_accepted"]),
        submitted_status=(
            None if payload.get("submitted_status") is None else str(payload["submitted_status"])
        ),
        final_order_status=(
            None
            if payload.get("final_order_status") is None
            else str(payload["final_order_status"])
        ),
        unexpected_fill=bool(payload["unexpected_fill"]),
        manual_cleanup_required=bool(payload["manual_cleanup_required"]),
        logical_requests=int(payload["logical_requests"]),
        write_requests=int(payload["write_requests"]),
        read_requests=int(payload["read_requests"]),
        used_weight_1m=(
            None if payload.get("used_weight_1m") is None else int(payload["used_weight_1m"])
        ),
        order_count_1m=(
            None if payload.get("order_count_1m") is None else int(payload["order_count_1m"])
        ),
        started_at=datetime.fromisoformat(str(payload["started_at"])),
        completed_at=datetime.fromisoformat(str(payload["completed_at"])),
        resumed=bool(payload["resumed"]),
        error=None if payload.get("error") is None else str(payload["error"]),
    )


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _notional(value: Decimal, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must use exact Decimal")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > DEFAULT_TESTNET_MAX_NOTIONAL:
        raise ValueError(f"{field_name} must be in (0, 25]")
    return parsed
