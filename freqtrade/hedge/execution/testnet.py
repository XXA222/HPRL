"""Guarded Binance USD-M Testnet execution composition.

This module is the only supported entry point for exchange-write validation before
mainnet production is enabled.  Testnet credentials, endpoints, account namespace,
arming token and notional limits are deliberately isolated from LIVE execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from freqtrade.hedge.contracts.ports import (
    AlwaysReadyGate,
    InMemoryPositionLock,
    InMemorySingleWriter,
    MarketRules,
    StaticMarketRules,
)
from freqtrade.hedge.exchange.base import ReadonlyState
from freqtrade.hedge.integration.repository import InMemoryReadonlyRepository
from freqtrade.hedge.exchange.binance_readonly import PermissionPolicy
from freqtrade.hedge.readonly.runtime import (
    BinanceReadonlyRuntime,
    BinanceReadonlyRuntimeConfig,
    build_binance_readonly_runtime,
)

from .action_group_store import ActionGroupRepository, InMemoryActionGroupRepository
from .binance_environment import (
    ExecutionEnvironment,
    TESTNET_PROFILE,
    execution_account_id,
)
from .binance_usdm_adapter import (
    BinanceExecutionCredentials,
    BinanceTestOrderValidation,
    HttpTransport,
)
from .event_publisher import InMemoryEventPublisher
from .idempotency import IdempotencyPort, InMemoryIdempotencyStore
from .ledger import InMemoryExecutionLedger
from .production_gate import ProductionExecutionGate, ProductionGateEvidence
from .production_runtime import ProductionExecutionRuntime, build_production_execution_runtime
from .service import (
    AllowAllRiskApproval,
    ApprovedOrderIntent,
    ExecutionResult,
    ExecutionStorePort,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    RiskApprovalPort,
)

TESTNET_CREDENTIAL_MARKER = "BINANCE_USDM_TESTNET"
TESTNET_ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
DEFAULT_TESTNET_MAX_NOTIONAL = Decimal("25")


@dataclass(frozen=True, slots=True, repr=False)
class BinanceTestnetCredentials:
    """Dedicated testnet credential material with an explicit file marker."""

    execution: BinanceExecutionCredentials
    account_fingerprint: str
    source_path: Path | None = None

    @property
    def account_id(self) -> str:
        return execution_account_id(
            ExecutionEnvironment.TESTNET,
            self.account_fingerprint,
        )

    def __repr__(self) -> str:
        source = "<memory>" if self.source_path is None else str(self.source_path)
        return (
            "BinanceTestnetCredentials(execution=<redacted>, "
            f"account_fingerprint={self.account_fingerprint!r}, source_path={source!r})"
        )


@dataclass(frozen=True, slots=True)
class TestnetReadonlyEvidence:
    account_id: str
    account_fingerprint: str
    allowed_symbols: tuple[str, ...]
    cross_margin_symbols: tuple[str, ...]
    clock_offset_ms: int
    futures_trading_permission: bool
    readonly_status: str
    user_stream_status: str


@dataclass(frozen=True, slots=True)
class GuardedTestnetRuntime:
    runtime: ProductionExecutionRuntime
    readonly: BinanceReadonlyRuntime | None
    evidence: ProductionGateEvidence
    store: ExecutionStorePort
    idempotency: IdempotencyPort[ExecutionResult]
    risk: RiskApprovalPort
    ledger: InMemoryExecutionLedger

    def arm(
        self,
        *,
        token: str,
        actor: str,
        confirmed: bool = True,
        ttl_seconds: int = 300,
    ) -> None:
        self.runtime.gate.arm(
            token=token,
            actor=actor,
            confirmed=confirmed,
            ttl_seconds=ttl_seconds,
        )

    def validate_order(self, intent: OrderIntent) -> BinanceTestOrderValidation:
        approved = approve_testnet_intent(intent, risk=self.risk)
        return self.runtime.exchange.validate_order(approved)

    def submit(self, intent: OrderIntent) -> ExecutionResult:
        return self.runtime.engine.submit(intent)

    def cancel(self, client_order_id: str) -> ExecutionResult:
        return self.runtime.engine.cancel(client_order_id)


@dataclass(frozen=True, slots=True)
class TestnetSubmitCancelReport:
    client_order_id: str
    submitted_status: str
    cancel_status: str
    unexpected_fill: bool
    exchange_order_id: str | None
    started_at: datetime
    completed_at: datetime
    write_requests: int

    @property
    def clean(self) -> bool:
        return not self.unexpected_fill and self.cancel_status == "CANCELED"


def load_binance_testnet_credentials(path: str | Path) -> BinanceTestnetCredentials:
    """Load a three-line credential file that cannot be mistaken for mainnet.

    Line 1: API key
    Line 2: API secret
    Line 3: exact marker ``BINANCE_USDM_TESTNET``
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    if b"\x00" in payload or len(payload) > 64 * 1024:
        raise ValueError("testnet credential file is invalid")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("testnet credential file must be UTF-8") from exc
    lines = [line.strip() for line in text.splitlines()]
    if len(lines) < 3:
        raise ValueError("testnet credential file must contain key, secret and marker")
    api_key, api_secret, marker = lines[:3]
    if marker != TESTNET_CREDENTIAL_MARKER:
        raise ValueError("testnet credential marker is missing or invalid")
    if any(line for line in lines[3:]):
        raise ValueError("testnet credential file contains unexpected extra data")
    credentials = BinanceExecutionCredentials(api_key=api_key, api_secret=api_secret)
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:24]
    return BinanceTestnetCredentials(
        execution=credentials,
        account_fingerprint=fingerprint,
        source_path=source,
    )


def build_testnet_readonly_config(
    *,
    credentials: BinanceTestnetCredentials,
    managed_symbols: Iterable[str] = TESTNET_ALLOWED_SYMBOLS,
    proxy_url: str | None = None,
) -> BinanceReadonlyRuntimeConfig:
    symbols = _testnet_symbols(managed_symbols)
    return BinanceReadonlyRuntimeConfig(
        account_id=credentials.account_id,
        managed_symbols=symbols,
        api_key=credentials.execution.api_key,
        api_secret=credentials.execution.api_secret,
        futures_base_url=TESTNET_PROFILE.rest_base_url,
        spot_base_url=TESTNET_PROFILE.rest_base_url,
        websocket_base_url=TESTNET_PROFILE.websocket_base_url,
        rest_proxy_url=proxy_url,
        websocket_proxy_url=proxy_url,
        permission_policy=PermissionPolicy(
            require_restriction_endpoint=False,
            require_read_enabled=True,
            require_futures_enabled=True,
            allow_testnet_restriction_endpoint_unavailable=True,
        ),
        system_client_order_prefixes=("FTH-", "FHTN-"),
    )


def build_testnet_readonly_runtime(
    *,
    credentials: BinanceTestnetCredentials,
    managed_symbols: Iterable[str] = TESTNET_ALLOWED_SYMBOLS,
    proxy_url: str | None = None,
    repository: InMemoryReadonlyRepository | None = None,
) -> BinanceReadonlyRuntime:
    return build_binance_readonly_runtime(
        config=build_testnet_readonly_config(
            credentials=credentials,
            managed_symbols=managed_symbols,
            proxy_url=proxy_url,
        ),
        repository=repository or InMemoryReadonlyRepository(),
    )


def evidence_from_ready_testnet_readonly(
    *,
    readonly: BinanceReadonlyRuntime,
    credentials: BinanceTestnetCredentials,
    expected_arm_token_sha256: str,
    max_order_notional: Decimal = DEFAULT_TESTNET_MAX_NOTIONAL,
    futures_trading_permission: bool = True,
    allow_market_orders: bool = False,
) -> ProductionGateEvidence:
    if not futures_trading_permission:
        raise RuntimeError("TESTNET_FUTURES_TRADING_PERMISSION_REQUIRED")
    snapshot = readonly.snapshot()
    health = snapshot.direction2_health
    if snapshot.status.state is not ReadonlyState.READY:
        raise RuntimeError("TESTNET_READONLY_NOT_READY")
    if health.account_id != credentials.account_id:
        raise RuntimeError("TESTNET_ACCOUNT_ID_MISMATCH")
    blockers: list[str] = []
    if not health.rest_fresh:
        blockers.append("REST_NOT_FRESH")
    if not health.stream_connected:
        blockers.append("STREAM_NOT_CONNECTED")
    if not health.stream_fresh:
        blockers.append("STREAM_NOT_FRESH")
    if not health.clock_synchronized:
        blockers.append("CLOCK_NOT_SYNCHRONIZED")
    if not health.configuration_valid:
        blockers.append("ACCOUNT_CONFIGURATION_INVALID")
    if not health.reconciliation_consistent:
        blockers.append("RECONCILIATION_NOT_CONSISTENT")
    if health.unmanaged_position_count:
        blockers.append("UNMANAGED_POSITION")
    if health.unmanaged_order_count:
        blockers.append("UNMANAGED_ORDER")
    if blockers:
        raise RuntimeError(",".join(blockers))
    clock_status = snapshot.clock_sync_status
    offset = int(round(float(getattr(clock_status, "offset_ms", 0.0))))
    symbols = _testnet_symbols(readonly.config.managed_symbols)
    return ProductionGateEvidence(
        environment=ExecutionEnvironment.TESTNET,
        account_fingerprint=credentials.account_fingerprint,
        account_id_prefix=TESTNET_PROFILE.account_prefix,
        allowed_symbols=symbols,
        cross_margin_symbols=symbols,
        readonly_status="FULL_PASS",
        user_stream_status=(
            "FULL_PASS" if snapshot.stream_health.last_event_at is not None
            else "PASS_WITH_NO_USER_EVENTS"
        ),
        hedge_mode_enabled=True,
        clock_offset_ms=offset,
        testnet_trading_enabled=True,
        futures_trading_permission=bool(futures_trading_permission),
        expected_arm_token_sha256=expected_arm_token_sha256,
        max_order_notional=_notional(max_order_notional),
        allow_market_orders=allow_market_orders,
    )


def build_guarded_testnet_runtime(
    *,
    credentials: BinanceTestnetCredentials,
    evidence: ProductionGateEvidence,
    readonly: BinanceReadonlyRuntime | None = None,
    proxy_url: str | None = None,
    transport: HttpTransport | None = None,
    risk: RiskApprovalPort | None = None,
    market_rules: MarketRules | None = None,
    store: ExecutionStorePort | None = None,
    idempotency: IdempotencyPort[ExecutionResult] | None = None,
    readiness: object | None = None,
    single_writer: object | None = None,
    position_lock: object | None = None,
    transaction: object | None = None,
    publisher: object | None = None,
    action_group_repository: ActionGroupRepository | None = None,
) -> GuardedTestnetRuntime:
    if evidence.environment is not ExecutionEnvironment.TESTNET:
        raise ValueError("testnet runtime requires TESTNET gate evidence")
    if evidence.account_id_prefix != TESTNET_PROFILE.account_prefix:
        raise ValueError("testnet runtime requires isolated testnet account namespace")
    if not evidence.futures_trading_permission:
        raise ValueError("testnet runtime requires futures trading permission evidence")
    if evidence.account_fingerprint != credentials.account_fingerprint:
        raise ValueError("credential fingerprint does not match gate evidence")
    if readonly is not None and readonly.config.account_id != credentials.account_id:
        raise ValueError("readonly runtime account does not match testnet credentials")
    selected_store = store or InMemoryExecutionStore()
    selected_idempotency = idempotency or InMemoryIdempotencyStore()
    selected_risk = risk or AllowAllRiskApproval()
    ledger = transaction or InMemoryExecutionLedger()
    runtime = build_production_execution_runtime(
        credentials=credentials.execution,
        gate=ProductionExecutionGate(evidence),
        risk=selected_risk,
        store=selected_store,
        idempotency=selected_idempotency,
        readiness=readiness or AlwaysReadyGate(),
        single_writer=single_writer or InMemorySingleWriter(),
        position_lock=position_lock or InMemoryPositionLock(),
        market_rules=StaticMarketRules(market_rules or MarketRules()),
        transaction=ledger,
        publisher=publisher or InMemoryEventPublisher(),
        action_group_repository=action_group_repository or InMemoryActionGroupRepository(),
        proxy_url=proxy_url,
        base_url=TESTNET_PROFILE.rest_base_url,
        transport=transport,
        user_stream=None if readonly is None else readonly.stream,
    )
    return GuardedTestnetRuntime(
        runtime=runtime,
        readonly=readonly,
        evidence=evidence,
        store=selected_store,
        idempotency=selected_idempotency,
        risk=selected_risk,
        ledger=ledger,
    )


def approve_testnet_intent(
    intent: OrderIntent,
    *,
    risk: RiskApprovalPort,
) -> ApprovedOrderIntent:
    if intent.account_id.split(":", 1)[0] != TESTNET_PROFILE.account_prefix:
        raise ValueError("testnet intent must use isolated testnet account namespace")
    approval = risk.approve(intent)
    if not approval.approved or approval.approved_quantity <= 0:
        raise PermissionError("testnet intent was rejected by risk approval")
    from .client_order_id import build_client_order_id

    client_order_id = build_client_order_id(
        account_id=intent.account_id,
        symbol=intent.symbol,
        position_side=intent.position_side.value,
        idempotency_key=intent.idempotency_key,
        prefix="FHTN",
    )
    return ApprovedOrderIntent(
        intent=intent,
        approved_quantity=approval.approved_quantity,
        client_order_id=client_order_id,
        approved_at=datetime.now(UTC),
        risk_reason_codes=approval.reason_codes,
    )


def run_submit_cancel_canary(
    *,
    guarded: GuardedTestnetRuntime,
    intent: OrderIntent,
) -> TestnetSubmitCancelReport:
    if intent.order_type is not OrderType.LIMIT:
        raise ValueError("testnet submit/cancel canary requires LIMIT order")
    if intent.action not in {IntentAction.OPEN, IntentAction.INCREASE}:
        raise ValueError("testnet submit/cancel canary must create a risk-increasing passive order")
    started = datetime.now(UTC)
    before = guarded.runtime.exchange.telemetry().write_requests
    submitted = guarded.submit(intent)
    submitted_order = submitted.order
    unexpected_fill = submitted_order.lifecycle.filled_quantity > 0
    canceled = submitted
    if not submitted_order.lifecycle.terminal:
        canceled = guarded.cancel(submitted_order.client_order_id)
    after = guarded.runtime.exchange.telemetry().write_requests
    return TestnetSubmitCancelReport(
        client_order_id=submitted_order.client_order_id,
        submitted_status=submitted_order.lifecycle.status.value,
        cancel_status=canceled.order.lifecycle.status.value,
        unexpected_fill=unexpected_fill or canceled.order.lifecycle.filled_quantity > 0,
        exchange_order_id=submitted_order.lifecycle.exchange_order_id,
        started_at=started,
        completed_at=datetime.now(UTC),
        write_requests=after - before,
    )


def make_testnet_limit_intent(
    *,
    credentials: BinanceTestnetCredentials,
    symbol: str,
    position_side: PositionSide,
    quantity: Decimal,
    limit_price: Decimal,
    idempotency_key: str,
    time_in_force: str = "GTC",
) -> OrderIntent:
    normalized = _testnet_symbols((symbol,))[0]
    tif = str(time_in_force).strip().upper()
    if tif not in {"GTC", "GTX"}:
        raise ValueError("testnet canary time_in_force must be GTC or GTX")
    return OrderIntent(
        account_id=credentials.account_id,
        symbol=normalized,
        position_side=position_side,
        action=IntentAction.OPEN,
        quantity=quantity,
        limit_price=limit_price,
        order_type=OrderType.LIMIT,
        reduce_only=False,
        idempotency_key=idempotency_key,
        metadata={"time_in_force": tif, "execution_environment": "TESTNET"},
    )


def _testnet_symbols(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        symbol = str(value).strip().upper().replace("/", "").split(":", 1)[0]
        if symbol not in TESTNET_ALLOWED_SYMBOLS:
            raise ValueError("testnet execution only supports BTCUSDT and ETHUSDT perpetual")
        if symbol not in result:
            result.append(symbol)
    if not result:
        raise ValueError("at least one testnet symbol is required")
    return tuple(result)


def _notional(value: Decimal) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError("max_order_notional must use exact Decimal")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("max_order_notional is invalid") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > DEFAULT_TESTNET_MAX_NOTIONAL:
        raise ValueError("testnet max_order_notional must be in (0, 25]")
    return parsed
