"""Fail-closed production execution gate for Binance USD-M hedge trading."""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from threading import RLock
from typing import Iterable

from .binance_environment import ExecutionEnvironment
from .service import ApprovedOrderIntent, OrderType


class ExecutionWriteLockedError(PermissionError):
    """Raised before any exchange write when production evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class ProductionGateEvidence:
    environment: ExecutionEnvironment
    account_fingerprint: str
    allowed_symbols: tuple[str, ...]
    cross_margin_symbols: tuple[str, ...]
    readonly_status: str
    user_stream_status: str
    hedge_mode_enabled: bool
    clock_offset_ms: int
    live_trading_enabled: bool = False
    testnet_trading_enabled: bool = False
    strict_key_policy_passed: bool = False
    futures_trading_permission: bool = False
    ip_restricted: bool = False
    expected_arm_token_sha256: str | None = None
    max_abs_clock_offset_ms: int = 1000
    max_order_notional: Decimal = Decimal("100")
    allow_market_orders: bool = True
    account_id_prefix: str = "binance-usdm"

    def __post_init__(self) -> None:
        environment = (
            self.environment
            if isinstance(self.environment, ExecutionEnvironment)
            else ExecutionEnvironment(self.environment)
        )
        fingerprint = str(self.account_fingerprint).strip()
        if not fingerprint or len(fingerprint) > 128:
            raise ValueError("account_fingerprint is required")
        allowed = _symbols(self.allowed_symbols, "allowed_symbols")
        cross = _symbols(self.cross_margin_symbols, "cross_margin_symbols")
        if not allowed:
            raise ValueError("allowed_symbols must not be empty")
        if not set(allowed).issubset(set(cross)):
            raise ValueError("every allowed symbol must be proven Cross margin")
        if not isinstance(self.clock_offset_ms, int) or isinstance(self.clock_offset_ms, bool):
            raise TypeError("clock_offset_ms must be an integer")
        if (
            not isinstance(self.max_abs_clock_offset_ms, int)
            or isinstance(self.max_abs_clock_offset_ms, bool)
            or self.max_abs_clock_offset_ms <= 0
        ):
            raise ValueError("max_abs_clock_offset_ms must be positive")
        maximum = _decimal(self.max_order_notional, "max_order_notional")
        if maximum <= 0:
            raise ValueError("max_order_notional must be positive")
        prefix = str(self.account_id_prefix).strip()
        if prefix not in {"binance-usdm", "binance-usdm-testnet"}:
            raise ValueError("account_id_prefix is invalid")
        digest = self.expected_arm_token_sha256
        if digest is not None:
            digest = str(digest).strip().lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("expected_arm_token_sha256 must be a SHA256 hex digest")
        for field_name in (
            "hedge_mode_enabled",
            "live_trading_enabled",
            "testnet_trading_enabled",
            "strict_key_policy_passed",
            "futures_trading_permission",
            "ip_restricted",
            "allow_market_orders",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "account_fingerprint", fingerprint)
        object.__setattr__(self, "allowed_symbols", allowed)
        object.__setattr__(self, "cross_margin_symbols", cross)
        object.__setattr__(self, "expected_arm_token_sha256", digest)
        object.__setattr__(self, "max_order_notional", maximum)
        object.__setattr__(self, "account_id_prefix", prefix)


@dataclass(frozen=True, slots=True)
class ExecutionPermit:
    environment: ExecutionEnvironment
    account_fingerprint: str
    symbol: str
    client_order_id: str
    actor: str
    issued_at: datetime
    expires_at_monotonic: float
    notional: Decimal


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    environment: ExecutionEnvironment
    armed: bool
    actor: str | None
    armed_until_monotonic: float | None
    blocking_reasons: tuple[str, ...]


class ProductionExecutionGate:
    """Two-step gate which cannot be bypassed by order metadata.

    Arming is process-local and time-limited. Restarting the process disarms the gate.
    """

    _READONLY_PASS = frozenset({"FULL_PASS", "PASS_WITH_DUAL_LEG_OBSERVATION_GAP"})
    _STREAM_PASS = frozenset({"FULL_PASS", "PASS_WITH_NO_USER_EVENTS"})

    def __init__(self, evidence: ProductionGateEvidence) -> None:
        if not isinstance(evidence, ProductionGateEvidence):
            raise TypeError("evidence must be ProductionGateEvidence")
        self.evidence = evidence
        self._armed_until: float | None = None
        self._actor: str | None = None
        self._lock = RLock()

    def arm(
        self,
        *,
        token: str,
        actor: str,
        confirmed: bool,
        ttl_seconds: int = 300,
    ) -> GateSnapshot:
        if not confirmed:
            raise PermissionError("execution arming requires secondary confirmation")
        expected = self.evidence.expected_arm_token_sha256
        if expected is None:
            raise ExecutionWriteLockedError("ARM_TOKEN_NOT_CONFIGURED")
        token_text = str(token)
        actor_text = str(actor).strip()
        if not actor_text or len(actor_text) > 128:
            raise ValueError("actor is required")
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool):
            raise TypeError("ttl_seconds must be an integer")
        if ttl_seconds < 10 or ttl_seconds > 3600:
            raise ValueError("ttl_seconds must be in [10, 3600]")
        actual = hashlib.sha256(token_text.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise PermissionError("execution arming token is invalid")
        reasons = self._evidence_reasons()
        if reasons:
            raise ExecutionWriteLockedError(",".join(reasons))
        with self._lock:
            self._actor = actor_text
            self._armed_until = time.monotonic() + ttl_seconds
        return self.snapshot()

    def disarm(self) -> GateSnapshot:
        with self._lock:
            self._actor = None
            self._armed_until = None
        return self.snapshot()

    def snapshot(self) -> GateSnapshot:
        with self._lock:
            armed = self._is_armed_locked()
            return GateSnapshot(
                environment=self.evidence.environment,
                armed=armed,
                actor=self._actor if armed else None,
                armed_until_monotonic=self._armed_until if armed else None,
                blocking_reasons=self._evidence_reasons(),
            )

    def assert_order_allowed(self, approved: ApprovedOrderIntent) -> ExecutionPermit:
        if not isinstance(approved, ApprovedOrderIntent):
            raise TypeError("approved must be ApprovedOrderIntent")
        reasons = list(self._evidence_reasons())
        symbol = approved.intent.symbol
        if symbol not in self.evidence.allowed_symbols:
            reasons.append("SYMBOL_NOT_ALLOWLISTED")
        expected_account = (
            f"{self.evidence.account_id_prefix}:{self.evidence.account_fingerprint}"
        )
        if approved.intent.account_id != expected_account:
            reasons.append("ACCOUNT_FINGERPRINT_MISMATCH")
        if approved.intent.order_type is OrderType.MARKET and not self.evidence.allow_market_orders:
            reasons.append("MARKET_ORDERS_DISABLED")
        notional = _order_notional(approved)
        if notional > self.evidence.max_order_notional:
            reasons.append("ORDER_NOTIONAL_LIMIT_EXCEEDED")
        with self._lock:
            if not self._is_armed_locked():
                reasons.append("EXECUTION_NOT_ARMED")
            actor = self._actor
            expires = self._armed_until
        if reasons:
            raise ExecutionWriteLockedError(",".join(dict.fromkeys(reasons)))
        if actor is None or expires is None:  # pragma: no cover - protected by lock check
            raise ExecutionWriteLockedError("EXECUTION_NOT_ARMED")
        return ExecutionPermit(
            environment=self.evidence.environment,
            account_fingerprint=self.evidence.account_fingerprint,
            symbol=symbol,
            client_order_id=approved.client_order_id,
            actor=actor,
            issued_at=datetime.now(UTC),
            expires_at_monotonic=expires,
            notional=notional,
        )

    def assert_cancel_allowed(self, order: object) -> None:
        """Allow cancellation without process arming when an existing order is in scope.

        Cancellation reduces outstanding exchange risk and must remain available after an
        operator disarms new submissions or after the arming TTL expires. The method still
        requires the configured environment, account, symbol, hedge mode, clock, and the
        Binance futures permission needed to perform the cancellation.
        """

        intent = getattr(order, "intent", None)
        if intent is None:
            raise TypeError("order must expose intent")
        reasons = list(self._cancellation_reasons())
        symbol = str(getattr(intent, "symbol", ""))
        if symbol not in self.evidence.allowed_symbols:
            reasons.append("SYMBOL_NOT_ALLOWLISTED")
        expected_account = (
            f"{self.evidence.account_id_prefix}:{self.evidence.account_fingerprint}"
        )
        if getattr(intent, "account_id", None) != expected_account:
            reasons.append("ACCOUNT_FINGERPRINT_MISMATCH")
        if reasons:
            raise ExecutionWriteLockedError(",".join(dict.fromkeys(reasons)))

    def _is_armed_locked(self) -> bool:
        if self._armed_until is None or self._actor is None:
            return False
        if time.monotonic() >= self._armed_until:
            self._armed_until = None
            self._actor = None
            return False
        return True

    def _cancellation_reasons(self) -> tuple[str, ...]:
        value = self.evidence
        reasons: list[str] = []
        if value.environment is ExecutionEnvironment.DISABLED:
            reasons.append("EXECUTION_ENVIRONMENT_DISABLED")
        if value.readonly_status not in self._READONLY_PASS:
            reasons.append("READONLY_PREREQUISITE_FAILED")
        if not value.hedge_mode_enabled:
            reasons.append("HEDGE_MODE_REQUIRED")
        if abs(value.clock_offset_ms) > value.max_abs_clock_offset_ms:
            reasons.append("CLOCK_OFFSET_OUT_OF_RANGE")
        if value.environment is ExecutionEnvironment.LIVE:
            if not value.live_trading_enabled:
                reasons.append("LIVE_TRADING_DISABLED")
            if not value.futures_trading_permission:
                reasons.append("FUTURES_TRADING_PERMISSION_REQUIRED")
        elif value.environment is ExecutionEnvironment.TESTNET:
            if not value.testnet_trading_enabled:
                reasons.append("TESTNET_TRADING_DISABLED")
        return tuple(dict.fromkeys(reasons))

    def _evidence_reasons(self) -> tuple[str, ...]:
        value = self.evidence
        reasons: list[str] = []
        if value.environment is ExecutionEnvironment.DISABLED:
            reasons.append("EXECUTION_ENVIRONMENT_DISABLED")
        if value.readonly_status not in self._READONLY_PASS:
            reasons.append("READONLY_PREREQUISITE_FAILED")
        if value.user_stream_status not in self._STREAM_PASS:
            reasons.append("USER_STREAM_PREREQUISITE_FAILED")
        if not value.hedge_mode_enabled:
            reasons.append("HEDGE_MODE_REQUIRED")
        if abs(value.clock_offset_ms) > value.max_abs_clock_offset_ms:
            reasons.append("CLOCK_OFFSET_OUT_OF_RANGE")
        if value.environment is ExecutionEnvironment.LIVE:
            if not value.live_trading_enabled:
                reasons.append("LIVE_TRADING_DISABLED")
            if not value.strict_key_policy_passed:
                reasons.append("STRICT_KEY_POLICY_REQUIRED")
            if not value.futures_trading_permission:
                reasons.append("FUTURES_TRADING_PERMISSION_REQUIRED")
            if not value.ip_restricted:
                reasons.append("IP_RESTRICTION_REQUIRED")
        elif value.environment is ExecutionEnvironment.TESTNET:
            if not value.testnet_trading_enabled:
                reasons.append("TESTNET_TRADING_DISABLED")
        return tuple(dict.fromkeys(reasons))


def _symbols(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values:
        symbol = str(raw).strip().upper().replace("/", "").split(":", 1)[0]
        if not symbol or not symbol.isascii() or not symbol.isalnum():
            raise ValueError(f"{field_name} contains an invalid symbol")
        if symbol not in result:
            result.append(symbol)
    return tuple(result)


def _decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must use an exact decimal")
    try:
        result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _order_notional(approved: ApprovedOrderIntent) -> Decimal:
    price = approved.intent.limit_price
    if price is None:
        raw = approved.intent.metadata.get("reference_price")
        if raw is None:
            raise ExecutionWriteLockedError("REFERENCE_PRICE_REQUIRED_FOR_MARKET_ORDER")
        price = _decimal(raw, "reference_price")
    if price <= 0:
        raise ExecutionWriteLockedError("REFERENCE_PRICE_INVALID")
    notional = approved.approved_quantity * price
    if notional <= 0 or not notional.is_finite():
        raise ExecutionWriteLockedError("ORDER_NOTIONAL_INVALID")
    return notional
