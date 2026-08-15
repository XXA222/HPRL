"""Strict RBAC and one-time dangerous-operation confirmation primitives."""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from hashlib import sha256
from threading import RLock


class HedgeRole(IntEnum):
    VIEWER = 10
    OPERATOR = 20
    RISK_MANAGER = 30
    ADMIN = 40


def _scope_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if (
        not result
        or len(result) > 128
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in result)
    ):
        raise ValueError(f"{field_name} is required and must be valid")
    return result


def _optional_scope_text(value: object | None, *, field_name: str) -> str:
    if value is None:
        return ""
    return _scope_text(value, field_name=field_name)


def _confirmation_scope(
    *,
    subject: str,
    action: str,
    account_id: str | None,
    symbol: str | None,
    payload_hash: str | None,
    idempotency_key: str | None,
) -> tuple[str, str, str, str, str, str, str]:
    subject_text = _scope_text(subject, field_name="subject")
    action_text = _scope_text(action, field_name="action")
    account_text = _optional_scope_text(account_id, field_name="account_id")
    symbol_text = _optional_scope_text(symbol, field_name="symbol")
    payload_text = _optional_scope_text(payload_hash, field_name="payload_hash")
    idempotency_text = _optional_scope_text(
        idempotency_key, field_name="idempotency_key"
    )
    scope = "\x1f".join(
        (
            subject_text,
            action_text,
            account_text,
            symbol_text,
            payload_text,
            idempotency_text,
        )
    )
    return (
        subject_text,
        action_text,
        account_text,
        symbol_text,
        payload_text,
        idempotency_text,
        scope,
    )


@dataclass(frozen=True, slots=True)
class HedgePrincipal:
    subject: str
    role: HedgeRole

    def __post_init__(self) -> None:
        subject = _scope_text(self.subject, field_name="principal subject")
        try:
            role = self.role if isinstance(self.role, HedgeRole) else HedgeRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("principal role is invalid") from exc
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "role", role)


class ConfirmationService:
    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: int = 120,
        max_pending: int = 10_000,
    ) -> None:
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < 32:
            raise ValueError("confirmation secret must be at least 32 bytes")
        if (
            not isinstance(ttl_seconds, int)
            or isinstance(ttl_seconds, bool)
            or not 1 <= ttl_seconds <= 86_400
        ):
            raise ValueError("ttl_seconds must be a positive integer up to 86400")
        if (
            not isinstance(max_pending, int)
            or isinstance(max_pending, bool)
            or not 1 <= max_pending <= 100_000
        ):
            raise ValueError("max_pending must be in [1, 100000]")
        self._secret = bytes(secret)
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_pending = max_pending
        self._issued: dict[str, tuple[str, datetime]] = {}
        self._scope_tokens: dict[str, str] = {}
        self._lock = RLock()

    def issue(
        self,
        *,
        subject: str,
        action: str,
        account_id: str | None = None,
        symbol: str | None = None,
        payload_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        (
            subject_text,
            action_text,
            account_text,
            symbol_text,
            payload_text,
            idempotency_text,
            scope,
        ) = _confirmation_scope(
            subject=subject,
            action=action,
            account_id=account_id,
            symbol=symbol,
            payload_hash=payload_hash,
            idempotency_key=idempotency_key,
        )
        nonce = secrets.token_urlsafe(18)
        expires = datetime.now(UTC) + self._ttl
        epoch = str(int(expires.timestamp()))
        material = "\x1f".join(
            (
                subject_text,
                action_text,
                account_text,
                symbol_text,
                payload_text,
                idempotency_text,
                nonce,
                epoch,
            )
        )
        signature = hmac.new(
            self._secret,
            material.encode("utf-8"),
            sha256,
        ).hexdigest()
        token = f"{epoch}.{nonce}.{signature}"
        with self._lock:
            self._purge_locked(datetime.now(UTC))
            previous = self._scope_tokens.get(scope)
            if previous is not None:
                self._issued.pop(previous, None)
            if len(self._issued) >= self._max_pending:
                raise RuntimeError("confirmation token capacity reached")
            self._issued[token] = (scope, expires)
            self._scope_tokens[scope] = token
        return token

    def consume(
        self,
        *,
        token: str,
        subject: str,
        action: str,
        account_id: str | None = None,
        symbol: str | None = None,
        payload_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> bool:
        if (
            not isinstance(token, str)
            or not 1 <= len(token) <= 512
            or any(ord(ch) < 33 or ord(ch) == 127 for ch in token)
        ):
            return False
        try:
            (
                subject_text,
                action_text,
                account_text,
                symbol_text,
                payload_text,
                idempotency_text,
                requested_scope,
            ) = _confirmation_scope(
                subject=subject,
                action=action,
                account_id=account_id,
                symbol=symbol,
                payload_hash=payload_hash,
                idempotency_key=idempotency_key,
            )
            epoch, nonce, signature = token.split(".", 2)
            epoch_value = int(epoch)
        except (ValueError, TypeError):
            return False
        now = datetime.now(UTC)
        with self._lock:
            self._purge_locked(now)
            record = self._issued.get(token)
            if record is None:
                return False
            expected_scope, expires = record
            if epoch_value != int(expires.timestamp()):
                self._remove_locked(token, expected_scope)
                return False
            if expected_scope != requested_scope:
                return False
            material = "\x1f".join(
                (
                    subject_text,
                    action_text,
                    account_text,
                    symbol_text,
                    payload_text,
                    idempotency_text,
                    nonce,
                    epoch,
                )
            )
            expected = hmac.new(
                self._secret,
                material.encode("utf-8"),
                sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return False
            self._remove_locked(token, expected_scope)
            return True

    def purge(self) -> int:
        with self._lock:
            before = len(self._issued)
            self._purge_locked(datetime.now(UTC))
            return before - len(self._issued)

    def _purge_locked(self, now: datetime) -> None:
        expired = [
            (token, scope)
            for token, (scope, deadline) in self._issued.items()
            if now >= deadline
        ]
        for token, scope in expired:
            self._remove_locked(token, scope)

    def _remove_locked(self, token: str, scope: str) -> None:
        self._issued.pop(token, None)
        if self._scope_tokens.get(scope) == token:
            self._scope_tokens.pop(scope, None)
