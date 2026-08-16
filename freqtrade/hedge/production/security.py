"""Credential and deployment security policy for production promotion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SecurityFacts:
    key_present: bool
    secret_present: bool
    futures_permission: bool
    withdrawal_permission: bool
    ip_restricted: bool
    hedge_mode: bool
    all_managed_symbols_cross: bool
    tls_verify: bool
    secrets_in_source_scan_passed: bool
    dependency_scan_passed: bool
    image_digest_pinned: bool


@dataclass(frozen=True, slots=True)
class SecurityDecision:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_security(value: SecurityFacts, *, live: bool) -> SecurityDecision:
    reasons: list[str] = []
    if not value.key_present or not value.secret_present: reasons.append("CREDENTIALS_MISSING")
    if not value.futures_permission: reasons.append("FUTURES_PERMISSION_MISSING")
    if value.withdrawal_permission: reasons.append("WITHDRAWAL_PERMISSION_MUST_BE_DISABLED")
    if live and not value.ip_restricted: reasons.append("LIVE_REQUIRES_IP_RESTRICTION")
    if not value.hedge_mode: reasons.append("HEDGE_MODE_REQUIRED")
    if not value.all_managed_symbols_cross: reasons.append("CROSS_MARGIN_REQUIRED")
    if not value.tls_verify: reasons.append("TLS_VERIFY_REQUIRED")
    if not value.secrets_in_source_scan_passed: reasons.append("SECRET_SCAN_FAILED")
    if not value.dependency_scan_passed: reasons.append("DEPENDENCY_SCAN_FAILED")
    if live and not value.image_digest_pinned: reasons.append("LIVE_REQUIRES_PINNED_IMAGE")
    return SecurityDecision(not reasons, tuple(reasons))
