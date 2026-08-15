"""Adapter for Freqtrade protections and PairLocks in a dual-leg Hedge runtime."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import (
    AdmissionCode,
    AdmissionDecision,
    HedgeSide,
    NativeOrderIntent,
    ProtectionSnapshot,
    utc_datetime,
)


class HedgeProtectionAdapter:
    """Evaluate global and per-pair protections independently for LONG and SHORT.

    Protection plugins may create PairLocks as a side effect.  Failures are fail-closed
    for risk-increasing intents, while reduce-only actions always retain an escape path.
    """

    def __init__(self, manager: Any, pairlocks: Any, *, fail_closed: bool = True) -> None:
        self.manager = manager
        self.pairlocks = pairlocks
        self.fail_closed = bool(fail_closed)

    def refresh(
        self,
        pair: str,
        side: HedgeSide | str,
        *,
        now: datetime | None = None,
        starting_balance: float = 0.0,
    ) -> ProtectionSnapshot:
        normalized_side = HedgeSide.parse(side)
        observed_at = utc_datetime(now)
        reasons: list[str] = []
        pairlock_side = normalized_side.pairlock_side
        try:
            global_lock = self.manager.global_stop(
                now=observed_at,
                side=pairlock_side,
                starting_balance=float(starting_balance),
            )
            local_lock = self.manager.stop_per_pair(
                pair,
                now=observed_at,
                side=pairlock_side,
                starting_balance=float(starting_balance),
            )
            if global_lock is not None:
                reasons.append(str(getattr(global_lock, "reason", "GLOBAL_PROTECTION")))
            if local_lock is not None:
                reasons.append(str(getattr(local_lock, "reason", "PAIR_PROTECTION")))
            global_locked = bool(self.pairlocks.is_global_lock(observed_at, side=pairlock_side))
            pair_locked = bool(
                self.pairlocks.is_pair_locked(pair, observed_at, side=pairlock_side)
            )
            lock = self.pairlocks.get_pair_longest_lock(pair, observed_at, pairlock_side)
            if lock is not None and getattr(lock, "reason", None):
                reasons.append(str(lock.reason))
            global_lock_record = self.pairlocks.get_pair_longest_lock(
                "*", observed_at, pairlock_side
            )
            if global_lock_record is not None and getattr(global_lock_record, "reason", None):
                reasons.append(str(global_lock_record.reason))
            return ProtectionSnapshot(
                pair=str(pair).strip().upper(),
                side=normalized_side,
                global_locked=global_locked,
                pair_locked=pair_locked,
                reasons=tuple(dict.fromkeys(item for item in reasons if item)),
                observed_at=observed_at,
            )
        except Exception as exc:
            if not self.fail_closed:
                return ProtectionSnapshot(
                    str(pair).strip().upper(),
                    normalized_side,
                    False,
                    False,
                    (f"PROTECTION_ERROR_IGNORED:{type(exc).__name__}",),
                    observed_at,
                )
            raise RuntimeError("Hedge protection evaluation failed closed") from exc

    def admit(
        self,
        intent: NativeOrderIntent,
        *,
        now: datetime | None = None,
        starting_balance: float = 0.0,
    ) -> AdmissionDecision:
        if intent.reduce_only:
            return AdmissionDecision.allow(
                reason="REDUCE_ONLY_PROTECTION_EXEMPT",
                reduce_only_exempt=True,
            )
        try:
            snapshot = self.refresh(
                intent.pair,
                intent.side,
                now=now,
                starting_balance=starting_balance,
            )
        except RuntimeError as exc:
            return AdmissionDecision.block(
                AdmissionCode.PROTECTION_ERROR,
                str(exc),
            )
        if snapshot.global_locked:
            return AdmissionDecision.block(
                AdmissionCode.GLOBAL_PAIRLOCK,
                ";".join(snapshot.reasons) or "global protection lock is active",
            )
        if snapshot.pair_locked:
            return AdmissionDecision.block(
                AdmissionCode.PAIR_LOCKED,
                ";".join(snapshot.reasons) or "pair protection lock is active",
            )
        return AdmissionDecision.allow(reason="PROTECTIONS_CLEAR")
