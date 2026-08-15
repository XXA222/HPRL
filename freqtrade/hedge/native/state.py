"""Bridge Freqtrade worker states into explicit Hedge execution semantics."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import (
    AdmissionCode,
    AdmissionDecision,
    BotStateSnapshot,
    NativeBotMode,
    NativeOrderIntent,
    utc_datetime,
)


class HedgeBotStateAdapter:
    """Translate upstream state values without importing Freqtrade enums.

    RUNNING permits all planner actions.  PAUSED keeps the process alive but permits
    only reduce-only actions.  STOPPED disables planning while still allowing recovery
    and managed-order cancellation.  RELOAD_CONFIG behaves like STOPPED and requests
    managed-order cancellation before the composition root is rebuilt.
    """

    def __init__(self, *, cancel_open_orders_on_exit: bool = False) -> None:
        self.cancel_open_orders_on_exit = bool(cancel_open_orders_on_exit)

    @staticmethod
    def _state_name(state: Any) -> str:
        raw = getattr(state, "value", state)
        if raw is None:
            return "UNKNOWN"
        name = str(raw).strip().upper()
        if name.startswith("STATE."):
            name = name.split(".", 1)[1]
        return name

    def snapshot(self, state: Any, *, at: datetime | None = None) -> BotStateSnapshot:
        name = self._state_name(state)
        observed_at = utc_datetime(at)
        if name == "RUNNING":
            return BotStateSnapshot(
                name,
                NativeBotMode.RUNNING,
                True,
                True,
                True,
                True,
                False,
                observed_at,
            )
        if name == "PAUSED":
            return BotStateSnapshot(
                name,
                NativeBotMode.REDUCE_ONLY,
                True,
                False,
                True,
                True,
                False,
                observed_at,
            )
        if name == "RELOAD_CONFIG":
            return BotStateSnapshot(
                name,
                NativeBotMode.RELOAD,
                False,
                False,
                False,
                True,
                self.cancel_open_orders_on_exit,
                observed_at,
            )
        if name == "STOPPED":
            return BotStateSnapshot(
                name,
                NativeBotMode.STOPPED,
                False,
                False,
                False,
                True,
                self.cancel_open_orders_on_exit,
                observed_at,
            )
        return BotStateSnapshot(
            name,
            NativeBotMode.UNKNOWN,
            False,
            False,
            True,
            True,
            False,
            observed_at,
        )

    def admit(self, state: Any, intent: NativeOrderIntent) -> AdmissionDecision:
        snapshot = self.snapshot(state)
        if intent.reduce_only and snapshot.allow_reduce_only:
            return AdmissionDecision.allow(reason=f"BOT_{snapshot.mode.value}_REDUCE_ONLY")
        if snapshot.allow_new_risk:
            return AdmissionDecision.allow(reason="BOT_RUNNING")
        if snapshot.mode is NativeBotMode.REDUCE_ONLY:
            return AdmissionDecision.block(
                AdmissionCode.BOT_PAUSED,
                "Freqtrade bot is PAUSED; only reduce-only Hedge actions are allowed",
            )
        if snapshot.mode is NativeBotMode.RELOAD:
            return AdmissionDecision.block(
                AdmissionCode.BOT_RELOAD,
                "Freqtrade bot is reloading configuration",
            )
        return AdmissionDecision.block(
            AdmissionCode.BOT_STOPPED,
            f"Freqtrade bot state {snapshot.source_state} does not allow new Hedge risk",
        )
