"""Composable fail-closed order admission and planner-result filtering."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Callable, Iterable

from .models import (
    AdmissionCode,
    AdmissionDecision,
    HedgeAction,
    HedgeBucket,
    HedgeSide,
    NativeOrderIntent,
)


AdmissionProvider = Callable[[NativeOrderIntent], AdmissionDecision | bool]


class CompositeAdmissionPolicy:
    def __init__(self, providers: Iterable[AdmissionProvider] = ()) -> None:
        self._providers = list(providers)

    def add(self, provider: AdmissionProvider) -> None:
        if not callable(provider):
            raise TypeError("admission provider must be callable")
        self._providers.append(provider)

    @property
    def provider_count(self) -> int:
        return len(self._providers)

    def evaluate(self, intent: NativeOrderIntent) -> AdmissionDecision:
        for provider in tuple(self._providers):
            try:
                result = provider(intent)
            except Exception as exc:
                return AdmissionDecision.block(
                    AdmissionCode.READINESS_BLOCKED,
                    f"admission provider failed closed: {type(exc).__name__}",
                )
            if isinstance(result, bool):
                decision = (
                    AdmissionDecision.allow(reason="BOOLEAN_PROVIDER_ALLOWED")
                    if result
                    else AdmissionDecision.block(
                        AdmissionCode.READINESS_BLOCKED,
                        "boolean admission provider blocked intent",
                    )
                )
            elif isinstance(result, AdmissionDecision):
                decision = result
            else:
                return AdmissionDecision.block(
                    AdmissionCode.READINESS_BLOCKED,
                    "admission provider returned an unsupported value",
                )
            if not decision.allowed:
                return decision
        return AdmissionDecision.allow(reason="ALL_ADMISSION_POLICIES_CLEAR")


def planner_intent_to_native(intent: object) -> NativeOrderIntent:
    action = HedgeAction(str(getattr(getattr(intent, "action", ""), "value", getattr(intent, "action", ""))))
    bucket = HedgeBucket(str(getattr(getattr(intent, "bucket", ""), "value", getattr(intent, "bucket", ""))))
    side = HedgeSide.parse(getattr(intent, "position_side"))
    return NativeOrderIntent(
        pair=str(getattr(intent, "symbol")),
        side=side,
        action=action,
        quantity=Decimal(str(getattr(intent, "quantity"))),
        price=Decimal(str(getattr(intent, "price"))),
        bucket=bucket,
        intent_id=str(getattr(intent, "intent_id", "")),
        metadata={"reason": str(getattr(intent, "reason", ""))},
    )


def apply_planning_admission_gate(
    planning: object,
    *,
    evaluate: AdmissionProvider,
    current_long_state: object,
    current_short_state: object,
) -> tuple[object, tuple[tuple[str, AdmissionDecision], ...]]:
    """Filter blocked risk-increasing planner intents while preserving reduce-only exits.

    The function is duck-typed to avoid importing the large planning graph from lightweight
    tools.  It expects a frozen dataclass compatible with ``PlanningResult``.
    """

    blocked: list[tuple[str, AdmissionDecision]] = []
    allowed: list[object] = []
    blocked_sides: set[str] = set()
    for item in tuple(getattr(planning, "submit_orders")):
        native = planner_intent_to_native(item)
        result = evaluate(native)
        if isinstance(result, bool):
            result = (
                AdmissionDecision.allow()
                if result
                else AdmissionDecision.block(
                    AdmissionCode.READINESS_BLOCKED,
                    "admission gate blocked intent",
                )
            )
        if native.reduce_only or result.allowed:
            allowed.append(item)
            continue
        blocked.append((native.intent_id, result))
        blocked_sides.add(native.side.value)
    if not blocked:
        return planning, ()
    diagnostics = tuple(getattr(planning, "diagnostics", ())) + tuple(
        f"NATIVE_ADMISSION_BLOCKED:{intent_id}:{decision.code.value}"
        for intent_id, decision in blocked
    )
    replacement = {
        "submit_orders": tuple(allowed),
        "diagnostics": diagnostics,
    }
    if "LONG" in blocked_sides:
        replacement["long_state"] = current_long_state
    if "SHORT" in blocked_sides:
        replacement["short_state"] = current_short_state
    return replace(planning, **replacement), tuple(blocked)
