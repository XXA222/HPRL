"""Pairlist-to-Hedge universe translation and account-level capital allocation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Iterable, Mapping

from freqtrade.hedge.symbols import canonicalize_symbol, raw_symbol

from .models import AdmissionCode, AdmissionDecision, NativeOrderIntent, ONE, ZERO, finite_decimal, utc_datetime


class SymbolAssignmentState(StrEnum):
    ACTIVE = "ACTIVE"
    REDUCE_ONLY = "REDUCE_ONLY"
    DRAINING = "DRAINING"
    REMOVED = "REMOVED"


@dataclass(frozen=True, slots=True)
class HedgeSymbolAssignment:
    pair: str
    raw_symbol: str
    state: SymbolAssignmentState
    weight: Decimal
    capital_limit: Decimal
    reason: str


@dataclass(frozen=True, slots=True)
class HedgeUniverseSnapshot:
    version: int
    assignments: tuple[HedgeSymbolAssignment, ...]
    source_pairs: tuple[str, ...]
    blacklist: tuple[str, ...]
    observed_at: datetime

    @property
    def active_pairs(self) -> tuple[str, ...]:
        return tuple(
            item.pair for item in self.assignments if item.state is SymbolAssignmentState.ACTIVE
        )

    def assignment(self, pair: str) -> HedgeSymbolAssignment | None:
        normalized = canonicalize_symbol(pair)
        return next((item for item in self.assignments if item.pair == normalized), None)


class HedgeUniverseManager:
    """Maintain a deterministic, side-neutral trading universe from Pairlist snapshots."""

    def __init__(
        self,
        *,
        max_symbols: int = 20,
        configured_weights: Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(max_symbols, bool) or max_symbols < 1:
            raise ValueError("max_symbols must be a positive integer")
        self.max_symbols = int(max_symbols)
        self.configured_weights = {
            canonicalize_symbol(pair): finite_decimal(weight, field_name=f"weight:{pair}")
            for pair, weight in (configured_weights or {}).items()
        }
        if any(value < ZERO for value in self.configured_weights.values()):
            raise ValueError("universe weights cannot be negative")
        self._version = 0
        self._snapshot = HedgeUniverseSnapshot(0, (), (), (), utc_datetime())

    @staticmethod
    def _normalize_pairs(pairs: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(canonicalize_symbol(str(item)) for item in pairs))

    def refresh(
        self,
        pairs: Iterable[str],
        *,
        blacklist: Iterable[str] = (),
        account_capital: object = ZERO,
        open_position_pairs: Iterable[str] = (),
        active_order_pairs: Iterable[str] = (),
        at: datetime | None = None,
    ) -> HedgeUniverseSnapshot:
        source = self._normalize_pairs(pairs)
        blocked = set(self._normalize_pairs(blacklist))
        open_pairs = set(self._normalize_pairs(open_position_pairs))
        order_pairs = set(self._normalize_pairs(active_order_pairs))
        selected = tuple(item for item in source if item not in blocked)[: self.max_symbols]
        retained = tuple(
            item
            for item in dict.fromkeys((*selected, *open_pairs, *order_pairs))
            if item not in selected
        )
        capital = finite_decimal(account_capital, field_name="account_capital")
        if capital < ZERO:
            raise ValueError("account_capital cannot be negative")

        raw_weights = {
            pair: self.configured_weights.get(pair, ONE)
            for pair in selected
        }
        weight_total = sum(raw_weights.values(), ZERO)
        if selected and weight_total <= ZERO:
            raw_weights = {pair: ONE for pair in selected}
            weight_total = Decimal(len(selected))

        assignments: list[HedgeSymbolAssignment] = []
        for pair in selected:
            weight = raw_weights[pair] / weight_total if weight_total > ZERO else ZERO
            assignments.append(
                HedgeSymbolAssignment(
                    pair=pair,
                    raw_symbol=raw_symbol(pair),
                    state=SymbolAssignmentState.ACTIVE,
                    weight=weight,
                    capital_limit=capital * weight,
                    reason="PAIRLIST_ACTIVE",
                )
            )
        for pair in retained:
            reason = "BLACKLIST_DRAIN" if pair in blocked else "PAIRLIST_REMOVED_DRAIN"
            assignments.append(
                HedgeSymbolAssignment(
                    pair=pair,
                    raw_symbol=raw_symbol(pair),
                    state=SymbolAssignmentState.DRAINING,
                    weight=ZERO,
                    capital_limit=ZERO,
                    reason=reason,
                )
            )
        self._version += 1
        self._snapshot = HedgeUniverseSnapshot(
            self._version,
            tuple(assignments),
            source,
            tuple(sorted(blocked)),
            utc_datetime(at),
        )
        return self._snapshot

    @property
    def snapshot(self) -> HedgeUniverseSnapshot:
        return self._snapshot

    def status(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "version": snapshot.version,
            "active_pairs": list(snapshot.active_pairs),
            "blacklist": list(snapshot.blacklist),
            "assignments": [
                {
                    "pair": item.pair,
                    "state": item.state.value,
                    "weight": str(item.weight),
                    "capital_limit": str(item.capital_limit),
                    "reason": item.reason,
                }
                for item in snapshot.assignments
            ],
            "observed_at": snapshot.observed_at.isoformat(),
        }

    def admit(self, intent: NativeOrderIntent) -> AdmissionDecision:
        assignment = self._snapshot.assignment(intent.pair)
        if assignment is None:
            if intent.reduce_only:
                return AdmissionDecision.allow(
                    reason="UNIVERSE_UNKNOWN_REDUCE_ONLY",
                    reduce_only_exempt=True,
                )
            return AdmissionDecision.block(
                AdmissionCode.UNIVERSE_REJECTED,
                "pair is not assigned to the Hedge universe",
            )
        if assignment.state is SymbolAssignmentState.ACTIVE:
            return AdmissionDecision.allow(reason="UNIVERSE_ACTIVE")
        if intent.reduce_only and assignment.state in {
            SymbolAssignmentState.REDUCE_ONLY,
            SymbolAssignmentState.DRAINING,
        }:
            return AdmissionDecision.allow(
                reason=f"UNIVERSE_{assignment.state.value}_REDUCE_ONLY",
                reduce_only_exempt=True,
            )
        return AdmissionDecision.block(
            AdmissionCode.UNIVERSE_REJECTED,
            f"pair assignment state {assignment.state.value} does not allow new risk",
        )
