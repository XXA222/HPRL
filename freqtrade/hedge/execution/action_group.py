"""Grouped actions, including durable close-both decomposition and reporting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Iterable, Protocol
from uuid import UUID, uuid4

from .action_group_store import (
    ActionGroupMember,
    ActionGroupMemberState,
    ActionGroupRecord,
    ActionGroupRepository,
)
from .service import ExecutionResult, IntentAction, OrderIntent, OrderType, PositionSide
from .state_machine import OrderState


class _SubmitPort(Protocol):
    def submit(self, intent: OrderIntent) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class ActionGroupReport:
    action_group_id: UUID
    results: tuple[ExecutionResult, ...]
    errors: tuple[str, ...]
    skipped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action_group_id, UUID):
            raise TypeError("action_group_id must be a UUID")
        if any(not isinstance(item, ExecutionResult) for item in self.results):
            raise TypeError("results must contain ExecutionResult values")
        if any(not isinstance(item, str) for item in self.errors):
            raise TypeError("errors must contain strings")
        if any(not isinstance(item, str) for item in self.skipped):
            raise TypeError("skipped must contain strings")

    @property
    def fully_successful(self) -> bool:
        return bool(self.results or self.skipped) and not self.errors and all(
            result.order.lifecycle.status
            in {OrderState.ACKNOWLEDGED, OrderState.PARTIAL, OrderState.FILLED}
            for result in self.results
        )

    @property
    def partially_successful(self) -> bool:
        successes = len(self.skipped) + sum(
            result.order.lifecycle.status
            in {OrderState.ACKNOWLEDGED, OrderState.PARTIAL, OrderState.FILLED}
            for result in self.results
        )
        return successes > 0 and not self.fully_successful

    @property
    def attempted(self) -> int:
        return len(self.results) + len(self.errors) + len(self.skipped)

    @property
    def terminal(self) -> bool:
        return bool(self.results or self.skipped) and not self.errors and all(
            result.order.lifecycle.terminal for result in self.results
        )

    @property
    def filled_quantity(self) -> Decimal:
        return sum(
            (result.order.lifecycle.filled_quantity for result in self.results),
            Decimal("0"),
        )

    @property
    def unknown_count(self) -> int:
        return sum(
            result.order.lifecycle.status is OrderState.UNKNOWN
            for result in self.results
        )

    @property
    def rejected_count(self) -> int:
        return sum(
            result.order.lifecycle.status is OrderState.REJECTED
            for result in self.results
        )

    @property
    def outcome(self) -> str:
        if self.errors:
            return "PARTIAL_FAILURE" if self.partially_successful else "FAILED"
        if self.unknown_count:
            return "UNKNOWN"
        if self.rejected_count:
            return "PARTIAL_FAILURE" if self.partially_successful else "REJECTED"
        if self.terminal:
            return "COMPLETED"
        if self.fully_successful:
            return "ACCEPTED"
        return "IN_PROGRESS"


def _group_id(value: UUID | None) -> UUID:
    if value is None:
        return uuid4()
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("action_group_id must be a UUID") from exc


def _base_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("idempotency_key must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("idempotency_key is required")
    if len(normalized) > 250 or any(ord(ch) < 32 or ord(ch) == 127 for ch in normalized):
        raise ValueError("idempotency_key is invalid")
    return normalized


def build_close_both_intents(
    *,
    account_id: str,
    symbol: str,
    long_quantity: Decimal,
    short_quantity: Decimal,
    idempotency_key: str,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    action_group_id: UUID | None = None,
) -> tuple[OrderIntent, OrderIntent]:
    """Backward-compatible strict builder requiring both legs to be non-zero."""
    group, intents = build_close_both_plan(
        account_id=account_id,
        symbol=symbol,
        long_quantity=long_quantity,
        short_quantity=short_quantity,
        idempotency_key=idempotency_key,
        order_type=order_type,
        long_limit_price=limit_price,
        short_limit_price=limit_price,
        action_group_id=action_group_id,
    )
    if len(intents) != 2:
        raise ValueError("strict close-both builder requires two positive leg quantities")
    return intents[0], intents[1]


def build_close_both_plan(
    *,
    account_id: str,
    symbol: str,
    long_quantity: Decimal,
    short_quantity: Decimal,
    idempotency_key: str,
    order_type: OrderType = OrderType.MARKET,
    long_limit_price: Decimal | None = None,
    short_limit_price: Decimal | None = None,
    action_group_id: UUID | None = None,
) -> tuple[ActionGroupRecord, tuple[OrderIntent, ...]]:
    """Create the durable group before either leg is submitted.

    Flat legs are represented as ``SKIPPED_ALREADY_FLAT`` instead of failing the whole group.
    """
    base_key = _base_key(idempotency_key)
    group_id = _group_id(action_group_id)
    quantities = {
        PositionSide.LONG: Decimal(long_quantity),
        PositionSide.SHORT: Decimal(short_quantity),
    }
    prices = {
        PositionSide.LONG: long_limit_price,
        PositionSide.SHORT: short_limit_price,
    }
    members: list[ActionGroupMember] = []
    intents: list[OrderIntent] = []
    for side in (PositionSide.LONG, PositionSide.SHORT):
        quantity = quantities[side]
        if not quantity.is_finite() or quantity < 0:
            raise ValueError("close-both quantities must be finite and non-negative")
        if quantity == 0:
            members.append(
                ActionGroupMember(
                    side,
                    ActionGroupMemberState.SKIPPED_ALREADY_FLAT,
                )
            )
            continue
        intent = OrderIntent(
            account_id=account_id,
            symbol=symbol,
            position_side=side,
            action=IntentAction.CLOSE,
            quantity=quantity,
            idempotency_key=f"{base_key}:{side.value}",
            order_type=order_type,
            limit_price=prices[side],
            reduce_only=True,
            action_group_id=group_id,
            metadata={"action_group_type": "CLOSE_BOTH"},
        )
        intents.append(intent)
        members.append(
            ActionGroupMember(
                side,
                ActionGroupMemberState.PLANNED,
                intent_id=str(intent.intent_id),
            )
        )
    record = ActionGroupRecord(
        action_group_id=group_id,
        action_type="CLOSE_BOTH",
        account_id=account_id,
        symbol=symbol,
        members=tuple(members),
    )
    return record, tuple(intents)


class ActionGroupExecutor:
    def __init__(
        self,
        service: _SubmitPort,
        repository: ActionGroupRepository | None = None,
    ) -> None:
        if not callable(getattr(service, "submit", None)):
            raise TypeError("service must expose submit(intent)")
        self._service = service
        self._repository = repository

    def execute(self, intents: Iterable[OrderIntent]) -> ActionGroupReport:
        source = tuple(intents)
        if not source:
            raise ValueError("action group must not be empty")
        if any(not isinstance(intent, OrderIntent) for intent in source):
            raise TypeError("action group must contain OrderIntent values")
        group_ids = {intent.action_group_id for intent in source}
        if len(group_ids) != 1 or None in group_ids:
            raise ValueError("all intents must share a non-null action_group_id")
        group_id = next(iter(group_ids))
        if not isinstance(group_id, UUID):
            raise TypeError("action_group_id must be a UUID")
        results: list[ExecutionResult] = []
        errors: list[str] = []
        for intent in source:
            try:
                result = self._service.submit(intent)
                if not isinstance(result, ExecutionResult):
                    raise TypeError("submit must return ExecutionResult")
                results.append(result)
                self._update_member(
                    group_id,
                    ActionGroupMember(
                        intent.position_side,
                        ActionGroupMemberState.SUBMITTED,
                        intent_id=str(intent.intent_id),
                        client_order_id=result.order.client_order_id,
                    ),
                )
            except Exception as exc:
                message = f"{intent.position_side.value}:{type(exc).__name__}:{exc}"
                errors.append(message)
                self._update_member(
                    group_id,
                    ActionGroupMember(
                        intent.position_side,
                        ActionGroupMemberState.FAILED,
                        intent_id=str(intent.intent_id),
                        error=message,
                    ),
                )
        return ActionGroupReport(group_id, tuple(results), tuple(errors))

    def execute_plan(
        self,
        record: ActionGroupRecord,
        intents: Iterable[OrderIntent],
    ) -> ActionGroupReport:
        if not isinstance(record, ActionGroupRecord):
            raise TypeError("record must be ActionGroupRecord")
        source = tuple(intents)
        if any(intent.action_group_id != record.action_group_id for intent in source):
            raise ValueError("action group record and intents must share identity")
        if self._repository is not None:
            self._repository.put(record)
        skipped = tuple(
            f"{member.position_side.value}:SKIPPED_ALREADY_FLAT"
            for member in record.members
            if member.state is ActionGroupMemberState.SKIPPED_ALREADY_FLAT
        )
        if not source:
            return ActionGroupReport(record.action_group_id, (), (), skipped)
        report = self.execute(source)
        return replace(report, skipped=skipped)

    def execute_close_both(
        self,
        *,
        account_id: str,
        symbol: str,
        long_quantity: Decimal,
        short_quantity: Decimal,
        idempotency_key: str,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        long_limit_price: Decimal | None = None,
        short_limit_price: Decimal | None = None,
        action_group_id: UUID | None = None,
    ) -> ActionGroupReport:
        record, intents = build_close_both_plan(
            account_id=account_id,
            symbol=symbol,
            long_quantity=long_quantity,
            short_quantity=short_quantity,
            idempotency_key=idempotency_key,
            order_type=order_type,
            long_limit_price=long_limit_price if long_limit_price is not None else limit_price,
            short_limit_price=short_limit_price if short_limit_price is not None else limit_price,
            action_group_id=action_group_id,
        )
        return self.execute_plan(record, intents)

    def refresh(self, report: ActionGroupReport) -> ActionGroupReport:
        """Refresh group orders from the exchange when the service supports it."""
        if not isinstance(report, ActionGroupReport):
            raise TypeError("report must be an ActionGroupReport")
        refresh_order = getattr(self._service, "refresh_order", None)
        get_order = getattr(self._service, "get_order", None)
        if not callable(refresh_order) and not callable(get_order):
            raise TypeError("service must expose refresh_order or get_order")
        results: list[ExecutionResult] = []
        errors = list(report.errors)
        for result in report.results:
            try:
                if callable(refresh_order):
                    latest_result = refresh_order(result.order.client_order_id)
                    if not isinstance(latest_result, ExecutionResult):
                        raise TypeError("refresh_order must return ExecutionResult")
                    results.append(latest_result)
                else:
                    latest = get_order(result.order.client_order_id)
                    results.append(replace(result, order=latest))
            except Exception as exc:
                errors.append(
                    f"{result.order.intent.position_side.value}:"
                    f"{type(exc).__name__}:{exc}"
                )
        return ActionGroupReport(
            report.action_group_id,
            tuple(results),
            tuple(errors),
            report.skipped,
        )

    def _update_member(self, group_id: UUID, member: ActionGroupMember) -> None:
        if self._repository is None:
            return
        self._repository.update_member(group_id, member)
