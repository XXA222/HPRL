"""Cancel-replace coordinator that never overlaps old and replacement risk."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from uuid import uuid4

from .service import ExecutionResult, ExecutionService, OrderIntent, OrderType
from .state_machine import OrderState


@dataclass(frozen=True, slots=True)
class CancelReplaceResult:
    canceled: ExecutionResult
    replacement: ExecutionResult | None
    completed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.canceled, ExecutionResult):
            raise TypeError("canceled must be an ExecutionResult")
        if self.replacement is not None and not isinstance(
            self.replacement,
            ExecutionResult,
        ):
            raise TypeError("replacement must be an ExecutionResult or None")
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason is required")


class CancelReplaceCoordinator:
    def __init__(self, service: ExecutionService) -> None:
        if not isinstance(service, ExecutionService):
            raise TypeError("service must be an ExecutionService")
        self._service = service

    def execute(
        self,
        *,
        original_client_order_id: str,
        replacement_intent: OrderIntent,
    ) -> CancelReplaceResult:
        if not isinstance(replacement_intent, OrderIntent):
            raise TypeError("replacement_intent must be an OrderIntent")
        original = self._service.get_order(original_client_order_id)
        with self._service.leg_guard(original):
            original = self._service.get_order(original_client_order_id)
            self._validate_replacement(original.intent, replacement_intent)
            remaining_before = (
                original.approved_quantity - original.lifecycle.filled_quantity
            )
            if remaining_before <= 0:
                canceled = self._service.cancel(original_client_order_id)
                return CancelReplaceResult(
                    canceled=canceled,
                    replacement=None,
                    completed=False,
                    reason="original order has no remaining quantity",
                )
            if replacement_intent.quantity > remaining_before:
                raise ValueError(
                    "replacement quantity exceeds original remaining quantity"
                )

            canceled = self._service.cancel(original_client_order_id)
            if canceled.order.lifecycle.status is not OrderState.CANCELED:
                return CancelReplaceResult(
                    canceled=canceled,
                    replacement=None,
                    completed=False,
                    reason="original order cancellation not confirmed",
                )
            remaining_after = (
                canceled.order.approved_quantity
                - canceled.order.lifecycle.filled_quantity
            )
            if remaining_after <= 0:
                return CancelReplaceResult(
                    canceled=canceled,
                    replacement=None,
                    completed=False,
                    reason="original order filled during cancellation",
                )
            if replacement_intent.quantity > remaining_after:
                return CancelReplaceResult(
                    canceled=canceled,
                    replacement=None,
                    completed=False,
                    reason="replacement exceeds post-cancel remaining quantity",
                )
            replacement = self._service.submit(replacement_intent)
            return CancelReplaceResult(
                canceled=canceled,
                replacement=replacement,
                completed=True,
                reason="replacement submitted after confirmed cancellation",
            )

    def execute_remaining(
        self,
        *,
        original_client_order_id: str,
        idempotency_key: str,
        order_type: OrderType | str | None = None,
        limit_price: Decimal | str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CancelReplaceResult:
        """Cancel the original and replace exactly its remaining quantity."""
        original = self._service.get_order(original_client_order_id)
        with self._service.leg_guard(original):
            original = self._service.get_order(original_client_order_id)
            remaining = (
                original.approved_quantity - original.lifecycle.filled_quantity
            )
            if remaining <= 0:
                canceled = self._service.cancel(original_client_order_id)
                return CancelReplaceResult(
                    canceled=canceled,
                    replacement=None,
                    completed=False,
                    reason="original order has no remaining quantity",
                )
            selected_type = original.intent.order_type
            if order_type is not None:
                try:
                    selected_type = (
                        order_type
                        if isinstance(order_type, OrderType)
                        else OrderType(order_type)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("order_type is invalid") from exc
            selected_price: Decimal | None
            if selected_type is OrderType.MARKET:
                selected_price = None
            elif limit_price is None:
                selected_price = original.intent.limit_price
            else:
                selected_price = Decimal(limit_price)
            replacement_metadata = dict(original.intent.metadata)
            replacement_metadata.update(metadata or {})
            replacement_metadata["replaces_client_order_id"] = (
                original_client_order_id
            )
            replacement_metadata["parent_intent_id"] = str(original.intent.intent_id)
            replacement = replace(
                original.intent,
                intent_id=uuid4(),
                quantity=remaining,
                idempotency_key=idempotency_key,
                order_type=selected_type,
                limit_price=selected_price,
                metadata=replacement_metadata,
            )
            return self.execute(
                original_client_order_id=original_client_order_id,
                replacement_intent=replacement,
            )

    @staticmethod
    def _validate_replacement(
        original: OrderIntent,
        replacement: OrderIntent,
    ) -> None:
        if replacement.idempotency_key == original.idempotency_key:
            raise ValueError("replacement intent must use a new idempotency key")
        if (
            replacement.account_id != original.account_id
            or replacement.symbol != original.symbol
            or replacement.position_side is not original.position_side
        ):
            raise ValueError(
                "replacement intent must target the same account, symbol and side"
            )
        if replacement.action is not original.action:
            raise ValueError("replacement intent must preserve the original action")
        if replacement.reduce_only != original.reduce_only:
            raise ValueError("replacement intent must preserve reduce_only semantics")
        if replacement.action_group_id != original.action_group_id:
            raise ValueError("replacement intent must preserve action_group_id")
