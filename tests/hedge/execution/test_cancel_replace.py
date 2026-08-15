from decimal import Decimal

from freqtrade.hedge.execution import (
    CancelReplaceCoordinator,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    build_fake_execution_harness,
)


def test_replacement_gets_new_intent_identity_and_parent_link() -> None:
    harness = build_fake_execution_harness()
    original = harness.service.submit(
        OrderIntent(
            account_id="acct",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key="original",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
        )
    )
    result = CancelReplaceCoordinator(harness.service).execute_remaining(
        original_client_order_id=original.order.client_order_id,
        idempotency_key="replacement",
        limit_price=Decimal("99"),
    )
    assert result.replacement is not None
    replacement = result.replacement.order.intent
    assert replacement.intent_id != original.order.intent.intent_id
    assert replacement.metadata["parent_intent_id"] == str(original.order.intent.intent_id)
