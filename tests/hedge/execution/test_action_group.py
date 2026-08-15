from decimal import Decimal

from freqtrade.hedge.execution import (
    ActionGroupExecutor,
    InMemoryActionGroupRepository,
    OrderType,
    build_integrated_fake_runtime,
)
from freqtrade.hedge.execution.action_group_store import ActionGroupMemberState


def test_close_both_allows_flat_leg_and_persists_expected_members() -> None:
    runtime = build_integrated_fake_runtime()
    repository = InMemoryActionGroupRepository()
    executor = ActionGroupExecutor(runtime.engine, repository)
    report = executor.execute_close_both(
        account_id="acct",
        symbol="ETHUSDT",
        long_quantity=Decimal("1"),
        short_quantity=Decimal("0"),
        idempotency_key="close",
        order_type=OrderType.MARKET,
    )
    assert report.attempted == 2
    assert report.skipped == ("SHORT:SKIPPED_ALREADY_FLAT",)
    record = repository.get(report.action_group_id)
    assert record is not None
    assert record.member(record.members[1].position_side).state in {
        ActionGroupMemberState.SKIPPED_ALREADY_FLAT,
        ActionGroupMemberState.SUBMITTED,
    }


def test_close_both_supports_distinct_limit_prices() -> None:
    runtime = build_integrated_fake_runtime()
    executor = ActionGroupExecutor(runtime.engine)
    report = executor.execute_close_both(
        account_id="acct",
        symbol="ETHUSDT",
        long_quantity=Decimal("1"),
        short_quantity=Decimal("2"),
        idempotency_key="close-limit",
        order_type=OrderType.LIMIT,
        long_limit_price=Decimal("101"),
        short_limit_price=Decimal("99"),
    )
    prices = {
        result.order.intent.position_side.value: result.order.intent.limit_price
        for result in report.results
    }
    assert prices == {"LONG": Decimal("101"), "SHORT": Decimal("99")}
