from decimal import Decimal

from freqtrade.hedge.telemetry.metrics import HedgeMetrics


def test_extended_operational_metric_families_are_stable():
    metrics = HedgeMetrics()
    metrics.order_ack_latency(Decimal("0.125"), exchange="binance")
    metrics.unknown_orders(2, leg="ETHUSDT:LONG")
    metrics.unknown_order_age(Decimal("12.5"), leg="ETHUSDT:LONG")
    metrics.cancel_replace("completed")
    metrics.duplicate_fill("user_stream")
    metrics.outbox(3)
    metrics.projection(9, projection="position")
    metrics.reconciliation_diff("HIGH", resolved=False)
    metrics.rest_error("429", endpoint="open_orders")
    metrics.clock_skew(Decimal("-15"))
    metrics.account_risk(
        gross_long=Decimal("100"),
        gross_short=Decimal("50"),
        net=Decimal("50"),
        gross_ratio=Decimal("0.15"),
        margin_utilization=Decimal("0.05"),
        pending_risk=Decimal("10"),
        account_id="hedge-main",
    )
    metrics.single_writer(leader=True, lease_age_seconds=Decimal("2"))
    metrics.lock_wait(Decimal("0.01"), scope="position")
    metrics.lock_timeout(scope="position")
    metrics.deadlock(scope="account-position")

    snapshot = metrics.snapshot()
    for family in HedgeMetrics.REQUIRED_FAMILIES:
        assert family in snapshot
    assert snapshot["net_notional"]["account_id=hedge-main"] == "50"
    assert snapshot["clock_skew_milliseconds"]["source=exchange"] == "15"


def test_halt_reason_is_normalized_in_separate_bounded_family():
    metrics = HedgeMetrics()
    metrics.halt(True, "single writer lost: lease 123")
    snapshot = metrics.snapshot()
    assert snapshot["halt_total"] == {"active=true": "1"}
    assert snapshot["halt_reason_total"] == {
        "reason=SINGLE_WRITER_LOST_LEASE_123": "1"
    }
