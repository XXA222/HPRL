import json
import logging
from decimal import Decimal

from freqtrade.hedge.telemetry.events import HedgeEventType, HedgeTelemetryEvent
from freqtrade.hedge.telemetry.logging import HedgeJsonFormatter
from freqtrade.hedge.telemetry.metrics import HedgeMetrics


def test_required_metric_families_are_always_present() -> None:
    metrics = HedgeMetrics()
    metrics.intent("approved")
    metrics.order("PREPARED", "SUBMITTING")
    metrics.fill(Decimal("0.1"))
    metrics.drift("position")
    metrics.halt(True, "manual")
    metrics.reconnect("user_stream", "success")
    metrics.lock("main:ETHUSDT:LONG", True)
    snapshot = metrics.snapshot()
    assert set(HedgeMetrics.REQUIRED_FAMILIES).issubset(snapshot)
    assert snapshot["fill_quantity_total"]["_"] == "0.1"


def test_event_schema_and_log_redaction() -> None:
    event = HedgeTelemetryEvent(
        HedgeEventType.ORDER,
        {"status": "ACKNOWLEDGED"},
        account_id="main",
        symbol="ETHUSDT",
    )
    payload = event.as_dict()
    assert payload["schema_version"] == 1
    assert payload["event_type"] == "ORDER"

    record = logging.LogRecord("hedge", logging.INFO, __file__, 1, "test", (), None)
    record.hedge = {"token": "secret-token", "nested": {"api_key": "secret-key"}}
    rendered = json.loads(HedgeJsonFormatter().format(record))
    assert rendered["hedge"]["token"] == "***"
    assert rendered["hedge"]["nested"]["api_key"] == "***"


def test_event_payload_is_json_safe_and_immutable() -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    source = {"quantity": Decimal("1.25"), "at": datetime.now(UTC), "id": uuid4()}
    event = HedgeTelemetryEvent(HedgeEventType.FILL, source)
    source["quantity"] = Decimal("9")
    payload = event.as_dict()["payload"]
    assert payload["quantity"] == "1.25"
    assert isinstance(payload["at"], str)
    assert isinstance(payload["id"], str)


def test_metrics_reject_negative_fill_and_redact_key_variants() -> None:
    import pytest

    metrics = HedgeMetrics()
    with pytest.raises(ValueError, match="non-negative"):
        metrics.fill(Decimal("-1"))
    record = logging.LogRecord("hedge", logging.INFO, __file__, 1, "test", (), None)
    record.hedge = {"access_token": "secret", "apiSecret": "secret-2"}
    rendered = json.loads(HedgeJsonFormatter().format(record))
    assert rendered["hedge"] == {"access_token": "***", "apiSecret": "***"}
