from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

import pytest

from freqtrade.hedge.telemetry.events import HedgeEventType, HedgeTelemetryEvent
from freqtrade.hedge.telemetry.logging import HedgeJsonFormatter, redact
from freqtrade.hedge.telemetry.metrics import HedgeMetrics


class SampleState(StrEnum):
    READY = "READY"


def test_str_enum_payload_is_serialized_as_plain_value() -> None:
    event = HedgeTelemetryEvent(
        HedgeEventType.READINESS,
        {"state": SampleState.READY},
    )
    assert event.as_dict()["payload"]["state"] == "READY"
    json.dumps(event.as_dict(), allow_nan=False)


def test_payload_is_deeply_immutable() -> None:
    source = {"nested": {"items": [1, 2]}}
    event = HedgeTelemetryEvent(HedgeEventType.ORDER, source)
    source["nested"]["items"].append(3)
    assert event.as_dict()["payload"]["nested"]["items"] == [1, 2]
    with pytest.raises(TypeError):
        event.payload["new"] = 1
    with pytest.raises(TypeError):
        event.payload["nested"]["new"] = 1


def test_payload_rejects_duplicate_stringified_keys() -> None:
    with pytest.raises(ValueError, match="collide"):
        HedgeTelemetryEvent(HedgeEventType.ORDER, {1: "a", "1": "b"})


def test_payload_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        HedgeTelemetryEvent(
            HedgeEventType.ORDER,
            {"at": datetime.now()},
        )


def test_payload_set_order_is_deterministic() -> None:
    first = HedgeTelemetryEvent(HedgeEventType.ORDER, {"values": {3, 1, 2}})
    second = HedgeTelemetryEvent(HedgeEventType.ORDER, {"values": {2, 3, 1}})
    assert first.as_dict()["payload"] == second.as_dict()["payload"]


@pytest.mark.parametrize("version", [True, 0, -1, 2_147_483_648])
def test_schema_version_is_strict_and_bounded(version) -> None:
    with pytest.raises((TypeError, ValueError)):
        HedgeTelemetryEvent(
            HedgeEventType.ORDER,
            {},
            schema_version=version,
        )


def test_event_symbol_is_canonical_and_settlement_checked() -> None:
    event = HedgeTelemetryEvent(
        HedgeEventType.ORDER,
        {},
        symbol="eth/usdt:usdt",
    )
    assert event.symbol == "ETHUSDT"
    with pytest.raises(ValueError):
        HedgeTelemetryEvent(
            HedgeEventType.ORDER,
            {},
            symbol="ETH/USDT:USDC",
        )


@pytest.mark.parametrize("amount", [True, 0.1, float("inf"), float("nan"), -1])
def test_metric_amount_rejects_unsafe_values(amount) -> None:
    metrics = HedgeMetrics()
    with pytest.raises(ValueError):
        metrics._inc("test_total", amount)


def test_metric_labels_escape_delimiters() -> None:
    metrics = HedgeMetrics()
    metrics.reconnect("user,stream=primary", "ok")
    keys = metrics.snapshot()["reconnect_total"]
    rendered = next(iter(keys))
    assert "%2C" in rendered
    assert "%3D" in rendered


def test_halt_reason_does_not_create_unbounded_label() -> None:
    metrics = HedgeMetrics()
    metrics.halt(True, "reason-a")
    metrics.halt(True, "reason-b")
    assert len(metrics.snapshot()["halt_total"]) == 1
    assert next(iter(metrics.snapshot()["halt_total"].values())) == "2"


def test_metric_lock_requires_real_boolean() -> None:
    with pytest.raises(TypeError):
        HedgeMetrics().lock("main:ETHUSDT:LONG", 1)


def make_record(message: str, hedge=None) -> logging.LogRecord:
    record = logging.LogRecord(
        "hedge",
        logging.INFO,
        __file__,
        1,
        message,
        (),
        None,
    )
    if hedge is not None:
        record.hedge = hedge
    return record


def test_logging_redacts_extended_secret_key_families() -> None:
    protected = {
        "client_secret": "one",
        "privateSecret": "two",
        "cookie": "three",
        "session_id": "***",
        "nested": {"some_token": "five"},
    }
    rendered = json.loads(HedgeJsonFormatter().format(make_record("ok", protected)))
    assert rendered["hedge"] == {
        "client_secret": "***",
        "privateSecret": "***",
        "cookie": "***",
        "session_id": "***",
        "nested": {"some_token": "***"},
    }


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Bearer abc.def.ghi",
        "authorization='secret-token'",
        "api_key=plain-secret",
        "password: \"quoted-secret\"",
        "request failed Bearer abc123",
    ],
)
def test_logging_redacts_inline_secrets(message: str) -> None:
    rendered = json.loads(HedgeJsonFormatter().format(make_record(message)))
    assert "secret-token" not in rendered["message"]
    assert "plain-secret" not in rendered["message"]
    assert "quoted-secret" not in rendered["message"]
    assert "abc.def.ghi" not in rendered["message"]
    assert "abc123" not in rendered["message"]
    assert "***" in rendered["message"]


def test_logging_rejects_nonfinite_values_in_structured_payload() -> None:
    with pytest.raises(ValueError):
        HedgeJsonFormatter().format(make_record("bad", {"value": float("nan")}))


def test_redact_preserves_safe_decimal_and_aware_datetime() -> None:
    value = redact(
        {
            "quantity": Decimal("1.25"),
            "at": datetime.now(UTC),
        }
    )
    assert value["quantity"] == "1.25"
    assert value["at"].endswith("+00:00")


def test_telemetry_payload_has_global_node_budget() -> None:
    payload = {f"k{i}": list(range(100)) for i in range(100)}
    with pytest.raises(ValueError, match="too many values"):
        HedgeTelemetryEvent(HedgeEventType.ORDER, payload)


def test_structured_logging_rejects_cycles_and_unsupported_objects() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="reference cycle"):
        redact(cyclic)
    with pytest.raises(TypeError, match="unsupported type"):
        redact({"object": object()})


def test_structured_logging_set_output_is_deterministic() -> None:
    assert redact({"values": {3, 1, 2}}) == {"values": [1, 2, 3]}


def test_structured_logging_rejects_key_collision_after_conversion() -> None:
    with pytest.raises(ValueError, match="collide"):
        redact({1: "a", "1": "b"})
