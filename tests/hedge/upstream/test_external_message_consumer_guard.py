from __future__ import annotations

import pytest

from freqtrade.rpc.external_message_consumer import _producer_ws_url


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("127.0.0.1", -1),
        ("10000.1241..2121/", 9989),
        ("bad host", 9989),
        ("host?query", 9989),
    ],
)
def test_external_message_consumer_rejects_invalid_endpoint(host: str, port: int) -> None:
    producer = {
        "name": "default",
        "host": host,
        "port": port,
        "secure": False,
        "ws_token": "token",
    }

    with pytest.raises(ValueError):
        _producer_ws_url(producer)


def test_external_message_consumer_builds_ipv4_and_ipv6_urls() -> None:
    ipv4 = {
        "name": "default",
        "host": "127.0.0.1",
        "port": 9989,
        "secure": False,
        "ws_token": "token",
    }
    ipv6 = dict(ipv4, host="::1", secure=True)

    assert _producer_ws_url(ipv4) == "ws://127.0.0.1:9989/api/v1/message/ws?token=token"
    assert _producer_ws_url(ipv6) == "wss://[::1]:9989/api/v1/message/ws?token=token"
