import pytest

from freqtrade.hedge.execution.client_order_id import (
    build_client_order_id,
    validate_client_order_id,
)


def test_client_order_id_is_deterministic_bounded_and_attempt_specific() -> None:
    common = dict(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="LONG",
        idempotency_key="intent-1",
    )
    first = build_client_order_id(**common)
    replay = build_client_order_id(**common)
    retry = build_client_order_id(**common, attempt=1)
    maximum = build_client_order_id(**common, attempt=36**4 - 1, prefix="HEDGE")

    assert first == replay
    assert first != retry
    assert len(first) <= 36
    assert len(maximum) <= 36
    validate_client_order_id(first)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"attempt": -1}, ValueError),
        ({"attempt": 36**4}, ValueError),
        ({"attempt": True}, TypeError),
        ({"symbol": "///"}, ValueError),
        ({"position_side": "BOTH"}, ValueError),
        ({"account_id": " "}, ValueError),
        ({"idempotency_key": " "}, ValueError),
        ({"prefix": "***"}, ValueError),
    ],
)
def test_client_order_id_rejects_invalid_inputs(overrides, error) -> None:
    arguments = {
        "account_id": "main",
        "symbol": "ETHUSDT",
        "position_side": "LONG",
        "idempotency_key": "intent-1",
    }
    arguments.update(overrides)
    with pytest.raises(error):
        build_client_order_id(**arguments)


def test_client_order_id_validator_rejects_invalid_characters_and_length() -> None:
    with pytest.raises(ValueError):
        validate_client_order_id("bad id")
    with pytest.raises(ValueError):
        validate_client_order_id("X" * 37)
