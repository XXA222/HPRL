from datetime import UTC, datetime
from decimal import Decimal

import pytest

from freqtrade.hedge.exchange.base import stable_fingerprint, to_primitive
from freqtrade.hedge.exchange.binance_normalizer import (
    finite_decimal,
    normalize_account_snapshot,
    normalize_configuration,
    normalize_fill,
    normalize_order,
    normalize_position,
    normalize_positions,
    observed_time,
)
from freqtrade.hedge.exchange.rate_limit import BinanceDataError


def test_finite_decimal_rejects_nan_and_infinity():
    for value in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(BinanceDataError):
            finite_decimal(value, field="x")


def test_explicit_position_side_must_match_signed_amount():
    payload = {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "positionAmt": "-1",
        "marginType": "cross",
    }
    with pytest.raises(BinanceDataError):
        normalize_position(payload, account_id="acct")

    payload.update(positionSide="SHORT", positionAmt="-2")
    fact = normalize_position(payload, account_id="acct")
    assert fact.quantity == Decimal(2)


def test_string_false_is_not_treated_as_true_for_reduce_only_and_buyer():
    order = normalize_order(
        {
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "orderId": 1,
            "side": "BUY",
            "type": "LIMIT",
            "status": "NEW",
            "origQty": "1",
            "executedQty": "0",
            "avgPrice": "0",
            "reduceOnly": "false",
        },
        account_id="acct",
    )
    assert order.reduce_only is False

    fill = normalize_fill(
        {
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "id": 1,
            "orderId": 1,
            "buyer": "false",
            "qty": "1",
            "price": "1",
        },
        account_id="acct",
    )
    assert fill.side == "SELL"


def test_configuration_boolean_is_strict():
    with pytest.raises(BinanceDataError):
        normalize_configuration(
            account_id="acct",
            dual_side_payload={"dualSidePosition": "yes"},
            symbol_configurations=[],
            managed_symbols=["BTCUSDT"],
        )


def test_duplicate_position_and_asset_rows_are_rejected():
    row = {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "positionAmt": "0",
        "marginType": "cross",
        "leverage": "5",
    }
    with pytest.raises(BinanceDataError, match="duplicate position"):
        normalize_positions([row, row], account_id="acct")

    now = datetime.now(UTC)
    account_payload = {
        "assets": [
            {"asset": "USDT"},
            {"asset": "USDT"},
        ]
    }
    with pytest.raises(BinanceDataError, match="duplicate account asset"):
        normalize_account_snapshot(
            account_payload,
            account_id="acct",
            collection_started_at=now,
            collection_completed_at=now,
        )


def test_position_requires_explicit_margin_mode_source():
    payload = {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "positionAmt": "1",
    }
    with pytest.raises(BinanceDataError, match="Margin mode"):
        normalize_position(payload, account_id="acct")



def test_v3_position_uses_symbol_configuration_without_mutating_raw():
    payload = {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "positionAmt": "1",
        "entryPrice": "3000",
        "markPrice": "3010",
        "unRealizedProfit": "10",
        "liquidationPrice": "0",
        "isolatedMargin": "0",
        "isolatedWallet": "0",
        "updateTime": 1,
    }
    configurations = [
        {
            "symbol": "ETHUSDT",
            "marginType": "CROSSED",
            "leverage": 7,
        }
    ]

    fact = normalize_positions(
        [payload],
        account_id="acct",
        symbol_configurations=configurations,
    )[0]

    assert fact.margin_mode == "cross"
    assert fact.leverage == 7
    assert fact.raw is payload
    assert "marginType" not in fact.raw
    assert "leverage" not in fact.raw


def test_configuration_ignores_unmanaged_delivery_contract_rows():
    configuration = normalize_configuration(
        account_id="acct",
        dual_side_payload={"dualSidePosition": True},
        symbol_configurations=[
            {
                "symbol": "ETHUSDT_261225",
                "marginType": "CROSSED",
                "leverage": 4,
            },
            {
                "symbol": "ETHUSDT",
                "marginType": "CROSSED",
                "leverage": 7,
            },
        ],
        managed_symbols=["ETHUSDT"],
    )

    assert configuration.active_margin_modes == ("cross",)
    assert configuration.leverage_by_symbol_side == {
        "ETHUSDT:LONG": 7,
        "ETHUSDT:SHORT": 7,
    }
    raw_rows = configuration.raw["symbol_configurations"]
    assert tuple(row["symbol"] for row in raw_rows) == ("ETHUSDT",)


def test_v3_position_ignores_unmanaged_delivery_symbol_configuration():
    payload = {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "positionAmt": "1",
        "entryPrice": "3000",
        "markPrice": "3010",
        "unRealizedProfit": "10",
        "liquidationPrice": "0",
        "isolatedMargin": "0",
        "isolatedWallet": "0",
        "updateTime": 1,
    }
    configurations = [
        {
            "symbol": "ETHUSDT_261225",
            "marginType": "CROSSED",
            "leverage": 4,
        },
        {
            "symbol": "ETHUSDT",
            "marginType": "CROSSED",
            "leverage": 7,
        },
    ]

    fact = normalize_positions(
        [payload],
        account_id="acct",
        symbol_configurations=configurations,
    )[0]

    assert fact.symbol == "ETHUSDT"
    assert fact.margin_mode == "cross"
    assert fact.leverage == 7


def test_zero_delivery_position_row_is_ignored_but_open_exposure_fails_closed():
    dormant = {
        "symbol": "ETHUSDT_261225",
        "positionSide": "LONG",
        "positionAmt": "0",
    }
    assert normalize_positions([dormant], account_id="acct") == ()

    active = dict(dormant, positionAmt="1")
    with pytest.raises(BinanceDataError, match="non-perpetual position exposure"):
        normalize_positions([active], account_id="acct")

def test_observed_time_rejects_naive_datetime():
    with pytest.raises(BinanceDataError, match="explicit timezone"):
        observed_time(datetime(2026, 7, 26, 12, 0, 0))  # noqa: DTZ001


def test_stable_fingerprint_is_deterministic_for_sets():
    left = {"values": {"beta", "alpha", "gamma"}}
    right = {"values": {"gamma", "beta", "alpha"}}

    assert to_primitive(left) == {"values": ["alpha", "beta", "gamma"]}
    assert stable_fingerprint(left) == stable_fingerprint(right)


def test_stable_fingerprint_preserves_sequence_order():
    assert stable_fingerprint(["alpha", "beta"]) != stable_fingerprint(
        ["beta", "alpha"]
    )
