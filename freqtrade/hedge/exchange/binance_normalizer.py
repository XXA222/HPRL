from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .base import (
    AccountConfigurationFact,
    AccountSnapshotFact,
    BalanceFact,
    FillFact,
    OrderFact,
    OrderOrigin,
    PositionFact,
    utc_now,
)
from .rate_limit import BinanceDataError, BinancePermissionError
from .symbol_codec import to_binance_symbol


def finite_decimal(value: Any, *, field: str, default: str | None = None) -> Decimal:
    if value in (None, "") and default is not None:
        value = default
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BinanceDataError(f"{field} is not a decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise BinanceDataError(f"{field} must be finite: {value!r}")
    return parsed


def nonnegative_decimal(value: Any, *, field: str, default: str = "0") -> Decimal:
    parsed = finite_decimal(value, field=field, default=default)
    if parsed < 0:
        raise BinanceDataError(f"{field} must be non-negative: {value!r}")
    return parsed


def int_value(value: Any, *, field: str, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BinanceDataError(f"{field} is not an integer: {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise BinanceDataError(f"{field} is not an integer: {value!r}")
    return int(parsed)


def nonnegative_int(value: Any, *, field: str, default: int = 0) -> int:
    parsed = int_value(value, field=field, default=default)
    if parsed < 0:
        raise BinanceDataError(f"{field} must be nonnegative: {value!r}")
    return parsed


def strict_bool(value: Any, *, field: str, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise BinanceDataError(f"{field} is not a boolean: {value!r}")


def nonempty_enum(value: Any, *, field: str, allowed: set[str] | None = None) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise BinanceDataError(f"{field} is required")
    if allowed is not None and normalized not in allowed:
        raise BinanceDataError(f"Invalid {field}: {value!r}")
    return normalized


def normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        raise BinanceDataError("symbol is required")
    try:
        return to_binance_symbol(symbol)
    except ValueError as exc:
        raise BinanceDataError(f"Invalid symbol: {value!r}") from exc


def normalize_position_side(value: Any, *, allow_both: bool = False) -> str:
    side = str(value or "").strip().upper()
    allowed = {"LONG", "SHORT"}
    if allow_both:
        allowed.add("BOTH")
    if side not in allowed:
        raise BinanceDataError(f"Invalid position side: {value!r}")
    return side


def observed_time(value: datetime | None = None) -> datetime:
    result = value or utc_now()
    if result.tzinfo is None or result.utcoffset() is None:
        raise BinanceDataError("observed_at must include an explicit timezone")
    return result.astimezone(UTC)


def _normalize_margin_mode(payload: Mapping[str, Any], *, symbol: str) -> str:
    raw_margin_type = payload.get("marginType", payload.get("mt"))
    if raw_margin_type in (None, "") and "isolated" not in payload:
        raise BinanceDataError(f"Margin mode is required for {symbol}")
    if raw_margin_type in (None, ""):
        is_isolated = strict_bool(payload.get("isolated"), field=f"{symbol}.isolated")
        return "isolated" if is_isolated else "cross"
    margin_mode = str(raw_margin_type).strip().lower()
    if margin_mode in {"crossed", "cross"}:
        return "cross"
    if margin_mode == "isolated":
        return margin_mode
    raise BinanceDataError(f"Invalid margin mode for {symbol}: {margin_mode!r}")


def _optional_liquidation_price(payload: Mapping[str, Any], *, symbol: str) -> Decimal | None:
    raw_value = payload.get("liquidationPrice")
    if raw_value in (None, "", "0", 0):
        return None
    return nonnegative_decimal(raw_value, field=f"{symbol}.liquidationPrice")


def _normalize_fill_side(payload: Mapping[str, Any], *, symbol: str) -> str:
    side = payload.get("side") or payload.get("S")
    if not side and "buyer" in payload:
        is_buyer = strict_bool(payload.get("buyer"), field=f"{symbol}.buyer")
        side = "BUY" if is_buyer else "SELL"
    return nonempty_enum(side, field=f"{symbol}.fillSide", allowed={"BUY", "SELL"})


def _normalize_symbol_configuration_row(
    item: Mapping[str, Any],
    *,
    expected: set[str],
) -> tuple[str, str, int] | None:
    # /fapi/v1/symbolConfig is account-wide when no symbol filter is supplied and
    # may include valid USD-M delivery contracts such as ETHUSDT_261225.  This
    # runtime manages perpetual symbols only, so filter unrelated exchange symbols
    # before applying the strict perpetual/canonical symbol codec.
    raw_symbol = str(item.get("symbol") or "").strip().upper()
    if raw_symbol not in expected:
        return None
    symbol = normalize_symbol(raw_symbol)
    raw_margin = str(item.get("marginType") or "").strip().lower()
    if raw_margin in {"cross", "crossed"}:
        margin_mode = "cross"
    elif raw_margin == "isolated":
        margin_mode = "isolated"
    else:
        raise BinanceDataError(
            f"Invalid symbol configuration margin type for {symbol}: {raw_margin!r}"
        )
    leverage = nonnegative_int(item.get("leverage"), field=f"{symbol}.leverage")
    if leverage <= 0:
        raise BinanceDataError(f"{symbol}.leverage must be positive")
    return symbol, margin_mode, leverage


def _validate_symbol_configuration_rows(
    rows: Mapping[str, tuple[str, int, Mapping[str, Any]]],
    *,
    expected: set[str],
) -> None:
    missing = sorted(expected.difference(rows))
    if missing:
        raise BinancePermissionError(
            "Missing managed symbol configuration rows: " + ",".join(missing)
        )
    non_cross = sorted(
        symbol
        for symbol, (margin_mode, _leverage, _raw) in rows.items()
        if margin_mode != "cross"
    )
    if non_cross:
        raise BinancePermissionError(
            "Non-cross managed symbol configuration: " + ",".join(non_cross)
        )


def _leverage_by_symbol_side(
    rows: Mapping[str, tuple[str, int, Mapping[str, Any]]],
    *,
    expected: set[str],
) -> dict[str, int]:
    return {
        f"{symbol}:{side}": rows[symbol][1]
        for symbol in sorted(expected)
        for side in ("LONG", "SHORT")
    }


def normalize_account_snapshot(
    payload: Mapping[str, Any],
    *,
    account_id: str,
    collection_started_at: datetime,
    collection_completed_at: datetime,
) -> tuple[AccountSnapshotFact, tuple[BalanceFact, ...]]:
    started_at = observed_time(collection_started_at)
    observed_at = observed_time(collection_completed_at)
    if observed_at < started_at:
        raise BinanceDataError("account snapshot completed before it started")
    snapshot = AccountSnapshotFact(
        account_id=account_id,
        total_wallet_balance=finite_decimal(
            payload.get("totalWalletBalance"), field="totalWalletBalance", default="0"
        ),
        total_available_balance=finite_decimal(
            payload.get("availableBalance"), field="availableBalance", default="0"
        ),
        total_margin_balance=finite_decimal(
            payload.get("totalMarginBalance"), field="totalMarginBalance", default="0"
        ),
        total_initial_margin=finite_decimal(
            payload.get("totalInitialMargin"), field="totalInitialMargin", default="0"
        ),
        total_maintenance_margin=finite_decimal(
            payload.get("totalMaintMargin"), field="totalMaintMargin", default="0"
        ),
        total_unrealized_pnl=finite_decimal(
            payload.get("totalUnrealizedProfit"), field="totalUnrealizedProfit", default="0"
        ),
        observed_at=observed_at,
        collection_started_at=started_at,
        collection_completed_at=observed_at,
        raw=payload,
    )
    assets = payload.get("assets", ())
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise BinanceDataError("account.assets must be a sequence")
    balances: list[BalanceFact] = []
    seen_assets: set[str] = set()
    for item in assets:
        if not isinstance(item, Mapping):
            raise BinanceDataError("account.assets contains a non-object item")
        asset = str(item.get("asset") or "").strip().upper()
        if not asset:
            raise BinanceDataError("asset name is required")
        if asset in seen_assets:
            raise BinanceDataError(f"duplicate account asset row: {asset}")
        seen_assets.add(asset)
        balances.append(
            BalanceFact(
                account_id=account_id,
                asset=asset,
                wallet_balance=finite_decimal(
                    item.get("walletBalance"), field=f"{asset}.walletBalance", default="0"
                ),
                available_balance=finite_decimal(
                    item.get("availableBalance"),
                    field=f"{asset}.availableBalance",
                    default="0",
                ),
                cross_wallet_balance=finite_decimal(
                    item.get("crossWalletBalance"),
                    field=f"{asset}.crossWalletBalance",
                    default="0",
                ),
                unrealized_pnl=finite_decimal(
                    item.get("unrealizedProfit"),
                    field=f"{asset}.unrealizedProfit",
                    default="0",
                ),
                observed_at=observed_at,
                source="BINANCE_REST",
                raw=item,
            )
        )
    return snapshot, tuple(balances)



def normalize_account_update_balance(
    payload: Mapping[str, Any],
    *,
    account_id: str,
    previous: BalanceFact | None = None,
    observed_at: datetime | None = None,
) -> BalanceFact:
    asset = str(payload.get("a") or payload.get("asset") or "").strip().upper()
    if not asset:
        raise BinanceDataError("ACCOUNT_UPDATE balance asset is required")
    wallet_balance = finite_decimal(
        payload.get("wb", payload.get("walletBalance")),
        field=f"{asset}.walletBalance",
    )
    cross_wallet_balance = finite_decimal(
        payload.get("cw", payload.get("crossWalletBalance", wallet_balance)),
        field=f"{asset}.crossWalletBalance",
    )
    available_balance = (
        previous.available_balance if previous is not None else cross_wallet_balance
    )
    unrealized_pnl = (
        previous.unrealized_pnl if previous is not None else Decimal(0)
    )
    return BalanceFact(
        account_id=account_id,
        asset=asset,
        wallet_balance=wallet_balance,
        available_balance=available_balance,
        cross_wallet_balance=cross_wallet_balance,
        unrealized_pnl=unrealized_pnl,
        observed_at=observed_time(observed_at),
        source="BINANCE_USER_STREAM",
        raw=payload,
    )


def normalize_account_update_balances(
    payload: Sequence[Mapping[str, Any]],
    *,
    account_id: str,
    previous_by_asset: Mapping[str, BalanceFact] | None = None,
    observed_at: datetime | None = None,
) -> tuple[BalanceFact, ...]:
    previous_rows = previous_by_asset or {}
    result: list[BalanceFact] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise BinanceDataError(
                f"ACCOUNT_UPDATE balance row {index} must be an object"
            )
        asset = str(item.get("a") or item.get("asset") or "").strip().upper()
        fact = normalize_account_update_balance(
            item,
            account_id=account_id,
            previous=previous_rows.get(asset),
            observed_at=observed_at,
        )
        if fact.asset in seen:
            raise BinanceDataError(
                f"ACCOUNT_UPDATE contains duplicate balance asset: {fact.asset}"
            )
        seen.add(fact.asset)
        result.append(fact)
    return tuple(result)

def normalize_position(
    payload: Mapping[str, Any],
    *,
    account_id: str,
    observed_at: datetime | None = None,
    source: str = "BINANCE_REST",
    margin_mode_override: str | None = None,
    leverage_override: int | None = None,
) -> PositionFact:
    symbol = normalize_symbol(payload.get("symbol") or payload.get("s"))
    side = normalize_position_side(payload.get("positionSide") or payload.get("ps"))
    raw_amount = finite_decimal(
        payload.get("positionAmt", payload.get("pa", "0")), field=f"{symbol}.positionAmt"
    )
    if side == "LONG" and raw_amount < 0:
        raise BinanceDataError(f"LONG position has negative signed amount for {symbol}")
    if side == "SHORT" and raw_amount > 0:
        raise BinanceDataError(f"SHORT position has positive signed amount for {symbol}")
    quantity = abs(raw_amount)
    margin_mode = (
        margin_mode_override
        if margin_mode_override is not None
        else _normalize_margin_mode(payload, symbol=symbol)
    )
    leverage_value = (
        leverage_override
        if leverage_override is not None
        else payload.get("leverage")
    )
    liquidation = _optional_liquidation_price(payload, symbol=symbol)
    return PositionFact(
        account_id=account_id,
        symbol=symbol,
        position_side=side,
        quantity=quantity,
        entry_price=nonnegative_decimal(
            payload.get("entryPrice", payload.get("ep", "0")), field=f"{symbol}.entryPrice"
        ),
        mark_price=nonnegative_decimal(
            payload.get("markPrice", payload.get("mp", "0")), field=f"{symbol}.markPrice"
        ),
        unrealized_pnl=finite_decimal(
            payload.get("unRealizedProfit", payload.get("up", "0")),
            field=f"{symbol}.unRealizedProfit",
        ),
        liquidation_price=liquidation,
        leverage=nonnegative_int(
            leverage_value, field=f"{symbol}.leverage", default=0
        ),
        margin_mode=margin_mode,
        update_time_ms=nonnegative_int(
            payload.get("updateTime", payload.get("T", payload.get("E", 0))),
            field=f"{symbol}.updateTime",
        ),
        observed_at=observed_time(observed_at),
        source=source,
        raw=payload,
    )


def normalize_positions(
    payload: Sequence[Mapping[str, Any]],
    *,
    account_id: str,
    observed_at: datetime | None = None,
    source: str = "BINANCE_REST",
    symbol_configurations: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[PositionFact, ...]:
    normalized_rows: list[tuple[Mapping[str, Any], str]] = []
    expected: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise BinanceDataError(f"positionRisk[{index}] must be an object")
        raw_symbol = str(item.get("symbol") or item.get("s") or "").strip().upper()
        try:
            symbol = normalize_symbol(raw_symbol)
        except BinanceDataError as exc:
            # V3 can expose account-wide contract rows.  Dormant delivery-contract
            # rows are outside the perpetual runtime universe and are harmless, but
            # an actual position in an unsupported contract must halt acceptance.
            raw_amount = finite_decimal(
                item.get("positionAmt", item.get("pa", "0")),
                field=f"{raw_symbol or 'UNKNOWN'}.positionAmt",
            )
            if raw_amount != 0:
                raise BinanceDataError(
                    "Unsupported non-perpetual position exposure: "
                    f"{raw_symbol or '<missing>'}"
                ) from exc
            continue
        normalized_rows.append((item, symbol))
        expected.add(symbol)

    configuration_by_symbol: dict[str, tuple[str, int]] = {}
    if symbol_configurations is not None:
        for index, item in enumerate(symbol_configurations):
            if not isinstance(item, Mapping):
                raise BinanceDataError(f"symbolConfig[{index}] must be an object")
            normalized = _normalize_symbol_configuration_row(item, expected=expected)
            if normalized is None:
                continue
            symbol, margin_mode, leverage = normalized
            if symbol in configuration_by_symbol:
                raise BinanceDataError(f"duplicate symbol configuration row: {symbol}")
            configuration_by_symbol[symbol] = (margin_mode, leverage)

    result: list[PositionFact] = []
    seen: set[tuple[str, str]] = set()
    for item, symbol in normalized_rows:
        fallback = configuration_by_symbol.get(symbol)
        margin_source_missing = (
            item.get("marginType", item.get("mt")) in (None, "")
            and "isolated" not in item
        )
        leverage_source_missing = item.get("leverage") in (None, "", 0, "0")
        fact = normalize_position(
            item,
            account_id=account_id,
            observed_at=observed_at,
            source=source,
            margin_mode_override=(
                fallback[0]
                if fallback is not None and margin_source_missing
                else None
            ),
            leverage_override=(
                fallback[1]
                if fallback is not None and leverage_source_missing
                else None
            ),
        )
        key = (fact.symbol, fact.position_side)
        if key in seen:
            raise BinanceDataError(
                f"duplicate position row: {fact.symbol}:{fact.position_side}"
            )
        seen.add(key)
        result.append(fact)
    return tuple(result)


def _order_origin(
    client_order_id: str,
    *,
    system_client_order_prefixes: Sequence[str],
) -> tuple[OrderOrigin, bool]:
    normalized_prefixes = tuple(
        prefix.strip() for prefix in system_client_order_prefixes if prefix.strip()
    )
    if any(client_order_id.startswith(prefix) for prefix in normalized_prefixes):
        return OrderOrigin.SYSTEM, False
    if client_order_id:
        return OrderOrigin.EXTERNAL, True
    return OrderOrigin.UNKNOWN, True


def normalize_order(
    payload: Mapping[str, Any],
    *,
    account_id: str,
    observed_at: datetime | None = None,
    source: str = "BINANCE_REST",
    system_client_order_prefixes: Sequence[str] = (),
) -> OrderFact:
    symbol = normalize_symbol(payload.get("symbol") or payload.get("s"))
    order_id = str(payload.get("orderId", payload.get("i", ""))).strip()
    if not order_id:
        raise BinanceDataError(f"exchange order id is required for {symbol}")
    original = nonnegative_decimal(
        payload.get("origQty", payload.get("q", "0")), field=f"{symbol}.origQty"
    )
    cumulative = nonnegative_decimal(
        payload.get("executedQty", payload.get("z", "0")), field=f"{symbol}.executedQty"
    )
    if cumulative > original:
        raise BinanceDataError(f"executed quantity exceeds original quantity for {order_id}")
    client_order_id = str(
        payload.get("clientOrderId", payload.get("c", ""))
    ).strip()
    origin, quarantined = _order_origin(
        client_order_id,
        system_client_order_prefixes=system_client_order_prefixes,
    )
    return OrderFact(
        account_id=account_id,
        symbol=symbol,
        position_side=normalize_position_side(
            payload.get("positionSide") or payload.get("ps")
        ),
        exchange_order_id=order_id,
        client_order_id=client_order_id,
        side=nonempty_enum(
            payload.get("side", payload.get("S")),
            field=f"{symbol}.side",
            allowed={"BUY", "SELL"},
        ),
        order_type=nonempty_enum(
            payload.get("type", payload.get("o")), field=f"{symbol}.orderType"
        ),
        status=nonempty_enum(
            payload.get("status", payload.get("X")), field=f"{symbol}.orderStatus"
        ),
        original_quantity=original,
        cumulative_filled_quantity=cumulative,
        average_price=nonnegative_decimal(
            payload.get("avgPrice", payload.get("ap", "0")), field=f"{symbol}.avgPrice"
        ),
        reduce_only=strict_bool(
            payload.get("reduceOnly", payload.get("R", False)),
            field=f"{symbol}.reduceOnly",
        ),
        update_time_ms=nonnegative_int(
            payload.get("updateTime", payload.get("T", payload.get("E", 0))),
            field=f"{symbol}.orderUpdateTime",
        ),
        observed_at=observed_time(observed_at),
        source=source,
        origin=origin,
        quarantined=quarantined,
        raw=payload,
    )


def normalize_orders(
    payload: Sequence[Mapping[str, Any]],
    *,
    account_id: str,
    observed_at: datetime | None = None,
    source: str = "BINANCE_REST",
    system_client_order_prefixes: Sequence[str] = (),
) -> tuple[OrderFact, ...]:
    result: list[OrderFact] = []
    seen: set[tuple[str, str]] = set()
    for item in payload:
        fact = normalize_order(
            item,
            account_id=account_id,
            observed_at=observed_at,
            source=source,
            system_client_order_prefixes=system_client_order_prefixes,
        )
        key = (fact.symbol, fact.exchange_order_id)
        if key in seen:
            raise BinanceDataError(
                f"duplicate order row: {fact.symbol}:{fact.exchange_order_id}"
            )
        seen.add(key)
        result.append(fact)
    return tuple(result)


def normalize_fill(
    payload: Mapping[str, Any],
    *,
    account_id: str,
    observed_at: datetime | None = None,
    source: str = "BINANCE_REST",
) -> FillFact:
    symbol = normalize_symbol(payload.get("symbol") or payload.get("s"))
    trade_id = str(payload.get("id", payload.get("t", ""))).strip()
    order_id = str(payload.get("orderId", payload.get("i", ""))).strip()
    if not trade_id or trade_id in {"0", "-1"}:
        raise BinanceDataError(f"exchange trade id is required for {symbol}")
    if not order_id:
        raise BinanceDataError(f"exchange order id is required for trade {trade_id}")
    side = _normalize_fill_side(payload, symbol=symbol)
    fact = FillFact(
        account_id=account_id,
        symbol=symbol,
        position_side=normalize_position_side(
            payload.get("positionSide") or payload.get("ps")
        ),
        exchange_trade_id=trade_id,
        exchange_order_id=order_id,
        side=side,
        quantity=nonnegative_decimal(
            payload.get("qty", payload.get("l", "0")), field=f"{symbol}.fillQty"
        ),
        price=nonnegative_decimal(
            payload.get("price", payload.get("L", "0")), field=f"{symbol}.fillPrice"
        ),
        commission=nonnegative_decimal(
            payload.get("commission", payload.get("n", "0")),
            field=f"{symbol}.commission",
        ),
        commission_asset=(
            str(payload.get("commissionAsset", payload.get("N"))).strip().upper()
            if payload.get("commissionAsset", payload.get("N"))
            else None
        ),
        realized_pnl=finite_decimal(
            payload.get("realizedPnl", payload.get("rp", "0")),
            field=f"{symbol}.realizedPnl",
        ),
        event_time_ms=nonnegative_int(
            payload.get("time", payload.get("T", payload.get("E", 0))),
            field=f"{symbol}.fillTime",
        ),
        observed_at=observed_time(observed_at),
        source=source,
        raw=payload,
    )
    if fact.quantity <= 0:
        raise BinanceDataError(f"{symbol}.fillQty must be positive")
    if fact.price <= 0:
        raise BinanceDataError(f"{symbol}.fillPrice must be positive")
    return fact


def normalize_fills(
    payload: Sequence[Mapping[str, Any]],
    *,
    account_id: str,
    observed_at: datetime | None = None,
    source: str = "BINANCE_REST",
) -> tuple[FillFact, ...]:
    return tuple(
        normalize_fill(item, account_id=account_id, observed_at=observed_at, source=source)
        for item in payload
    )


def normalize_configuration(
    *,
    account_id: str,
    dual_side_payload: Mapping[str, Any],
    symbol_configurations: Sequence[Mapping[str, Any]],
    managed_symbols: Sequence[str],
    observed_at: datetime | None = None,
) -> AccountConfigurationFact:
    hedge_mode = strict_bool(
        dual_side_payload.get("dualSidePosition"), field="dualSidePosition"
    )
    if isinstance(managed_symbols, (str, bytes)):
        raise BinanceDataError("managed_symbols must be a sequence")
    expected = {normalize_symbol(symbol) for symbol in managed_symbols}
    if not expected:
        raise BinanceDataError("managed_symbols must not be empty")

    rows: dict[str, tuple[str, int, Mapping[str, Any]]] = {}
    for index, item in enumerate(symbol_configurations):
        if not isinstance(item, Mapping):
            raise BinanceDataError(f"symbolConfig[{index}] must be an object")
        normalized = _normalize_symbol_configuration_row(item, expected=expected)
        if normalized is None:
            continue
        symbol, margin_mode, leverage = normalized
        if symbol in rows:
            raise BinanceDataError(f"duplicate symbol configuration row: {symbol}")
        rows[symbol] = (margin_mode, leverage, item)

    _validate_symbol_configuration_rows(rows, expected=expected)
    leverage_by_side = _leverage_by_symbol_side(rows, expected=expected)
    margin_modes = tuple(sorted({rows[symbol][0] for symbol in expected}))
    return AccountConfigurationFact(
        account_id=account_id,
        hedge_mode=hedge_mode,
        active_margin_modes=margin_modes,
        leverage_by_symbol_side=leverage_by_side,
        observed_at=observed_time(observed_at),
        raw={
            "position_mode": dict(dual_side_payload),
            "symbol_configurations": tuple(rows[symbol][2] for symbol in sorted(rows)),
        },
    )
