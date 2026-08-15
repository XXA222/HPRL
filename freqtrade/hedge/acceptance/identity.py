from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from freqtrade.hedge.exchange.base import AccountConfigurationFact, PositionFact
from freqtrade.hedge.acceptance.models import LegIdentity


VALID_SIDES = ("LONG", "SHORT")


def build_leg_identities(
    *,
    account_id: str,
    managed_symbols: Sequence[str],
    positions: Sequence[PositionFact],
    configuration: AccountConfigurationFact,
) -> tuple[LegIdentity, ...]:
    if not configuration.hedge_mode:
        raise ValueError("account is not in Hedge Mode")
    by_key: dict[tuple[str, str], PositionFact] = {}
    for position in positions:
        side = position.position_side.upper()
        if side not in VALID_SIDES:
            raise ValueError(f"invalid hedge position side: {side}")
        key = (position.symbol.upper(), side)
        previous = by_key.get(key)
        if previous is not None and previous != position:
            raise ValueError(f"conflicting position identity: {key}")
        by_key[key] = position

    identities: list[LegIdentity] = []
    for symbol in sorted({str(value).strip().upper() for value in managed_symbols}):
        for side in VALID_SIDES:
            config_key = f"{symbol}:{side}"
            if config_key not in configuration.leverage_by_symbol_side:
                raise ValueError(f"missing leverage identity: {config_key}")
            position = by_key.get((symbol, side))
            identities.append(
                LegIdentity(
                    account_id=account_id,
                    symbol=symbol,
                    position_side=side,
                    present_in_rest=position is not None,
                    quantity=Decimal(0) if position is None else position.quantity,
                    leverage=int(configuration.leverage_by_symbol_side[config_key]),
                    margin_mode=(
                        "cross" if configuration.active_margin_modes == ("cross",) else "mixed"
                    ),
                )
            )
    return tuple(identities)


def identity_mismatch_count(
    identities: Sequence[LegIdentity], *, managed_symbols: Sequence[str], account_id: str
) -> int:
    expected = {
        f"{account_id}:{str(symbol).strip().upper()}:{side}"
        for symbol in managed_symbols
        for side in VALID_SIDES
    }
    observed = {item.key for item in identities}
    return len(expected.symmetric_difference(observed))


def leverage_mismatches(
    leverage_by_side: Mapping[str, int], *, target_leverage: int | None
) -> tuple[str, ...]:
    if target_leverage is None:
        return ()
    return tuple(
        sorted(
            key
            for key, value in leverage_by_side.items()
            if int(value) != int(target_leverage)
        )
    )
