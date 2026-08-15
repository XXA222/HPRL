"""Explicit Hedge exchange capability contract.

Freqtrade's ExchangeResolver can create many exchange clients, but that alone does not
make their position-side, order-id and user-stream semantics safe for Hedge.  This
registry prevents accidental capability inference and keeps Binance USD-M as the only
write-ready adapter until another adapter supplies equivalent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class CapabilityLevel(StrEnum):
    UNSUPPORTED = "UNSUPPORTED"
    PUBLIC_DATA = "PUBLIC_DATA"
    READ_ONLY = "READ_ONLY"
    PAPER = "PAPER"
    TESTNET_WRITE = "TESTNET_WRITE"
    MAINNET_WRITE = "MAINNET_WRITE"


@dataclass(frozen=True, slots=True)
class HedgeExchangeCapabilities:
    exchange: str
    market: str
    position_mode: str
    margin_mode: str
    level: CapabilityLevel
    simultaneous_long_short: bool
    independent_position_side_orders: bool
    reduce_only: bool
    client_order_id: bool
    user_stream: bool
    rest_reconciliation: bool
    funding: bool
    liquidation_price: bool
    cross_wallet: bool
    unknown_order_recovery: bool
    max_client_order_id_length: int = 0
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange", self.exchange.lower().strip())
        object.__setattr__(self, "market", self.market.lower().strip())
        object.__setattr__(self, "position_mode", self.position_mode.upper().strip())
        object.__setattr__(self, "margin_mode", self.margin_mode.upper().strip())
        object.__setattr__(self, "level", CapabilityLevel(self.level))
        object.__setattr__(self, "evidence", dict(self.evidence))
        if self.max_client_order_id_length < 0:
            raise ValueError("max_client_order_id_length cannot be negative")

    @property
    def safe_for_live_write(self) -> bool:
        required = (
            self.simultaneous_long_short,
            self.independent_position_side_orders,
            self.reduce_only,
            self.client_order_id,
            self.user_stream,
            self.rest_reconciliation,
            self.funding,
            self.cross_wallet,
            self.unknown_order_recovery,
        )
        return self.level in {CapabilityLevel.TESTNET_WRITE, CapabilityLevel.MAINNET_WRITE} and all(required)

    def missing_for_live(self) -> tuple[str, ...]:
        fields = (
            "simultaneous_long_short", "independent_position_side_orders", "reduce_only",
            "client_order_id", "user_stream", "rest_reconciliation", "funding",
            "cross_wallet", "unknown_order_recovery",
        )
        return tuple(name for name in fields if not getattr(self, name))


class HedgeExchangeCapabilityRegistry:
    def __init__(self, entries: tuple[HedgeExchangeCapabilities, ...] = ()) -> None:
        self._entries: dict[tuple[str, str], HedgeExchangeCapabilities] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: HedgeExchangeCapabilities, *, replace: bool = False) -> None:
        key = (entry.exchange, entry.market)
        if key in self._entries and not replace:
            raise ValueError(f"capability entry already exists: {key}")
        self._entries[key] = entry

    def get(self, exchange: str, market: str) -> HedgeExchangeCapabilities:
        key = (exchange.lower().strip(), market.lower().strip())
        try:
            return self._entries[key]
        except KeyError as exc:
            raise LookupError(f"Hedge exchange adapter not registered: {key}") from exc

    def require(
        self,
        exchange: str,
        market: str,
        *,
        minimum: CapabilityLevel,
    ) -> HedgeExchangeCapabilities:
        entry = self.get(exchange, market)
        order = list(CapabilityLevel)
        if order.index(entry.level) < order.index(CapabilityLevel(minimum)):
            raise PermissionError(
                f"{entry.exchange}/{entry.market} capability {entry.level.value} is below {minimum.value}"
            )
        if minimum in {CapabilityLevel.TESTNET_WRITE, CapabilityLevel.MAINNET_WRITE} and not entry.safe_for_live_write:
            raise PermissionError(
                "Hedge write capability evidence incomplete: " + ",".join(entry.missing_for_live())
            )
        return entry

    def snapshot(self) -> dict[str, Any]:
        return {
            f"{exchange}:{market}": {
                "level": entry.level.value,
                "safe_for_live_write": entry.safe_for_live_write,
                "missing_for_live": list(entry.missing_for_live()),
                "position_mode": entry.position_mode,
                "margin_mode": entry.margin_mode,
                "evidence": dict(entry.evidence),
            }
            for (exchange, market), entry in sorted(self._entries.items())
        }


def default_exchange_registry() -> HedgeExchangeCapabilityRegistry:
    return HedgeExchangeCapabilityRegistry(
        (
            HedgeExchangeCapabilities(
                exchange="binance",
                market="usdm",
                position_mode="HEDGE",
                margin_mode="CROSS",
                level=CapabilityLevel.TESTNET_WRITE,
                simultaneous_long_short=True,
                independent_position_side_orders=True,
                reduce_only=True,
                client_order_id=True,
                user_stream=True,
                rest_reconciliation=True,
                funding=True,
                liquidation_price=True,
                cross_wallet=True,
                unknown_order_recovery=True,
                max_client_order_id_length=36,
                evidence={
                    "adapter": "freqtrade.hedge.exchange.binance_*",
                    "mainnet_write": "locked_pending_promotion",
                },
            ),
            # Other official Freqtrade exchanges are intentionally data-only until a
            # Hedge adapter proves position-side, reconciliation and recovery semantics.
            *tuple(
                HedgeExchangeCapabilities(
                    exchange=name,
                    market="swap",
                    position_mode="UNKNOWN",
                    margin_mode="UNKNOWN",
                    level=CapabilityLevel.PUBLIC_DATA,
                    simultaneous_long_short=False,
                    independent_position_side_orders=False,
                    reduce_only=False,
                    client_order_id=False,
                    user_stream=False,
                    rest_reconciliation=False,
                    funding=False,
                    liquidation_price=False,
                    cross_wallet=False,
                    unknown_order_recovery=False,
                )
                for name in ("bybit", "okx", "gate", "bitget")
            ),
        )
    )
