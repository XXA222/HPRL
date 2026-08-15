"""Strongly typed Paper/simulation configuration.

The runtime owns one canonical ``hedge.paper`` section.  The historical
``hedge.simulation`` section is accepted only as a compatibility input and is
merged deterministically during configuration normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import logging
from typing import Any, Mapping, MutableMapping

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class PaperOhlcvSource(StrEnum):
    DATAPROVIDER = "dataprovider"
    TICKER_COMPAT = "ticker_compat"


class PaperFundingSource(StrEnum):
    EXCHANGE = "exchange"
    NONE = "none"


def _decimal(
    value: object,
    *,
    field: str,
    default: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    exclusive_minimum: bool = False,
) -> Decimal:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise OperationalException(f"{field} must be an exact decimal value")
    try:
        result = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise OperationalException(f"{field} must be a valid decimal value") from exc
    if not result.is_finite():
        raise OperationalException(f"{field} must be finite")
    if minimum is not None:
        invalid = result <= minimum if exclusive_minimum else result < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            raise OperationalException(f"{field} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise OperationalException(f"{field} must be <= {maximum}")
    return result


def _integer(
    value: object,
    *,
    field: str,
    default: int,
    minimum: int = 0,
) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise OperationalException(f"{field} must be an integer")
    if raw < minimum:
        raise OperationalException(f"{field} must be >= {minimum}")
    return raw


def _boolean(value: object, *, field: str, default: bool) -> bool:
    raw = default if value is None else value
    if not isinstance(raw, bool):
        raise OperationalException(f"{field} must be a boolean")
    return raw


def _choice(value: object, *, field: str, default: str, choices: set[str]) -> str:
    raw = default if value is None else value
    if not isinstance(raw, str) or not raw.strip():
        raise OperationalException(f"{field} must be a non-empty string")
    normalized = raw.strip().lower()
    if normalized not in choices:
        raise OperationalException(
            f"{field} must be one of: {', '.join(sorted(choices))}"
        )
    return normalized


def _legacy_values_equal(left: object, right: object, *, integer: bool = False) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if integer:
        return isinstance(left, int) and isinstance(right, int) and left == right
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return str(left) == str(right)


def _legacy_merge(
    paper: MutableMapping[str, Any],
    simulation: Mapping[str, Any],
) -> None:
    aliases = {
        "maker_fee": "maker_fee_rate",
        "taker_fee": "taker_fee_rate",
        "partial_fill_ratio": "max_fill_ratio_per_order",
        "max_fills_per_bar": "max_fills_per_bar",
        "volume_participation": "volume_participation",
        "market_slippage_bps": "market_slippage_bps",
        "bar_volume": "bar_volume",
    }
    for legacy_key, canonical_key in aliases.items():
        if legacy_key not in simulation:
            continue
        legacy_value = simulation[legacy_key]
        if canonical_key in paper and not _legacy_values_equal(
            paper[canonical_key],
            legacy_value,
            integer=canonical_key == "max_fills_per_bar",
        ):
            raise OperationalException(
                f"hedge.simulation.{legacy_key} conflicts with "
                f"hedge.paper.{canonical_key}; keep only the canonical paper value"
            )
        paper.setdefault(canonical_key, legacy_value)


@dataclass(frozen=True, slots=True)
class PaperSimulationConfig:
    initial_balance: Decimal
    leverage: Decimal
    auto_fill: bool
    fill_model: str
    ephemeral: bool
    state_backend: str
    state_path: str | None
    ohlcv_source: PaperOhlcvSource
    require_closed_candle: bool
    candle_max_age_seconds: int
    max_catchup_candles: int
    max_missing_candles: int
    reject_revised_candle: bool
    funding_source: PaperFundingSource
    funding_max_age_seconds: int
    funding_poll_interval_seconds: int
    account_events_enabled: bool
    idempotency_lease_seconds: int
    default_long_signal: Decimal
    default_short_signal: Decimal
    tick_size: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    volume_participation: Decimal
    market_slippage_bps: Decimal
    max_entry_layers_per_bar: int
    max_reduce_layers_per_bar: int
    max_fill_ratio_per_order: Decimal
    max_fills_per_bar: int
    bar_volume: Decimal | None

    @classmethod
    def from_hedge_mapping(
        cls,
        hedge: MutableMapping[str, Any],
    ) -> "PaperSimulationConfig":
        raw_paper = hedge.setdefault("paper", {})
        if not isinstance(raw_paper, MutableMapping):
            raise OperationalException("hedge.paper must be a JSON object")
        raw_simulation = hedge.get("simulation", {})
        if not isinstance(raw_simulation, Mapping):
            raise OperationalException("hedge.simulation must be a JSON object")
        if raw_simulation:
            _legacy_merge(raw_paper, raw_simulation)
            logger.warning(
                "hedge.simulation is deprecated; values were merged into hedge.paper"
            )

        ephemeral = _boolean(
            raw_paper.get("ephemeral"), field="hedge.paper.ephemeral", default=False
        )
        fill_model = _choice(
            raw_paper.get("fill_model"),
            field="hedge.paper.fill_model",
            default="conservative",
            choices={"conservative", "instant"},
        )
        ohlcv_source = PaperOhlcvSource(
            _choice(
                raw_paper.get("ohlcv_source"),
                field="hedge.paper.ohlcv_source",
                default=PaperOhlcvSource.DATAPROVIDER.value,
                choices={item.value for item in PaperOhlcvSource},
            )
        )
        if fill_model == "instant" and not ephemeral:
            raise OperationalException(
                "hedge.paper.fill_model='instant' is test-only and requires ephemeral=true"
            )
        if ohlcv_source is PaperOhlcvSource.TICKER_COMPAT and not ephemeral:
            raise OperationalException(
                "hedge.paper.ohlcv_source='ticker_compat' is test-only; production Paper "
                "must use DataProvider OHLCV"
            )

        funding_source = PaperFundingSource(
            _choice(
                raw_paper.get("funding_source"),
                field="hedge.paper.funding_source",
                default=PaperFundingSource.EXCHANGE.value,
                choices={item.value for item in PaperFundingSource},
            )
        )
        account_events_enabled = _boolean(
            raw_paper.get("account_events_enabled"),
            field="hedge.paper.account_events_enabled",
            default=True,
        )
        if funding_source is PaperFundingSource.NONE and not ephemeral:
            raise OperationalException(
                "hedge.paper.funding_source='none' is test-only and requires ephemeral=true"
            )
        if not account_events_enabled and not ephemeral:
            raise OperationalException(
                "hedge.paper.account_events_enabled=false is test-only and requires "
                "ephemeral=true"
            )

        state_backend = _choice(
            raw_paper.get("state_backend"),
            field="hedge.paper.state_backend",
            default="sql",
            choices={"sql", "json"},
        )
        state_path_raw = raw_paper.get("state_path")
        state_path = None
        if state_path_raw is not None:
            if not isinstance(state_path_raw, str) or not state_path_raw.strip():
                raise OperationalException("hedge.paper.state_path must be a non-empty string")
            state_path = state_path_raw.strip()

        fee_rate = raw_paper.get("fee_rate")
        maker = raw_paper.get("maker_fee_rate", fee_rate)
        taker = raw_paper.get("taker_fee_rate", fee_rate)
        bar_volume_raw = raw_paper.get("bar_volume")
        bar_volume = (
            None
            if bar_volume_raw is None
            else _decimal(
                bar_volume_raw,
                field="hedge.paper.bar_volume",
                default="0",
                minimum=Decimal("0"),
            )
        )

        result = cls(
            initial_balance=_decimal(
                raw_paper.get("initial_balance"),
                field="hedge.paper.initial_balance",
                default="1000",
                minimum=Decimal("0"),
                exclusive_minimum=True,
            ),
            leverage=_decimal(
                raw_paper.get("leverage", hedge.get("target_leverage")),
                field="hedge.paper.leverage",
                default="3",
                minimum=Decimal("1"),
            ),
            auto_fill=_boolean(
                raw_paper.get("auto_fill"), field="hedge.paper.auto_fill", default=True
            ),
            fill_model=fill_model,
            ephemeral=ephemeral,
            state_backend=state_backend,
            state_path=state_path,
            ohlcv_source=ohlcv_source,
            require_closed_candle=_boolean(
                raw_paper.get("require_closed_candle"),
                field="hedge.paper.require_closed_candle",
                default=True,
            ),
            candle_max_age_seconds=_integer(
                raw_paper.get("candle_max_age_seconds"),
                field="hedge.paper.candle_max_age_seconds",
                default=0,
                minimum=0,
            ),
            max_catchup_candles=_integer(
                raw_paper.get("max_catchup_candles"),
                field="hedge.paper.max_catchup_candles",
                default=288,
                minimum=1,
            ),
            max_missing_candles=_integer(
                raw_paper.get("max_missing_candles"),
                field="hedge.paper.max_missing_candles",
                default=0,
                minimum=0,
            ),
            reject_revised_candle=_boolean(
                raw_paper.get("reject_revised_candle"),
                field="hedge.paper.reject_revised_candle",
                default=True,
            ),
            funding_source=funding_source,
            funding_max_age_seconds=_integer(
                raw_paper.get("funding_max_age_seconds"),
                field="hedge.paper.funding_max_age_seconds",
                default=3600,
                minimum=1,
            ),
            funding_poll_interval_seconds=_integer(
                raw_paper.get("funding_poll_interval_seconds"),
                field="hedge.paper.funding_poll_interval_seconds",
                default=60,
                minimum=0,
            ),
            account_events_enabled=account_events_enabled,
            idempotency_lease_seconds=_integer(
                raw_paper.get("idempotency_lease_seconds"),
                field="hedge.paper.idempotency_lease_seconds",
                default=300,
                minimum=1,
            ),
            default_long_signal=_decimal(
                raw_paper.get("long_signal"),
                field="hedge.paper.long_signal",
                default="0",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
            default_short_signal=_decimal(
                raw_paper.get("short_signal"),
                field="hedge.paper.short_signal",
                default="0",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
            tick_size=_decimal(
                raw_paper.get("tick_size"),
                field="hedge.paper.tick_size",
                default="0.01",
                minimum=Decimal("0"),
                exclusive_minimum=True,
            ),
            qty_step=_decimal(
                raw_paper.get("qty_step"),
                field="hedge.paper.qty_step",
                default="0.001",
                minimum=Decimal("0"),
                exclusive_minimum=True,
            ),
            min_qty=_decimal(
                raw_paper.get("min_qty"),
                field="hedge.paper.min_qty",
                default="0",
                minimum=Decimal("0"),
            ),
            min_notional=_decimal(
                raw_paper.get("min_notional"),
                field="hedge.paper.min_notional",
                default="0",
                minimum=Decimal("0"),
            ),
            maker_fee_rate=_decimal(
                maker,
                field="hedge.paper.maker_fee_rate",
                default="0.0002",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
            taker_fee_rate=_decimal(
                taker,
                field="hedge.paper.taker_fee_rate",
                default="0.0004",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
            volume_participation=_decimal(
                raw_paper.get("volume_participation"),
                field="hedge.paper.volume_participation",
                default="0.10",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
                exclusive_minimum=True,
            ),
            market_slippage_bps=_decimal(
                raw_paper.get("market_slippage_bps"),
                field="hedge.paper.market_slippage_bps",
                default="0",
                minimum=Decimal("0"),
            ),
            max_entry_layers_per_bar=_integer(
                raw_paper.get("max_entry_layers_per_bar"),
                field="hedge.paper.max_entry_layers_per_bar",
                default=1,
            ),
            max_reduce_layers_per_bar=_integer(
                raw_paper.get("max_reduce_layers_per_bar"),
                field="hedge.paper.max_reduce_layers_per_bar",
                default=1,
            ),
            max_fill_ratio_per_order=_decimal(
                raw_paper.get("max_fill_ratio_per_order"),
                field="hedge.paper.max_fill_ratio_per_order",
                default="1",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
                exclusive_minimum=True,
            ),
            max_fills_per_bar=_integer(
                raw_paper.get("max_fills_per_bar"),
                field="hedge.paper.max_fills_per_bar",
                default=0,
            ),
            bar_volume=bar_volume,
        )
        if result.max_missing_candles >= result.max_catchup_candles:
            raise OperationalException(
                "hedge.paper.max_missing_candles must be lower than "
                "max_catchup_candles"
            )
        hedge["paper"] = result.to_mapping()
        hedge.pop("simulation", None)
        return result

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "initial_balance": str(self.initial_balance),
            "leverage": str(self.leverage),
            "auto_fill": self.auto_fill,
            "fill_model": self.fill_model,
            "ephemeral": self.ephemeral,
            "state_backend": self.state_backend,
            "ohlcv_source": self.ohlcv_source.value,
            "require_closed_candle": self.require_closed_candle,
            "candle_max_age_seconds": self.candle_max_age_seconds,
            "max_catchup_candles": self.max_catchup_candles,
            "max_missing_candles": self.max_missing_candles,
            "reject_revised_candle": self.reject_revised_candle,
            "funding_source": self.funding_source.value,
            "funding_max_age_seconds": self.funding_max_age_seconds,
            "funding_poll_interval_seconds": self.funding_poll_interval_seconds,
            "account_events_enabled": self.account_events_enabled,
            "idempotency_lease_seconds": self.idempotency_lease_seconds,
            "long_signal": str(self.default_long_signal),
            "short_signal": str(self.default_short_signal),
            "tick_size": str(self.tick_size),
            "qty_step": str(self.qty_step),
            "min_qty": str(self.min_qty),
            "min_notional": str(self.min_notional),
            "maker_fee_rate": str(self.maker_fee_rate),
            "taker_fee_rate": str(self.taker_fee_rate),
            "volume_participation": str(self.volume_participation),
            "market_slippage_bps": str(self.market_slippage_bps),
            "max_entry_layers_per_bar": self.max_entry_layers_per_bar,
            "max_reduce_layers_per_bar": self.max_reduce_layers_per_bar,
            "max_fill_ratio_per_order": str(self.max_fill_ratio_per_order),
            "max_fills_per_bar": self.max_fills_per_bar,
            "bar_volume": None if self.bar_volume is None else str(self.bar_volume),
        }
        if self.state_path is not None:
            payload["state_path"] = self.state_path
        return payload
