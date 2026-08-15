# pragma pylint: disable=W0603
"""Wallet"""

import logging
import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, NamedTuple

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config, IntOrInf
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import Exchange
from freqtrade.hedge.compatibility import effective_trade_position_side
from freqtrade.hedge.symbols import canonicalize_symbol
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence import LocalTrade, Trade, WalletHistory
from freqtrade.util import dt_floor_day, dt_now


logger = logging.getLogger(__name__)


def _finite_float(value: object, *, field_name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite numeric value.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value.") from exc
    if not math.isfinite(result):
        qualifier = "a nonnegative finite value" if nonnegative else "finite"
        raise ValueError(f"{field_name} must be {qualifier}.")
    if nonnegative and result < 0:
        raise ValueError(f"{field_name} must be a nonnegative finite value.")
    return result


# wallet data structure

def normalize_position_wallet_side(side: object | None) -> str:
    # Normalize CCXT and Freqtrade position sides to LONG/SHORT/BOTH.
    raw = getattr(side, "value", side)
    normalized = str(raw or "BOTH").strip().upper()
    aliases = {
        "LONG": "LONG",
        "SHORT": "SHORT",
        "BOTH": "BOTH",
        "BUY": "LONG",
        "SELL": "SHORT",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported position wallet side: {side}")
    return aliases[normalized]




def normalize_wallet_symbol(symbol: object) -> str:
    """Normalize a generic Freqtrade pair without imposing Hedge USDT-M rules.

    Contract pairs containing a settlement suffix still use the strict Hedge
    codec.  Plain spot/one-way pairs such as ETH/BTC and NEO/USDT retain the
    normal Freqtrade pair form.
    """

    raw = str(symbol or "").strip().upper().replace("-", "/").replace("_", "/")
    if not raw:
        raise ValueError("Wallet symbol must not be empty.")
    if ":" in raw:
        return canonicalize_symbol(raw)
    if raw.count("/") == 1:
        base, quote = raw.split("/", 1)
        if base and quote:
            return f"{base}/{quote}"
    return canonicalize_symbol(raw)

def position_wallet_key(symbol: str, side: object | None) -> tuple[str, str]:
    return (
        normalize_wallet_symbol(symbol),
        normalize_position_wallet_side(side),
    )


def legacy_position_projection(
    positions_by_side: dict[tuple[str, str], "PositionWallet"],
) -> dict[str, "PositionWallet"]:
    """Expose the legacy pair index only when exactly one leg exists."""

    grouped: dict[str, list[PositionWallet]] = {}
    for (symbol, _side), position in positions_by_side.items():
        grouped.setdefault(symbol, []).append(position)
    return {
        symbol: positions[0]
        for symbol, positions in grouped.items()
        if len(positions) == 1
    }


class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class PositionWallet(NamedTuple):
    symbol: str
    position: float = 0
    leverage: float | None = 0  # Don't use this - it's not guaranteed to be set
    collateral: float = 0
    side: str = "long"
    mark_price: float | None = None
    # Exchange-reported unrealized PnL. Always zero in dry-run/backtest.
    unrealized_pnl: float = 0

    @property
    def normalized_side(self) -> str:
        return normalize_position_wallet_side(self.side)

    @property
    def absolute_position(self) -> float:
        return abs(_finite_float(self.position, field_name="position"))

    @property
    def validated_collateral(self) -> float:
        return _finite_float(
            self.collateral,
            field_name="collateral",
            nonnegative=True,
        )

    @property
    def notional(self) -> float | None:
        if self.mark_price is None:
            return None
        price = _finite_float(self.mark_price, field_name="mark_price")
        if price <= 0:
            return None
        return self.absolute_position * price


class HedgePositionQuantities(NamedTuple):
    symbol: str
    gross_long: float
    gross_short: float
    gross_total: float
    net: float


class HedgePositionExposure(NamedTuple):
    symbol: str
    gross_long_notional: Decimal
    gross_short_notional: Decimal
    gross_total_notional: Decimal
    net_notional: Decimal
    gross_exposure_ratio: Decimal | None
    net_exposure_ratio: Decimal | None


class Wallets:
    def __init__(self, config: Config, exchange: Exchange, is_backtest: bool = False) -> None:
        self._config = config
        self._is_backtest = is_backtest
        self._exchange = exchange
        self._wallets: dict[str, Wallet] = {}
        self._positions: dict[str, PositionWallet] = {}
        self._positions_by_side: dict[tuple[str, str], PositionWallet] = {}
        self._start_cap: dict[str, float] = {}

        self._stake_currency = self._exchange.get_proxy_coin()

        if isinstance(_start_cap := config["dry_run_wallet"], float | int):
            self._start_cap[self._stake_currency] = _start_cap
        else:
            self._start_cap = _start_cap

        self._last_wallet_refresh: datetime | None = None
        self.update()

    def __repr__(self) -> str:
        return (
            f"Wallets(stake_currency={self._stake_currency}, start_cap={self._start_cap}, "
            f"wallets={len(self._wallets)}, positions={len(self._positions_by_side)})"
        )

    def get_free(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        return 0.0 if balance is None else _finite_float(
            balance.free, field_name=f"wallet[{currency}].free"
        )

    def get_used(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        return 0.0 if balance is None else _finite_float(
            balance.used, field_name=f"wallet[{currency}].used"
        )

    def get_total(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        return 0.0 if balance is None else _finite_float(
            balance.total, field_name=f"wallet[{currency}].total"
        )

    def get_collateral(self) -> float:
        """Return account collateral without collapsing LONG and SHORT legs."""
        if self._config.get("margin_mode") == "cross":
            return self.get_free(self._stake_currency) + sum(
                (pos.validated_collateral for pos in self._positions_by_side.values()),
                0.0,
            )
        return self.get_total(self._stake_currency)

    def get_position_collateral(
        self,
        pair: str,
        position_side: object,
    ) -> float:
        position = self.get_position_by_side(pair, position_side)
        return 0.0 if position is None else position.validated_collateral

    def get_owned(
        self,
        pair: str,
        base_currency: str,
        position_side: object | None = None,
    ) -> float:
        """
        Get currently owned value.

        In Hedge Mode callers must pass position_side when both legs exist.
        The legacy pair-only path fails closed by returning zero for ambiguity.
        """
        if self._config.get("trading_mode", "spot") != TradingMode.FUTURES:
            return self.get_total(base_currency) or 0
        if position_side is None:
            canonical = normalize_wallet_symbol(pair)
            matches = [
                position
                for (symbol, _side), position in self.get_all_positions_by_side().items()
                if symbol == canonical
            ]
            if len(matches) > 1:
                raise OperationalException(
                    "HEDGE_POSITION_SIDE_REQUIRED: pair-only wallet ownership is ambiguous"
                )
        if pos := self.get_position(pair, position_side):
            return pos.absolute_position
        return 0

    def get_owned_by_side(
        self,
        pair: str,
        base_currency: str,
        position_side: object,
    ) -> float:
        """Explicit side-aware ownership API for every hedge caller."""
        return self.get_owned(pair, base_currency, position_side)

    def get_exit_quantity(
        self,
        pair: str,
        position_side: object,
        *,
        requested_quantity: float | None = None,
        pending_reduce_quantity: float = 0.0,
    ) -> float:
        """Clip an exit to confirmed side quantity minus local pending reductions."""
        confirmed = self.get_owned_by_side(pair, "", position_side)
        pending = float(pending_reduce_quantity)
        if not math.isfinite(pending) or pending < 0:
            raise ValueError("pending_reduce_quantity must be a nonnegative finite value.")
        available = max(confirmed - pending, 0.0)
        if requested_quantity is None:
            return available
        requested = float(requested_quantity)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("requested_quantity must be a nonnegative finite value.")
        return min(requested, available)

    def get_hedge_quantities(self, pair: str) -> HedgePositionQuantities:
        symbol = canonicalize_symbol(pair)
        long_position = self.get_position_by_side(symbol, "LONG")
        short_position = self.get_position_by_side(symbol, "SHORT")
        gross_long = 0.0 if long_position is None else long_position.absolute_position
        gross_short = 0.0 if short_position is None else short_position.absolute_position
        return HedgePositionQuantities(
            symbol=symbol,
            gross_long=gross_long,
            gross_short=gross_short,
            gross_total=gross_long + gross_short,
            net=gross_long - gross_short,
        )

    def get_hedge_exposure(
        self,
        pair: str,
        *,
        reference_price: float | Decimal | None = None,
        equity: float | Decimal | None = None,
    ) -> HedgePositionExposure:
        """Return quote-notional and ratio fields as separate typed values."""
        quantities = self.get_hedge_quantities(pair)
        long_position = self.get_position_by_side(pair, "LONG")
        short_position = self.get_position_by_side(pair, "SHORT")

        def resolve_price(position: PositionWallet | None, side: str) -> Decimal:
            if reference_price is not None:
                raw = reference_price
            elif position is not None:
                raw = position.mark_price
            else:
                return Decimal("0")
            if raw is None:
                raise ValueError(
                    f"reference_price is required when the {side} mark price is unavailable."
                )
            price = Decimal(str(raw))
            if not price.is_finite() or price <= 0:
                raise ValueError(f"{side} reference price must be a positive finite value.")
            return price

        long_price = resolve_price(long_position, "LONG")
        short_price = resolve_price(short_position, "SHORT")
        gross_long_notional = Decimal(str(quantities.gross_long)) * long_price
        gross_short_notional = Decimal(str(quantities.gross_short)) * short_price
        gross_total_notional = gross_long_notional + gross_short_notional
        net_notional = gross_long_notional - gross_short_notional
        gross_ratio: Decimal | None = None
        net_ratio: Decimal | None = None
        if equity is not None:
            equity_value = Decimal(str(equity))
            if not equity_value.is_finite() or equity_value <= 0:
                raise ValueError("equity must be a positive finite value.")
            gross_ratio = gross_total_notional / equity_value
            net_ratio = net_notional / equity_value
        return HedgePositionExposure(
            symbol=quantities.symbol,
            gross_long_notional=gross_long_notional,
            gross_short_notional=gross_short_notional,
            gross_total_notional=gross_total_notional,
            net_notional=net_notional,
            gross_exposure_ratio=gross_ratio,
            net_exposure_ratio=net_ratio,
        )

    def get_position_risk_legs(
        self,
        *,
        account_id: str,
        pair: str | None = None,
        liquidation_prices: dict[tuple[str, str], float | Decimal] | None = None,
        maintenance_margins: dict[tuple[str, str], float | Decimal] | None = None,
    ) -> tuple[object, ...]:
        """Convert wallet positions to direction-three ``PositionRiskLeg`` objects.

        The import is intentionally local so the legacy Wallet module remains
        importable before the risk package is initialized.
        """

        from freqtrade.hedge.risk.portfolio import PositionRiskLeg

        canonical_pair = None if pair is None else canonicalize_symbol(pair)
        liquidation_prices = liquidation_prices or {}
        maintenance_margins = maintenance_margins or {}
        legs = []
        for (symbol, side), position in self.get_all_positions_by_side().items():
            if canonical_pair is not None and symbol != canonical_pair:
                continue
            quantity = Decimal(str(position.absolute_position))
            if quantity <= 0:
                continue
            if position.mark_price is None:
                raise ValueError(f"mark_price is required for {symbol} {side}.")
            mark_price = Decimal(str(position.mark_price))
            if not mark_price.is_finite() or mark_price <= 0:
                raise ValueError(f"mark_price must be positive and finite for {symbol} {side}.")
            leverage = Decimal(str(position.leverage or 1))
            key = position_wallet_key(symbol, side)
            liquidation = liquidation_prices.get(key)
            maintenance = maintenance_margins.get(key, Decimal("0"))
            legs.append(
                PositionRiskLeg(
                    account_id=account_id,
                    symbol=symbol,
                    position_side=side,
                    quantity=quantity,
                    mark_price=mark_price,
                    leverage=leverage,
                    reported_initial_margin=Decimal(str(position.validated_collateral)),
                    maintenance_margin=Decimal(str(maintenance)),
                    liquidation_price=(
                        None if liquidation is None else Decimal(str(liquidation))
                    ),
                )
            )
        return tuple(legs)

    def build_hedge_risk_portfolio(
        self,
        *,
        account_id: str,
        equity: float | Decimal,
        available_balance: float | Decimal | None = None,
        wallet_balance: float | Decimal | None = None,
        pending_orders=(),
        pair: str | None = None,
        liquidation_prices: dict[tuple[str, str], float | Decimal] | None = None,
        maintenance_margins: dict[tuple[str, str], float | Decimal] | None = None,
        maintenance_margin: float | Decimal | None = None,
        risk_data_valid: bool = True,
        observed_at_ms: int | None = None,
        exchange: str = "binance",
        snapshot_id: str | None = None,
        source_version: int | None = None,
        exchange_time_ms: int | None = None,
        strict_completeness: bool = True,
    ):
        """Build a compatibility risk snapshot from the current Wallet projection.

        Live approval code should prefer :meth:`build_hedge_risk_portfolio_from_facts`,
        because direction-two REST/User Stream facts carry authoritative versions,
        timestamps, pending orders and reconciliation state.  This compatibility path
        fails closed when liquidation or maintenance-margin data is incomplete.
        """

        from freqtrade.hedge.risk.portfolio import build_risk_portfolio

        positions = self.get_position_risk_legs(
            account_id=account_id,
            pair=pair,
            liquidation_prices=liquidation_prices,
            maintenance_margins=maintenance_margins,
        )
        wallet_value = (
            Decimal(str(self.get_collateral()))
            if wallet_balance is None
            else Decimal(str(wallet_balance))
        )
        available_value = (
            Decimal(str(self.get_free(self._stake_currency)))
            if available_balance is None
            else Decimal(str(available_balance))
        )
        return build_risk_portfolio(
            exchange=exchange,
            account_id=account_id,
            equity=Decimal(str(equity)),
            wallet_balance=wallet_value,
            available_balance=available_value,
            positions=positions,
            pending_orders=tuple(pending_orders),
            maintenance_margin=(
                None if maintenance_margin is None else Decimal(str(maintenance_margin))
            ),
            risk_data_valid=risk_data_valid,
            observed_at_ms=observed_at_ms,
            snapshot_id=snapshot_id,
            source_version=0 if source_version is None else source_version,
            exchange_time_ms=exchange_time_ms,
            strict_completeness=strict_completeness,
        )

    def build_hedge_risk_portfolio_from_facts(self, facts):
        """Build risk state from authoritative account facts supplied by direction two.

        The explicit type check prevents an arbitrary Wallet-derived mapping from being
        mistaken for a reconciled REST/User Stream/ledger snapshot.
        """

        from freqtrade.hedge.risk.facts import AccountRiskFacts

        if not isinstance(facts, AccountRiskFacts):
            raise TypeError("facts must be an AccountRiskFacts instance.")
        return facts.to_portfolio()

    def _update_dry(self) -> None:
        """
        Update from database in dry-run mode
        - Apply profits of closed trades on top of stake amount
        - Subtract currently tied up stake_amount in open trades
        - update balances for currencies currently in trades
        """
        # Recreate _wallets to reset closed trade balances
        self._positions_by_side = {}
        _wallets = {}
        _positions = {}
        open_trades = Trade.get_trades_proxy(is_open=True)
        if not self._is_backtest:
            # Live / Dry-run mode
            tot_profit = Trade.get_total_closed_profit()
        else:
            # Backtest mode
            tot_profit = LocalTrade.bt_total_profit
        tot_profit += sum(trade.realized_profit for trade in open_trades)
        tot_in_trades = sum(trade.stake_amount for trade in open_trades)
        used_stake = 0.0

        if self._config.get("trading_mode", "spot") != TradingMode.FUTURES:
            for trade in open_trades:
                curr = self._exchange.get_pair_base_currency(trade.pair)
                used_stake += sum(
                    o.stake_amount for o in trade.open_orders if o.ft_order_side == trade.entry_side
                )
                pending = sum(
                    o.amount
                    for o in trade.open_orders
                    if o.amount and o.ft_order_side == trade.exit_side
                )
                curr_wallet_bal = self._start_cap.get(curr, 0)

                _wallets[curr] = Wallet(
                    curr,
                    curr_wallet_bal + trade.amount - pending,
                    pending,
                    trade.amount + curr_wallet_bal,
                )
        else:
            for position in open_trades:
                wallet = PositionWallet(
                    normalize_wallet_symbol(position.pair),
                    position=position.amount,
                    leverage=position.leverage,
                    collateral=position.stake_amount,
                    side=position.trade_direction,
                    mark_price=getattr(position, "open_rate", None),
                )
                self._positions_by_side[
                    position_wallet_key(position.pair, position.trade_direction)
                ] = wallet

            _positions = legacy_position_projection(self._positions_by_side)
            used_stake = tot_in_trades

        cross_margin = 0.0
        if self._config.get("margin_mode") == "cross":
            # In cross-margin mode, the total balance is used as collateral.
            # This is moved as "free" into the stake currency balance.
            # strongly tied to the get_collateral() implementation.
            for curr, bal in self._start_cap.items():
                if curr == self._stake_currency:
                    continue
                rate = self._exchange.get_conversion_rate(curr, self._stake_currency)
                if rate:
                    cross_margin += bal * rate

        current_stake = self._start_cap.get(self._stake_currency, 0) + tot_profit - tot_in_trades
        total_stake = current_stake + used_stake

        _wallets[self._stake_currency] = Wallet(
            currency=self._stake_currency,
            free=current_stake + cross_margin,
            used=used_stake,
            total=total_stake,
        )
        for currency, bal in self._start_cap.items():
            if currency not in _wallets:
                _wallets[currency] = Wallet(currency, bal, 0, bal)

        self._wallets = _wallets
        self._positions = _positions

    def _update_live(self) -> None:
        self._positions_by_side = {}
        balances = self._exchange.get_balances()
        _wallets = {}

        for currency in balances:
            if isinstance(balances[currency], dict):
                _wallets[currency] = Wallet(
                    currency,
                    balances[currency].get("free", 0),
                    balances[currency].get("used", 0),
                    balances[currency].get("total", 0),
                )

        positions = self._exchange.fetch_positions()
        _parsed_positions = {}
        for position in positions:
            symbol = position["symbol"]
            if position["side"] is None:
                continue
            size = self._exchange._contracts_to_amount(symbol, position["contracts"])
            if not size:
                continue
            collateral = safe_value_fallback(position, "initialMargin", "collateral", 0.0)
            leverage: float | None = position.get("leverage")
            if not leverage:
                trade = Trade.get_trades_proxy(is_open=True, pair=symbol)
                leverage = trade[0].leverage if trade else None
            wallet = PositionWallet(
                normalize_wallet_symbol(symbol),
                position=size,
                leverage=leverage,
                collateral=collateral,
                side=position["side"],
                mark_price=safe_value_fallback(position, "markPrice", "mark_price", None),
                unrealized_pnl=float(position.get("unrealizedPnl") or 0.0),
            )
            self._positions_by_side[
                position_wallet_key(symbol, position["side"])
            ] = wallet
        _parsed_positions = legacy_position_projection(self._positions_by_side)
        self._positions = _parsed_positions
        self._wallets = self._strip_unrealized_pnl(_wallets, self._positions_by_side)

    def _strip_unrealized_pnl(
        self,
        wallets: dict[str, Wallet],
        positions: dict[tuple[str, str], PositionWallet],
    ) -> dict[str, Wallet]:
        """Remove exchange-included unrealized PnL from stake wallet totals.

        Some exchanges map account equity to ``Wallet.total``. Hedge mode must
        aggregate every independent LONG/SHORT leg rather than the legacy pair
        projection, otherwise one leg can be omitted.
        """

        if not positions or not self._exchange.balance_includes_unrealized_pnl():
            return wallets
        upnl = sum(position.unrealized_pnl for position in positions.values())
        stake_wallet = wallets.get(self._stake_currency)
        if not upnl or stake_wallet is None:
            return wallets
        wallets[self._stake_currency] = stake_wallet._replace(total=stake_wallet.total - upnl)
        return wallets

    def update(self, require_update: bool = True) -> None:
        """
        Updates wallets from the configured version.
        By default, updates from the exchange.
        Update-skipping should only be used for user-invoked /balance calls, since
        for trading operations, the latest balance is needed.
        :param require_update: Allow skipping an update if balances were recently refreshed
        """
        now = dt_now()
        if (
            require_update
            or self._last_wallet_refresh is None
            or (self._last_wallet_refresh + timedelta(seconds=3600) < now)
        ):
            if not self._config["dry_run"] or self._config.get("runmode") == RunMode.LIVE:
                self._update_live()
            else:
                self._update_dry()
            self._local_log("Wallets synced.")
            self._last_wallet_refresh = dt_now()

    def get_all_balances(self) -> dict[str, Wallet]:
        return self._wallets

    def get_all_positions_by_side(
        self,
    ) -> dict[tuple[str, str], PositionWallet]:
        positions = dict(self._positions_by_side)
        side_indexed_symbols = {symbol for symbol, _side in positions}

        # The side-aware index is authoritative.  The legacy pair-only
        # projection may contain an opaque sentinel or stale compatibility
        # object, so never inspect it when that symbol is already represented
        # by explicit LONG/SHORT entries.
        for symbol, position in self._positions.items():
            canonical_symbol = normalize_wallet_symbol(symbol)
            if canonical_symbol in side_indexed_symbols:
                continue

            side = getattr(position, "normalized_side", None)
            if side is None:
                raw_side = getattr(position, "side", None)
                if raw_side is None:
                    # A pair-only object without side information cannot be
                    # projected safely into a LONG/SHORT identity.
                    continue
                side = normalize_position_wallet_side(raw_side)

            key = (canonical_symbol, normalize_position_wallet_side(side))
            positions.setdefault(key, position)
            side_indexed_symbols.add(canonical_symbol)
        return positions

    def get_position_by_side(
        self,
        symbol: str,
        position_side: object,
    ) -> PositionWallet | None:
        key = position_wallet_key(symbol, position_side)
        position = self._positions_by_side.get(key)
        if position is not None:
            return position
        legacy = self._positions.get(normalize_wallet_symbol(symbol))
        if legacy is not None and legacy.normalized_side == key[1]:
            return legacy
        return None

    def get_position(
        self,
        symbol: str,
        position_side: object | None = None,
    ) -> PositionWallet | None:
        if position_side is not None:
            return self.get_position_by_side(symbol, position_side)
        canonical = normalize_wallet_symbol(symbol)
        matches = [
            position
            for (pair, _side), position in self.get_all_positions_by_side().items()
            if pair == canonical
        ]
        if len(matches) == 1:
            return matches[0]
        return self._positions.get(canonical) if not matches else None

    def get_all_positions(self) -> dict[str, PositionWallet]:
        """Legacy projection; hedge callers must use get_all_positions_by_side()."""
        return dict(self._positions)

    def _check_exit_amount(self, trade: Trade) -> bool:
        if trade.trading_mode != TradingMode.FUTURES:
            # Slightly higher offset than in safe_exit_amount.
            wallet_amount: float = self.get_total(trade.safe_base_currency) * (2 - 0.981)
        else:
            # wallet_amount: float = self.wallets.get_free(trade.safe_base_currency)
            side = effective_trade_position_side(trade)
            wallet_amount = self.get_exit_quantity(
                trade.pair,
                side,
                requested_quantity=trade.amount,
            )

        if wallet_amount >= trade.amount:
            return True
        return False

    def check_exit_amount(self, trade: Trade) -> bool:
        """
        Checks if the exit amount is available in the wallet.
        :param trade: Trade to check
        :return: True if the exit amount is available, False otherwise
        """
        if not self._check_exit_amount(trade):
            # Update wallets just to make sure
            self.update()
            return self._check_exit_amount(trade)

        return True

    def get_starting_balance(self) -> float:
        """
        Retrieves starting balance - based on either available capital,
        or by using current balance subtracting
        """
        if "available_capital" in self._config:
            return self._config["available_capital"]
        else:
            tot_profit = Trade.get_total_closed_profit()
            open_stakes = Trade.total_open_trades_stakes()
            available_balance = self.get_free(self._stake_currency)
            return (available_balance - tot_profit + open_stakes) * self._config[
                "tradable_balance_ratio"
            ]

    def get_total_stake_amount(self):
        """
        Return the total currently available balance in stake currency, including tied up stake and
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount) * tradable_balance_ratio
        """
        val_tied_up = Trade.total_open_trades_stakes()
        if "available_capital" in self._config:
            starting_balance = self._config["available_capital"]
            tot_profit = Trade.get_total_closed_profit()
            available_amount = starting_balance + tot_profit

        else:
            # Ensure <tradable_balance_ratio>% is used from the overall balance
            # Otherwise we'd risk lowering stakes with each open trade.
            # (tied up + current free) * ratio) - tied up
            available_amount = (val_tied_up + self.get_free(self._stake_currency)) * self._config[
                "tradable_balance_ratio"
            ]
        return available_amount

    def get_available_stake_amount(self) -> float:
        """
        Return the total currently available balance in stake currency,
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount) * tradable_balance_ratio - <open_trade stakes>
        """

        free = self.get_free(self._stake_currency)
        return min(self.get_total_stake_amount() - Trade.total_open_trades_stakes(), free)

    def _calculate_unlimited_stake_amount(
        self, available_amount: float, val_tied_up: float, max_open_trades: IntOrInf
    ) -> float:
        """
        Calculate stake amount for "unlimited" stake amount
        :return: 0 if max number of trades reached, else stake_amount to use.
        """
        if max_open_trades == 0:
            return 0

        possible_stake = (available_amount + val_tied_up) / max_open_trades
        # Theoretical amount can be above available amount - therefore limit to available amount!
        return min(possible_stake, available_amount)

    def _check_available_stake_amount(self, stake_amount: float, available_amount: float) -> float:
        """
        Check if stake amount can be fulfilled with the available balance
        for the stake currency
        :return: float: Stake amount
        :raise: DependencyException if balance is lower than stake-amount
        """

        if self._config["amend_last_stake_amount"]:
            # Remaining amount needs to be at least stake_amount * last_stake_amount_min_ratio
            # Otherwise the remaining amount is too low to trade.
            if available_amount > (stake_amount * self._config["last_stake_amount_min_ratio"]):
                stake_amount = min(stake_amount, available_amount)
            else:
                stake_amount = 0

        if available_amount < stake_amount:
            raise DependencyException(
                f"Available balance ({available_amount} {self._config['stake_currency']}) is "
                f"lower than stake amount ({stake_amount} {self._config['stake_currency']})"
            )

        return max(stake_amount, 0)

    def get_trade_stake_amount(
        self, pair: str, max_open_trades: IntOrInf, update: bool = True
    ) -> float:
        """
        Calculate stake amount for the trade
        :return: float: Stake amount
        :raise: DependencyException if the available stake amount is too low
        """
        stake_amount: float
        # Ensure wallets are up-to-date.
        if update:
            self.update()
        val_tied_up = Trade.total_open_trades_stakes()
        available_amount = self.get_available_stake_amount()

        stake_amount = self._config["stake_amount"]
        if stake_amount == UNLIMITED_STAKE_AMOUNT:
            stake_amount = self._calculate_unlimited_stake_amount(
                available_amount, val_tied_up, max_open_trades
            )

        return self._check_available_stake_amount(stake_amount, available_amount)

    def validate_stake_amount(
        self,
        pair: str,
        stake_amount: float | None,
        min_stake_amount: float | None,
        max_stake_amount: float,
        trade_amount: float | None,
    ):
        if not stake_amount or isinstance(stake_amount, str) or stake_amount <= 0:
            self._local_log(
                f"Stake amount is {stake_amount}, ignoring possible trade for {pair}.",
                level="debug",
            )
            return 0

        max_allowed_stake = min(max_stake_amount, self.get_available_stake_amount())
        if trade_amount:
            # if in a trade, then the resulting trade size cannot go beyond the max stake
            # Otherwise we could no longer exit.
            max_allowed_stake = min(max_allowed_stake, max_stake_amount - trade_amount)

        if min_stake_amount is not None and min_stake_amount > max_allowed_stake:
            self._local_log(
                "Minimum stake amount > available balance. "
                f"{min_stake_amount} > {max_allowed_stake}",
                level="warning",
            )
            return 0
        if min_stake_amount is not None and stake_amount < min_stake_amount:
            self._local_log(
                f"Stake amount for pair {pair} is too small "
                f"({stake_amount} < {min_stake_amount}), adjusting to {min_stake_amount}."
            )
            if stake_amount * 1.3 < min_stake_amount:
                # Top-cap stake-amount adjustments to +30%.
                self._local_log(
                    f"Adjusted stake amount for pair {pair} is more than 30% bigger than "
                    f"the desired stake amount of ({stake_amount:.8f} * 1.3 = "
                    f"{stake_amount * 1.3:.8f}) < {min_stake_amount}), ignoring trade."
                )
                return 0
            stake_amount = min_stake_amount

        if stake_amount > max_allowed_stake:
            self._local_log(
                f"Stake amount for pair {pair} is too big "
                f"({stake_amount} > {max_allowed_stake}), adjusting to {max_allowed_stake}."
            )
            stake_amount = max_allowed_stake
        return stake_amount

    def _local_log(self, msg: str, level: Literal["info", "warning", "debug"] = "info") -> None:
        """
        Log a message to the local log.
        """
        if not self._is_backtest:
            if level == "warning":
                logger.warning(msg)
            elif level == "debug":
                logger.debug(msg)
            else:
                logger.info(msg)

    def record_wallet_state(self) -> None:
        """Record daily wallet totals to database"""
        if self._is_backtest:
            # only record in live mode.
            return
        timestamp = dt_floor_day(dt_now())

        # Record total balances for all currencies
        wallet_records = []
        position_collaterals = 0.0
        open_assets: dict[str, Trade] = {t.safe_base_currency: t for t in Trade.get_open_trades()}
        for pos in self.get_all_positions_by_side().values():
            base = self._exchange.get_pair_base_currency(pos.symbol)
            rate = self._exchange.get_conversion_rate(base, self._stake_currency)
            total_quote = None
            leverage = pos.leverage or 1.0
            if rate:
                # Same formula than in rpc's _rpc_balance
                total_quote = (
                    rate * pos.position - pos.collateral * (leverage - 1)
                    if pos.normalized_side == "LONG"
                    else pos.collateral * (1 + leverage) - rate * pos.position
                )

            position_record = WalletHistory(
                timestamp=timestamp,
                currency=pos.symbol,
                quote_currency=self._stake_currency,
                rate=rate,
                balance=pos.absolute_position,
                total_quote=total_quote,
                total_position_value=rate * pos.absolute_position if rate else None,
                collateral=pos.collateral,
                leverage=leverage,
                bot_managed=base in open_assets,
            )
            position_collaterals += pos.collateral
            wallet_records.append(position_record)

        for wallet in self.get_all_balances().values():
            if wallet.total == 0:
                continue
            rate = self._exchange.get_conversion_rate(wallet.currency, self._stake_currency)
            is_bot_managed = (
                self._stake_currency == wallet.currency or wallet.currency in open_assets
            )
            balance = wallet.total - (
                position_collaterals if wallet.currency == self._stake_currency else 0
            )
            total_quote = rate * balance if rate else None

            wallet_record = WalletHistory(
                timestamp=timestamp,
                currency=wallet.currency,
                quote_currency=self._stake_currency,
                rate=rate,
                balance=balance,
                leverage=1.0,
                total_quote=total_quote,
                bot_managed=is_bot_managed,
            )
            wallet_records.append(wallet_record)
        try:
            WalletHistory.session.bulk_save_objects(wallet_records)
            WalletHistory.session.commit()
        except Exception as e:
            WalletHistory.session.rollback()
            logger.error(f"Error saving wallet balance records: {e}")
