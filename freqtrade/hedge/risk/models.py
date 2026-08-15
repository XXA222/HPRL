"""Risk-domain models with explicit quantity, notional and ratio units."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import uuid

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.errors import HedgeConfigurationError
from freqtrade.hedge.numeric import (
    require_nonnegative,
    require_positive,
    require_unit_interval,
    to_decimal,
)
from freqtrade.hedge.symbols import canonicalize_symbol
from freqtrade.hedge.risk.identity import RiskPositionKey


_NET_ABSOLUTE_TOLERANCE = Decimal("1e-8")
_NET_RELATIVE_TOLERANCE = Decimal("1e-12")


def _normalized_account_id(value: str) -> str:
    if not isinstance(value, str):
        raise HedgeConfigurationError("account_id must be a string.")
    account_id = value.strip()
    if not account_id:
        raise HedgeConfigurationError("account_id must not be empty.")
    return account_id


@dataclass(frozen=True, slots=True)
class AccountRiskSnapshot:
    """Account-level risk facts.

    All ``*_notional`` values are quote-currency amounts. All ``*_ratio``
    values are dimensionless. They are intentionally separate fields so a
    value such as ``0.8`` can never be compared with ``8000`` by accident.
    """

    account_id: str
    equity: Decimal
    wallet_balance: Decimal
    available_balance: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    gross_long_notional: Decimal
    gross_short_notional: Decimal
    net_notional: Decimal
    pending_order_notional: Decimal = Decimal("0")
    liquidation_buffer_ratio: Decimal = Decimal("1")
    risk_data_valid: bool = True
    observed_at_ms: int | None = None
    pending_order_initial_margin: Decimal = Decimal("0")
    pending_net_notional_delta: Decimal = Decimal("0")
    pending_long_notional: Decimal = Decimal("0")
    pending_short_notional: Decimal = Decimal("0")
    pending_order_maintenance_margin: Decimal = Decimal("0")
    exchange: str = "binance"
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_version: int = 0
    exchange_time_ms: int | None = None
    liquidation_data_complete: bool = True
    maintenance_margin_complete: bool = True
    risk_data_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _normalized_account_id(self.account_id))
        for field_name in (
            "equity",
            "wallet_balance",
            "available_balance",
            "initial_margin",
            "maintenance_margin",
            "gross_long_notional",
            "gross_short_notional",
            "pending_order_notional",
            "pending_order_initial_margin",
            "pending_long_notional",
            "pending_short_notional",
            "pending_order_maintenance_margin",
        ):
            object.__setattr__(
                self,
                field_name,
                require_nonnegative(getattr(self, field_name), field=field_name),
            )
        if self.equity <= 0:
            raise HedgeConfigurationError("equity must be positive.")
        if not isinstance(self.risk_data_valid, bool):
            raise HedgeConfigurationError("risk_data_valid must be a boolean.")
        for field_name in ("liquidation_data_complete", "maintenance_margin_complete"):
            if not isinstance(getattr(self, field_name), bool):
                raise HedgeConfigurationError(f"{field_name} must be a boolean.")
        if not isinstance(self.exchange, str) or not self.exchange.strip():
            raise HedgeConfigurationError("exchange must not be empty.")
        object.__setattr__(self, "exchange", self.exchange.strip().lower())
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise HedgeConfigurationError("snapshot_id must not be empty.")
        object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())
        if (
            isinstance(self.source_version, bool)
            or not isinstance(self.source_version, int)
            or self.source_version < 0
        ):
            raise HedgeConfigurationError("source_version must be a nonnegative integer.")
        if not isinstance(self.risk_data_errors, tuple):
            raise HedgeConfigurationError("risk_data_errors must be a tuple.")
        errors = tuple(
            item.strip()
            for item in self.risk_data_errors
            if isinstance(item, str) and item.strip()
        )
        if len(errors) != len(self.risk_data_errors):
            raise HedgeConfigurationError("risk_data_errors must contain non-empty strings only.")
        object.__setattr__(self, "risk_data_errors", tuple(dict.fromkeys(errors)))

        net = to_decimal(self.net_notional, field="net_notional")
        pending_net_delta = to_decimal(
            self.pending_net_notional_delta,
            field="pending_net_notional_delta",
        )
        if net is None or pending_net_delta is None:  # defensive; allow_none=False above
            raise HedgeConfigurationError("Net notional fields are required.")
        object.__setattr__(self, "net_notional", net)
        object.__setattr__(self, "pending_net_notional_delta", pending_net_delta)
        if abs(pending_net_delta) > self.pending_order_notional:
            raise HedgeConfigurationError(
                "abs(pending_net_notional_delta) must not exceed pending_order_notional."
            )
        if self.pending_order_initial_margin > self.pending_order_notional:
            raise HedgeConfigurationError(
                "pending_order_initial_margin must not exceed pending_order_notional."
            )
        attributed_pending = self.pending_long_notional + self.pending_short_notional
        if attributed_pending > self.pending_order_notional:
            raise HedgeConfigurationError(
                "pending_long_notional + pending_short_notional must not exceed "
                "pending_order_notional."
            )

        expected_net = self.gross_long_notional - self.gross_short_notional
        tolerance = max(
            _NET_ABSOLUTE_TOLERANCE,
            self.gross_total_notional * _NET_RELATIVE_TOLERANCE,
        )
        if abs(net - expected_net) > tolerance:
            raise HedgeConfigurationError(
                "net_notional must equal gross_long_notional - gross_short_notional "
                "within the configured numeric tolerance."
            )

        object.__setattr__(
            self,
            "liquidation_buffer_ratio",
            require_unit_interval(
                self.liquidation_buffer_ratio,
                field="liquidation_buffer_ratio",
            ),
        )
        for field_name in ("observed_at_ms", "exchange_time_ms"):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise HedgeConfigurationError(f"{field_name} must be an integer.")
                if value < 0:
                    raise HedgeConfigurationError(f"{field_name} must not be negative.")

    @property
    def gross_total_notional(self) -> Decimal:
        return self.gross_long_notional + self.gross_short_notional

    @property
    def gross_notional(self) -> Decimal:
        """Compatibility alias for the P2-H2 API."""

        return self.gross_total_notional

    @property
    def gross_exposure_ratio(self) -> Decimal:
        return self.gross_total_notional / self.equity

    @property
    def net_exposure_ratio(self) -> Decimal:
        return self.net_notional / self.equity

    @property
    def projected_net_notional(self) -> Decimal:
        return self.net_notional + self.pending_net_notional_delta

    @property
    def unattributed_pending_notional(self) -> Decimal:
        return max(
            self.pending_order_notional
            - self.pending_long_notional
            - self.pending_short_notional,
            Decimal("0"),
        )

    @property
    def projected_gross_long_notional(self) -> Decimal:
        return (
            self.gross_long_notional
            + self.pending_long_notional
            + self.unattributed_pending_notional
        )

    @property
    def projected_gross_short_notional(self) -> Decimal:
        return (
            self.gross_short_notional
            + self.pending_short_notional
            + self.unattributed_pending_notional
        )

    @property
    def margin_utilization(self) -> Decimal:
        return self.initial_margin / self.equity

    @property
    def projected_margin_utilization(self) -> Decimal:
        return (self.initial_margin + self.pending_order_initial_margin) / self.equity

    @property
    def effective_risk_data_valid(self) -> bool:
        return (
            self.risk_data_valid
            and self.liquidation_data_complete
            and self.maintenance_margin_complete
            and not self.risk_data_errors
        )

    @property
    def projected_maintenance_buffer_ratio(self) -> Decimal:
        projected = self.maintenance_margin + self.pending_order_maintenance_margin
        return min(max((self.equity - projected) / self.equity, Decimal("0")), Decimal("1"))

    def as_dict(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "equity": str(self.equity),
            "wallet_balance": str(self.wallet_balance),
            "available_balance": str(self.available_balance),
            "initial_margin": str(self.initial_margin),
            "maintenance_margin": str(self.maintenance_margin),
            "gross_long_notional": str(self.gross_long_notional),
            "gross_short_notional": str(self.gross_short_notional),
            "gross_total_notional": str(self.gross_total_notional),
            "net_notional": str(self.net_notional),
            "pending_order_notional": str(self.pending_order_notional),
            "pending_order_initial_margin": str(self.pending_order_initial_margin),
            "pending_order_maintenance_margin": str(self.pending_order_maintenance_margin),
            "pending_long_notional": str(self.pending_long_notional),
            "pending_short_notional": str(self.pending_short_notional),
            "pending_net_notional_delta": str(self.pending_net_notional_delta),
            "gross_exposure_ratio": str(self.gross_exposure_ratio),
            "net_exposure_ratio": str(self.net_exposure_ratio),
            "margin_utilization": str(self.margin_utilization),
            "projected_margin_utilization": str(self.projected_margin_utilization),
            "liquidation_buffer_ratio": str(self.liquidation_buffer_ratio),
            "risk_data_valid": self.risk_data_valid,
            "effective_risk_data_valid": self.effective_risk_data_valid,
            "risk_data_errors": list(self.risk_data_errors),
            "liquidation_data_complete": self.liquidation_data_complete,
            "maintenance_margin_complete": self.maintenance_margin_complete,
            "projected_maintenance_buffer_ratio": str(self.projected_maintenance_buffer_ratio),
            "exchange": self.exchange,
            "snapshot_id": self.snapshot_id,
            "source_version": self.source_version,
            "exchange_time_ms": self.exchange_time_ms,
            "observed_at_ms": self.observed_at_ms,
        }


@dataclass(frozen=True, slots=True)
class PendingOrderRisk:
    account_id: str
    symbol: str
    position_side: PositionSide
    action: PositionAction
    remaining_quantity: Decimal
    reference_price: Decimal
    leverage: Decimal = Decimal("1")
    maintenance_margin_rate: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _normalized_account_id(self.account_id))
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(str(self.position_side).upper())
        )
        if side is PositionSide.BOTH:
            raise HedgeConfigurationError("Pending hedge order side must be LONG or SHORT.")
        action = (
            self.action
            if isinstance(self.action, PositionAction)
            else PositionAction(str(self.action).upper())
        )
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(
            self,
            "remaining_quantity",
            require_nonnegative(self.remaining_quantity, field="remaining_quantity"),
        )
        object.__setattr__(
            self,
            "reference_price",
            require_positive(self.reference_price, field="reference_price"),
        )
        leverage = require_positive(self.leverage, field="leverage")
        if leverage < 1:
            raise HedgeConfigurationError("leverage must be greater than or equal to 1.")
        object.__setattr__(self, "leverage", leverage)
        rate = require_unit_interval(
            self.maintenance_margin_rate,
            field="maintenance_margin_rate",
        )
        if rate <= 0:
            raise HedgeConfigurationError("maintenance_margin_rate must be positive.")
        object.__setattr__(self, "maintenance_margin_rate", rate)

    @property
    def remaining_notional(self) -> Decimal:
        return self.remaining_quantity * self.reference_price

    @property
    def remaining_initial_margin(self) -> Decimal:
        return self.remaining_notional / self.leverage

    @property
    def remaining_maintenance_margin(self) -> Decimal:
        return self.remaining_notional * self.maintenance_margin_rate

    @property
    def signed_net_notional_delta(self) -> Decimal:
        if not self.increases_risk:
            return Decimal("0")
        sign = Decimal("1") if self.position_side is PositionSide.LONG else Decimal("-1")
        return sign * self.remaining_notional

    @property
    def increases_risk(self) -> bool:
        return self.action.increases_risk


@dataclass(frozen=True, slots=True)
class RiskRequest:
    account_id: str
    symbol: str
    position_side: PositionSide
    action: PositionAction
    requested_quantity: Decimal
    reference_price: Decimal
    current_leg_notional: Decimal = Decimal("0")
    current_symbol_gross_notional: Decimal = Decimal("0")
    pending_leg_increase_notional: Decimal = Decimal("0")
    pending_symbol_increase_notional: Decimal = Decimal("0")
    confirmed_quantity: Decimal | None = None
    pending_reduce_quantity: Decimal = Decimal("0")
    leverage: Decimal = Decimal("1")
    exchange: str = "binance"
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    expires_at_ms: int | None = None
    target_snapshot_version: int | None = None
    maintenance_margin_rate: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _normalized_account_id(self.account_id))
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(str(self.position_side).upper())
        )
        if side is PositionSide.BOTH:
            raise HedgeConfigurationError("Risk request side must be LONG or SHORT.")
        action = (
            self.action
            if isinstance(self.action, PositionAction)
            else PositionAction(str(self.action).upper())
        )
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(
            self,
            "requested_quantity",
            require_positive(self.requested_quantity, field="requested_quantity"),
        )
        object.__setattr__(
            self,
            "reference_price",
            require_positive(self.reference_price, field="reference_price"),
        )
        leverage = require_positive(self.leverage, field="leverage")
        if leverage < 1:
            raise HedgeConfigurationError("leverage must be greater than or equal to 1.")
        object.__setattr__(self, "leverage", leverage)
        if not isinstance(self.exchange, str) or not self.exchange.strip():
            raise HedgeConfigurationError("exchange must not be empty.")
        object.__setattr__(self, "exchange", self.exchange.strip().lower())
        for field_name in ("intent_id", "idempotency_key", "correlation_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HedgeConfigurationError(f"{field_name} must not be empty.")
            object.__setattr__(self, field_name, value.strip())
        for field_name in ("expires_at_ms", "target_snapshot_version"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise HedgeConfigurationError(f"{field_name} must be a nonnegative integer.")
        rate = require_unit_interval(
            self.maintenance_margin_rate,
            field="maintenance_margin_rate",
        )
        if rate <= 0:
            raise HedgeConfigurationError("maintenance_margin_rate must be positive.")
        object.__setattr__(self, "maintenance_margin_rate", rate)
        for field_name in (
            "current_leg_notional",
            "current_symbol_gross_notional",
            "pending_leg_increase_notional",
            "pending_symbol_increase_notional",
            "pending_reduce_quantity",
        ):
            object.__setattr__(
                self,
                field_name,
                require_nonnegative(getattr(self, field_name), field=field_name),
            )
        if self.confirmed_quantity is not None:
            object.__setattr__(
                self,
                "confirmed_quantity",
                require_nonnegative(self.confirmed_quantity, field="confirmed_quantity"),
            )

    @property
    def position_key(self) -> RiskPositionKey:
        return RiskPositionKey(
            exchange=self.exchange,
            account_id=self.account_id,
            symbol=self.symbol,
            position_side=self.position_side,
        )

    @property
    def requested_notional(self) -> Decimal:
        return self.requested_quantity * self.reference_price

    @property
    def requested_initial_margin(self) -> Decimal:
        return self.requested_notional / self.leverage

    @property
    def requested_maintenance_margin(self) -> Decimal:
        return self.requested_notional * self.maintenance_margin_rate

    @property
    def net_notional_sign(self) -> Decimal:
        return Decimal("1") if self.position_side is PositionSide.LONG else Decimal("-1")

    def as_dict(self) -> dict[str, object]:
        return {
            "exchange": self.exchange,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "position_key": self.position_key.as_dict(),
            "intent_id": self.intent_id,
            "idempotency_key": self.idempotency_key,
            "correlation_id": self.correlation_id,
            "expires_at_ms": self.expires_at_ms,
            "target_snapshot_version": self.target_snapshot_version,
            "position_side": self.position_side.value,
            "action": self.action.value,
            "requested_quantity": str(self.requested_quantity),
            "requested_notional": str(self.requested_notional),
            "reference_price": str(self.reference_price),
            "leverage": str(self.leverage),
            "maintenance_margin_rate": str(self.maintenance_margin_rate),
            "requested_maintenance_margin": str(self.requested_maintenance_margin),
            "current_leg_notional": str(self.current_leg_notional),
            "current_symbol_gross_notional": str(self.current_symbol_gross_notional),
            "pending_leg_increase_notional": str(self.pending_leg_increase_notional),
            "pending_symbol_increase_notional": str(self.pending_symbol_increase_notional),
            "confirmed_quantity": (
                None if self.confirmed_quantity is None else str(self.confirmed_quantity)
            ),
            "pending_reduce_quantity": str(self.pending_reduce_quantity),
        }


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    approved_quantity: Decimal
    reason_codes: tuple[str, ...]
    approved_notional: Decimal = Decimal("0")
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    risk_snapshot_id: str | None = None
    rules_version: str = "direction3-risk-v1.4"
    evaluated_at_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approved_quantity",
            require_nonnegative(self.approved_quantity, field="approved_quantity"),
        )
        object.__setattr__(
            self,
            "approved_notional",
            require_nonnegative(self.approved_notional, field="approved_notional"),
        )
        if self.allowed and (self.approved_quantity <= 0 or self.approved_notional <= 0):
            raise HedgeConfigurationError(
                "Allowed risk decision must approve positive quantity and notional."
            )
        if not self.allowed and (
            self.approved_quantity != 0 or self.approved_notional != 0
        ):
            raise HedgeConfigurationError(
                "Denied risk decision must approve zero quantity/notional."
            )
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise HedgeConfigurationError("decision_id must not be empty.")
        object.__setattr__(self, "decision_id", self.decision_id.strip())
        if self.risk_snapshot_id is not None:
            if not isinstance(self.risk_snapshot_id, str) or not self.risk_snapshot_id.strip():
                raise HedgeConfigurationError("risk_snapshot_id must not be empty.")
            object.__setattr__(self, "risk_snapshot_id", self.risk_snapshot_id.strip())
        if not isinstance(self.rules_version, str) or not self.rules_version.strip():
            raise HedgeConfigurationError("rules_version must not be empty.")
        object.__setattr__(self, "rules_version", self.rules_version.strip())
        if self.evaluated_at_ms is not None and (
            isinstance(self.evaluated_at_ms, bool)
            or not isinstance(self.evaluated_at_ms, int)
            or self.evaluated_at_ms < 0
        ):
            raise HedgeConfigurationError("evaluated_at_ms must be a nonnegative integer.")
        if not isinstance(self.reason_codes, tuple):
            raise HedgeConfigurationError("Risk decision reason_codes must be a tuple.")
        if not self.reason_codes:
            raise HedgeConfigurationError("Risk decision must contain at least one reason code.")
        normalized_reasons = tuple(
            reason.strip()
            for reason in self.reason_codes
            if isinstance(reason, str) and reason.strip()
        )
        if len(normalized_reasons) != len(self.reason_codes):
            raise HedgeConfigurationError(
                "Risk decision reason_codes must contain non-empty strings only."
            )
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(normalized_reasons)))


def risk_decision_as_dict(decision: RiskDecision) -> dict[str, object]:
    """Serialize a decision without leaking Decimal values into JSON encoders."""

    return {
        "allowed": decision.allowed,
        "approved_quantity": str(decision.approved_quantity),
        "approved_notional": str(decision.approved_notional),
        "reason_codes": list(decision.reason_codes),
        "decision_id": decision.decision_id,
        "risk_snapshot_id": decision.risk_snapshot_id,
        "rules_version": decision.rules_version,
        "evaluated_at_ms": decision.evaluated_at_ms,
    }
