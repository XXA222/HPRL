"""Side-aware account risk engine."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.local_reduce_only import calculate_safe_reduce
from freqtrade.hedge.numeric import require_nonnegative, require_positive
from freqtrade.hedge.risk.limits import RiskLimits
from freqtrade.hedge.risk.models import AccountRiskSnapshot, RiskDecision, RiskRequest

if TYPE_CHECKING:
    from freqtrade.hedge.risk.portfolio import RiskPortfolioSnapshot


_LEGACY_POSITION_SIDE_UNSET = object()


class HedgeRiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def evaluate_request(
        self,
        *,
        request: RiskRequest,
        account: AccountRiskSnapshot,
    ) -> RiskDecision:
        if request.account_id != account.account_id:
            return RiskDecision(
                False,
                Decimal("0"),
                ("ACCOUNT_ID_MISMATCH",),
                Decimal("0"),
            )
        if request.action in {PositionAction.REDUCE, PositionAction.CLOSE}:
            return self._evaluate_reduce(request)
        return self._evaluate_increase(request, account)

    def evaluate_portfolio_order(
        self,
        *,
        portfolio: "RiskPortfolioSnapshot",
        symbol: str,
        position_side: PositionSide | str,
        action: PositionAction | str,
        requested_quantity: Decimal,
        reference_price: Decimal,
        leverage: Decimal = Decimal("1"),
        maintenance_margin_rate: Decimal = Decimal("0.01"),
    ) -> RiskDecision:
        """Build and evaluate a request from a complete portfolio snapshot."""

        request = portfolio.build_request(
            symbol=symbol,
            position_side=position_side,
            action=action,
            requested_quantity=requested_quantity,
            reference_price=reference_price,
            leverage=leverage,
            maintenance_margin_rate=maintenance_margin_rate,
        )
        return self.evaluate_request(request=request, account=portfolio.account)

    def evaluate(
        self,
        *,
        action: PositionAction,
        requested_quantity: Decimal,
        reference_price: Decimal,
        account: AccountRiskSnapshot,
        symbol: str = "UNKNOWN",
        position_side: PositionSide | str | object = _LEGACY_POSITION_SIDE_UNSET,
        current_leg_notional: Decimal = Decimal("0"),
        current_symbol_gross_notional: Decimal = Decimal("0"),
        pending_leg_increase_notional: Decimal = Decimal("0"),
        pending_symbol_increase_notional: Decimal = Decimal("0"),
        confirmed_quantity: Decimal | None = None,
        pending_reduce_quantity: Decimal = Decimal("0"),
        leverage: Decimal = Decimal("1"),
        maintenance_margin_rate: Decimal = Decimal("0.01"),
    ) -> RiskDecision:
        """Compatibility entry point plus the direction-three fields."""

        legacy_unscoped_call = position_side is _LEGACY_POSITION_SIDE_UNSET
        resolved_position_side = (
            PositionSide.LONG if legacy_unscoped_call else position_side
        )
        resolved_confirmed_quantity = confirmed_quantity
        if (
            legacy_unscoped_call
            and action in {PositionAction.REDUCE, PositionAction.CLOSE}
            and resolved_confirmed_quantity is None
        ):
            # P2-H2 exposed an account-level compatibility helper before the
            # side-aware confirmed-position contract existed.  Keep that old
            # call shape risk-reducing and margin-independent, while all modern
            # side-aware calls and evaluate_request() continue to fail closed
            # unless confirmed quantity is supplied.
            resolved_confirmed_quantity = requested_quantity

        request = RiskRequest(
            account_id=account.account_id,
            symbol=symbol,
            position_side=resolved_position_side,
            action=action,
            requested_quantity=requested_quantity,
            reference_price=reference_price,
            current_leg_notional=current_leg_notional,
            current_symbol_gross_notional=current_symbol_gross_notional,
            pending_leg_increase_notional=pending_leg_increase_notional,
            pending_symbol_increase_notional=pending_symbol_increase_notional,
            confirmed_quantity=resolved_confirmed_quantity,
            pending_reduce_quantity=pending_reduce_quantity,
            leverage=leverage,
            maintenance_margin_rate=maintenance_margin_rate,
        )
        return self.evaluate_request(request=request, account=account)

    def _evaluate_reduce(self, request: RiskRequest) -> RiskDecision:
        if request.confirmed_quantity is None:
            return RiskDecision(
                False,
                Decimal("0"),
                ("CONFIRMED_POSITION_REQUIRED",),
                Decimal("0"),
            )
        safe = calculate_safe_reduce(
            requested_quantity=request.requested_quantity,
            confirmed_quantity=request.confirmed_quantity,
            pending_reduce_quantity=request.pending_reduce_quantity,
        )
        approved = safe.allowed_quantity
        reason = safe.reason_code
        if approved <= 0:
            return RiskDecision(False, Decimal("0"), (reason,), Decimal("0"))
        approved = min(approved, request.requested_quantity)
        return RiskDecision(
            True,
            approved,
            (reason,),
            approved * request.reference_price,
        )

    def _evaluate_increase(
        self,
        request: RiskRequest,
        account: AccountRiskSnapshot,
    ) -> RiskDecision:
        if not account.effective_risk_data_valid:
            reasons = tuple(account.risk_data_errors) or ("RISK_DATA_INVALID",)
            return RiskDecision(
                False,
                Decimal("0"),
                reasons,
                Decimal("0"),
                risk_snapshot_id=account.snapshot_id,
            )
        if account.liquidation_buffer_ratio < self._limits.min_liquidation_buffer_ratio:
            return RiskDecision(
                False,
                Decimal("0"),
                ("LIQUIDATION_BUFFER_LOW",),
                Decimal("0"),
                risk_snapshot_id=account.snapshot_id,
            )
        if account.projected_maintenance_buffer_ratio < self._limits.min_liquidation_buffer_ratio:
            return RiskDecision(
                False,
                Decimal("0"),
                ("PENDING_LIQUIDATION_BUFFER_LOW",),
                Decimal("0"),
                risk_snapshot_id=account.snapshot_id,
            )

        approved_notional = request.requested_notional
        reasons: list[str] = []
        minimum_net_remediation = Decimal("0")
        net_remediation_reasons: list[str] = []

        def clip(capacity: Decimal, reason_code: str) -> None:
            nonlocal approved_notional
            capacity = require_nonnegative(capacity, field=reason_code.lower())
            if approved_notional > capacity:
                approved_notional = capacity
                reasons.append(reason_code)

        if self._limits.max_single_order_notional is not None:
            clip(self._limits.max_single_order_notional, "SINGLE_ORDER_NOTIONAL_CLIPPED")

        margin_headroom = max(
            self._limits.max_margin_utilization * account.equity
            - account.initial_margin
            - account.pending_order_initial_margin,
            Decimal("0"),
        )
        clip(margin_headroom * request.leverage, "MARGIN_UTILIZATION_CLIPPED")

        available_margin = max(
            account.available_balance - self._limits.min_available_balance,
            Decimal("0"),
        )
        clip(available_margin * request.leverage, "AVAILABLE_MARGIN_CLIPPED")

        projected_maintenance_capacity = max(
            account.equity * (Decimal("1") - self._limits.min_liquidation_buffer_ratio)
            - account.maintenance_margin
            - account.pending_order_maintenance_margin,
            Decimal("0"),
        )
        clip(
            projected_maintenance_capacity / request.maintenance_margin_rate,
            "PROJECTED_LIQUIDATION_BUFFER_CLIPPED",
        )

        if self._limits.max_pending_order_notional is not None:
            clip(
                max(
                    self._limits.max_pending_order_notional - account.pending_order_notional,
                    Decimal("0"),
                ),
                "PENDING_ORDER_NOTIONAL_CLIPPED",
            )
        if self._limits.max_pending_order_initial_margin is not None:
            clip(
                max(
                    self._limits.max_pending_order_initial_margin
                    - account.pending_order_initial_margin,
                    Decimal("0"),
                )
                * request.leverage,
                "PENDING_ORDER_MARGIN_CLIPPED",
            )

        if self._limits.max_leg_notional is not None:
            clip(
                max(
                    self._limits.max_leg_notional
                    - request.current_leg_notional
                    - request.pending_leg_increase_notional,
                    Decimal("0"),
                ),
                "LEG_NOTIONAL_CLIPPED",
            )

        if self._limits.max_symbol_gross_notional is not None:
            clip(
                max(
                    self._limits.max_symbol_gross_notional
                    - request.current_symbol_gross_notional
                    - request.pending_symbol_increase_notional,
                    Decimal("0"),
                ),
                "SYMBOL_GROSS_NOTIONAL_CLIPPED",
            )

        if self._limits.max_gross_notional is not None:
            clip(
                max(
                    self._limits.max_gross_notional
                    - account.gross_total_notional
                    - account.pending_order_notional,
                    Decimal("0"),
                ),
                "GROSS_NOTIONAL_CLIPPED",
            )

        if self._limits.max_gross_exposure_ratio is not None:
            ratio_notional_limit = self._limits.max_gross_exposure_ratio * account.equity
            clip(
                max(
                    ratio_notional_limit
                    - account.gross_total_notional
                    - account.pending_order_notional,
                    Decimal("0"),
                ),
                "GROSS_EXPOSURE_RATIO_CLIPPED",
            )

        if request.position_side is PositionSide.LONG:
            side_current = account.projected_gross_long_notional
            side_notional_limit = self._limits.max_long_notional
            side_ratio_limit = self._limits.max_long_exposure_ratio
            side_reason = "LONG_NOTIONAL_CLIPPED"
            side_ratio_reason = "LONG_EXPOSURE_RATIO_CLIPPED"
        else:
            side_current = account.projected_gross_short_notional
            side_notional_limit = self._limits.max_short_notional
            side_ratio_limit = self._limits.max_short_exposure_ratio
            side_reason = "SHORT_NOTIONAL_CLIPPED"
            side_ratio_reason = "SHORT_EXPOSURE_RATIO_CLIPPED"
        if side_notional_limit is not None:
            clip(max(side_notional_limit - side_current, Decimal("0")), side_reason)
        if side_ratio_limit is not None:
            clip(
                max(side_ratio_limit * account.equity - side_current, Decimal("0")),
                side_ratio_reason,
            )

        def apply_net_limit(
            limit: Decimal,
            *,
            clipped_reason: str,
            remediation_reason: str,
        ) -> None:
            nonlocal minimum_net_remediation
            signed_current = request.net_notional_sign * account.projected_net_notional
            minimum_required = max(-limit - signed_current, Decimal("0"))
            maximum_allowed = max(limit - signed_current, Decimal("0"))
            minimum_net_remediation = max(minimum_net_remediation, minimum_required)
            if minimum_required > 0:
                net_remediation_reasons.append(remediation_reason)
            clip(maximum_allowed, clipped_reason)

        if self._limits.max_net_notional is not None:
            apply_net_limit(
                self._limits.max_net_notional,
                clipped_reason="NET_NOTIONAL_CLIPPED",
                remediation_reason="NET_NOTIONAL_REMEDIATION_INSUFFICIENT",
            )

        if self._limits.max_net_exposure_ratio is not None:
            apply_net_limit(
                self._limits.max_net_exposure_ratio * account.equity,
                clipped_reason="NET_EXPOSURE_RATIO_CLIPPED",
                remediation_reason="NET_EXPOSURE_REMEDIATION_INSUFFICIENT",
            )

        if approved_notional < minimum_net_remediation:
            return RiskDecision(
                False,
                Decimal("0"),
                tuple(dict.fromkeys([*reasons, *net_remediation_reasons])),
                Decimal("0"),
                risk_snapshot_id=account.snapshot_id,
            )

        price = require_positive(request.reference_price, field="reference_price")
        approved_quantity = min(request.requested_quantity, approved_notional / price)
        if approved_quantity <= 0:
            return RiskDecision(
                False,
                Decimal("0"),
                tuple(reasons or ["NO_RISK_CAPACITY"]),
                Decimal("0"),
                risk_snapshot_id=account.snapshot_id,
            )
        return RiskDecision(
            True,
            approved_quantity,
            tuple(reasons or ["WITHIN_LIMITS"]),
            approved_quantity * price,
            risk_snapshot_id=account.snapshot_id,
        )
