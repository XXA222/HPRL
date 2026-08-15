"""Account-level action masking and safety constraints for Hedge agents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .actions import DEFAULT_ACTION_CATALOG, HedgeActionCatalog, HedgeActions, LegCommand
from .config import HedgeRLConfig
from .state import HedgeAccountState, HedgeLegState


@dataclass(frozen=True, slots=True)
class ActionSafetyDecision:
    allowed: bool
    reasons: tuple[str, ...]
    projected_long_exposure: float
    projected_short_exposure: float
    projected_gross_exposure: float
    projected_net_exposure: float


class HedgeActionMasker:
    def __init__(
        self,
        config: HedgeRLConfig,
        catalog: HedgeActionCatalog = DEFAULT_ACTION_CATALOG,
    ) -> None:
        self.config = config
        self.catalog = catalog

    @staticmethod
    def _project_leg_exposure(
        leg: HedgeLegState,
        command: LegCommand,
        fraction: float,
        *,
        mark: float,
        equity: float,
    ) -> float:
        current = leg.notional(mark) / max(equity, 1e-12)
        if command in {LegCommand.OPEN, LegCommand.INCREASE}:
            return current + fraction
        if command is LegCommand.REDUCE:
            return current * (1.0 - fraction)
        if command is LegCommand.CLOSE:
            return 0.0
        return current

    @staticmethod
    def _leg_semantic_reasons(leg: HedgeLegState, command: LegCommand, label: str) -> list[str]:
        if command is LegCommand.OPEN and leg.quantity > 0:
            return [f"{label}_OPEN_REQUIRES_FLAT"]
        if command is LegCommand.INCREASE and leg.quantity <= 0:
            return [f"{label}_INCREASE_REQUIRES_POSITION"]
        if command in {LegCommand.REDUCE, LegCommand.CLOSE} and leg.quantity <= 0:
            return [f"{label}_{command.value}_REQUIRES_POSITION"]
        return []

    def evaluate(
        self,
        action: int,
        *,
        account: HedgeAccountState,
        mark: float,
    ) -> ActionSafetyDecision:
        spec = self.catalog.decode(action)
        composite_reduce = spec.action in {
            HedgeActions.BOTH_REDUCE_SMALL,
            HedgeActions.CLOSE_BOTH,
            HedgeActions.EMERGENCY_REDUCE_BOTH,
        }
        if composite_reduce:
            # A close-both/emergency command must remain available when only one
            # leg exists.  The flat leg is a safe no-op in the simulator/executor.
            reasons = []
            if account.long.quantity <= 0 and account.short.quantity <= 0:
                reasons.append("COMPOSITE_REDUCE_REQUIRES_ANY_POSITION")
        else:
            reasons = self._leg_semantic_reasons(account.long, spec.long_command, "LONG")
            reasons += self._leg_semantic_reasons(account.short, spec.short_command, "SHORT")
        equity = max(account.equity, 1e-12)
        long_exposure = self._project_leg_exposure(
            account.long,
            spec.long_command,
            spec.long_fraction,
            mark=mark,
            equity=equity,
        )
        short_exposure = self._project_leg_exposure(
            account.short,
            spec.short_command,
            spec.short_fraction,
            mark=mark,
            equity=equity,
        )
        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
        risk_reducing = spec.long_command in {
            LegCommand.HOLD,
            LegCommand.REDUCE,
            LegCommand.CLOSE,
        } and (
            spec.short_command in {LegCommand.HOLD, LegCommand.REDUCE, LegCommand.CLOSE}
        )
        if account.equity <= 0 and not risk_reducing:
            reasons.append("NONPOSITIVE_EQUITY")
        if not risk_reducing:
            if long_exposure > self.config.max_side_exposure + 1e-12:
                reasons.append("LONG_SIDE_EXPOSURE_LIMIT")
            if short_exposure > self.config.max_side_exposure + 1e-12:
                reasons.append("SHORT_SIDE_EXPOSURE_LIMIT")
            if gross > self.config.max_gross_exposure + 1e-12:
                reasons.append("GROSS_EXPOSURE_LIMIT")
            if abs(net) > self.config.max_net_exposure + 1e-12:
                reasons.append("NET_EXPOSURE_LIMIT")
        return ActionSafetyDecision(
            allowed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            projected_long_exposure=long_exposure,
            projected_short_exposure=short_exposure,
            projected_gross_exposure=gross,
            projected_net_exposure=net,
        )

    def mask(self, *, account: HedgeAccountState, mark: float) -> npt.NDArray[np.bool_]:
        return np.asarray(
            [
                self.evaluate(int(spec.action), account=account, mark=mark).allowed
                for spec in self.catalog.specs()
            ],
            dtype=np.bool_,
        )
