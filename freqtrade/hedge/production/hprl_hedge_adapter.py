"""Production bridge from HPRL dual-leg targets into Clean Mainline planning.

HPRL intentionally owns no exchange write capability.  This module preserves that
boundary: model output is validated as an account exposure target and is converted to
canonical SignalSnapshot / SignalEvent values which are then consumed by the existing
planner, risk and execution stack.

The key semantic rule is explicit unit ownership.  HPRL tier levels are margin-budget
ratios.  The action codec also exposes ``target_notional = target_margin * leverage``.
Production callers must state which of those two units a PlannedExecutionIntent carries;
we never guess from a float.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from typing import TYPE_CHECKING

from freqtrade.hedge.planning.context import PlannerConfig, PlanningContext
from freqtrade.hedge.production.model_targets import (
    ModelTarget,
    ModelTargetDecision,
    ModelTargetPolicy,
    validate_model_target,
)
from freqtrade.hedge.simulation.exchange import SignalEvent

if TYPE_CHECKING:  # avoid importing torch or the HPRL package at module-import time
    from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
    from freqtrade.hedge.integration.signal_provider import SignalSnapshot

ZERO = Decimal("0")
ONE = Decimal("1")


class HprlTargetUnit(StrEnum):
    MARGIN_EQUITY_RATIO = "MARGIN_EQUITY_RATIO"
    NOTIONAL_EQUITY_RATIO = "NOTIONAL_EQUITY_RATIO"


def _d(value: object, *, field: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive normalization
        raise ValueError(f"{field} must be decimal-compatible") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _aware(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class HprlHedgeAdapterPolicy:
    leverage: Decimal = ONE
    target_unit: HprlTargetUnit = HprlTargetUnit.NOTIONAL_EQUITY_RATIO
    max_leg_margin_ratio: Decimal = Decimal("0.40")
    max_gross_margin_ratio: Decimal = Decimal("0.80")
    max_abs_net_margin_ratio: Decimal = Decimal("0.40")
    max_increase_margin_delta: Decimal = Decimal("0.15")
    min_confidence_for_increase: Decimal = Decimal("0.50")
    max_target_age: timedelta = timedelta(seconds=5)

    def __post_init__(self) -> None:
        object.__setattr__(self, "leverage", _d(self.leverage, field="leverage"))
        unit = self.target_unit if isinstance(self.target_unit, HprlTargetUnit) else HprlTargetUnit(self.target_unit)
        object.__setattr__(self, "target_unit", unit)
        for name in (
            "max_leg_margin_ratio",
            "max_gross_margin_ratio",
            "max_abs_net_margin_ratio",
            "max_increase_margin_delta",
            "min_confidence_for_increase",
        ):
            object.__setattr__(self, name, _d(getattr(self, name), field=name))
        if self.leverage <= ZERO:
            raise ValueError("leverage must be positive")
        if not ZERO < self.max_leg_margin_ratio <= ONE:
            raise ValueError("max_leg_margin_ratio must be in (0, 1]")
        if not ZERO < self.max_gross_margin_ratio <= ONE:
            raise ValueError("max_gross_margin_ratio must be in (0, 1]")
        if self.max_gross_margin_ratio < self.max_leg_margin_ratio:
            raise ValueError("max_gross_margin_ratio cannot be below a leg cap")
        if not ZERO <= self.max_abs_net_margin_ratio <= self.max_gross_margin_ratio:
            raise ValueError("max_abs_net_margin_ratio is invalid")
        if not ZERO <= self.max_increase_margin_delta <= ONE:
            raise ValueError("max_increase_margin_delta must be within [0,1]")
        if not ZERO <= self.min_confidence_for_increase <= ONE:
            raise ValueError("min_confidence_for_increase must be within [0,1]")
        if self.max_target_age <= timedelta(0):
            raise ValueError("max_target_age must be positive")

    @classmethod
    def from_hprl_action_config(
        cls,
        config: object,
        *,
        target_unit: HprlTargetUnit = HprlTargetUnit.NOTIONAL_EQUITY_RATIO,
        max_target_age: timedelta = timedelta(seconds=5),
    ) -> "HprlHedgeAdapterPolicy":
        required = (
            "leverage",
            "max_leg_margin_ratio",
            "max_gross_margin_ratio",
            "max_abs_net_margin_ratio",
        )
        missing = [name for name in required if not hasattr(config, name)]
        if missing:
            raise TypeError("HPRL action config missing: " + ",".join(missing))
        levels = tuple(Decimal(str(x)) for x in getattr(config, "position_levels", ()))
        max_delta = max((right - left for left, right in zip(levels, levels[1:], strict=False)), default=Decimal("0.15"))
        increase_levels = int(getattr(config, "max_increase_levels", 1))
        if increase_levels > 1 and levels:
            deltas = []
            for index in range(len(levels)):
                right = min(len(levels) - 1, index + increase_levels)
                deltas.append(levels[right] - levels[index])
            max_delta = max(deltas, default=max_delta)
        return cls(
            leverage=Decimal(str(getattr(config, "leverage"))),
            target_unit=target_unit,
            max_leg_margin_ratio=Decimal(str(getattr(config, "max_leg_margin_ratio"))),
            max_gross_margin_ratio=Decimal(str(getattr(config, "max_gross_margin_ratio"))),
            max_abs_net_margin_ratio=Decimal(str(getattr(config, "max_abs_net_margin_ratio"))),
            max_increase_margin_delta=max_delta,
            max_target_age=max_target_age,
        )

    @property
    def max_leg_notional_ratio(self) -> Decimal:
        return self.max_leg_margin_ratio * self.leverage

    @property
    def max_gross_notional_ratio(self) -> Decimal:
        return self.max_gross_margin_ratio * self.leverage

    @property
    def max_abs_net_notional_ratio(self) -> Decimal:
        return self.max_abs_net_margin_ratio * self.leverage

    def model_policy(self) -> ModelTargetPolicy:
        return ModelTargetPolicy(
            max_age=self.max_target_age,
            max_long_ratio=self.max_leg_margin_ratio,
            max_short_ratio=self.max_leg_margin_ratio,
            max_gross_ratio=self.max_gross_margin_ratio,
            max_abs_net_ratio=self.max_abs_net_margin_ratio,
            max_step_delta_ratio=self.max_increase_margin_delta,
            min_confidence_for_increase=self.min_confidence_for_increase,
            min_risk_budget_multiplier=ONE,
            max_risk_budget_multiplier=ONE,
        )


@dataclass(frozen=True, slots=True)
class HprlTargetProjection:
    sequence: int
    observed_at: datetime
    symbol: str
    model_id: str
    long_margin_ratio: Decimal
    short_margin_ratio: Decimal
    long_notional_ratio: Decimal
    short_notional_ratio: Decimal
    confidence: Decimal
    accepted: bool
    reasons: tuple[str, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be nonnegative")
        object.__setattr__(self, "observed_at", _aware(self.observed_at, field="observed_at"))
        if not self.symbol.strip() or not self.model_id.strip():
            raise ValueError("symbol/model_id are required")
        for name in (
            "long_margin_ratio", "short_margin_ratio", "long_notional_ratio",
            "short_notional_ratio", "confidence",
        ):
            value = _d(getattr(self, name), field=name)
            if value < ZERO:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if self.confidence > ONE:
            raise ValueError("confidence cannot exceed one")
        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("source_sha256 must be SHA-256 hex")
        object.__setattr__(self, "source_sha256", digest)

    @property
    def gross_margin_ratio(self) -> Decimal:
        return self.long_margin_ratio + self.short_margin_ratio

    @property
    def net_margin_ratio(self) -> Decimal:
        return self.long_margin_ratio - self.short_margin_ratio

    @property
    def gross_notional_ratio(self) -> Decimal:
        return self.long_notional_ratio + self.short_notional_ratio

    @property
    def net_notional_ratio(self) -> Decimal:
        return self.long_notional_ratio - self.short_notional_ratio

    @property
    def semantic_sha256(self) -> str:
        payload = {
            "sequence": self.sequence,
            "observed_at": self.observed_at.isoformat(),
            "symbol": self.symbol,
            "model_id": self.model_id,
            "long_margin_ratio": str(self.long_margin_ratio),
            "short_margin_ratio": str(self.short_margin_ratio),
            "long_notional_ratio": str(self.long_notional_ratio),
            "short_notional_ratio": str(self.short_notional_ratio),
            "confidence": str(self.confidence),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "source_sha256": self.source_sha256,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HprlPlannerProfile:
    planner_config: PlannerConfig
    max_leg_notional_ratio: Decimal
    max_gross_notional_ratio: Decimal
    semantic_sha256: str


class HprlHedgeAdapter:
    """Fail-closed HPRL target adapter.

    A rejected target falls back to the previous accepted target (or flat when no prior
    target exists).  The returned projection is still marked ``accepted=False`` and
    callers should disable new risk for that cycle; reductions toward the fallback remain
    possible through the ordinary execution guard.
    """

    def __init__(self, policy: HprlHedgeAdapterPolicy | None = None) -> None:
        self.policy = policy or HprlHedgeAdapterPolicy()

    @staticmethod
    def _source_hash(
        *, sequence: int, observed_at: datetime, symbol: str, model_id: str,
        long: Decimal, short: Decimal, confidence: Decimal, unit: HprlTargetUnit,
    ) -> str:
        payload = {
            "sequence": sequence,
            "observed_at": observed_at.astimezone(UTC).isoformat(),
            "symbol": symbol,
            "model_id": model_id,
            "long": str(long),
            "short": str(short),
            "confidence": str(confidence),
            "unit": unit.value,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _raw_ratios(self, intent: "PlannedExecutionIntent") -> tuple[Decimal, Decimal, Decimal, Decimal]:
        long_raw = _d(intent.target_long_exposure, field="target_long_exposure")
        short_raw = _d(intent.target_short_exposure, field="target_short_exposure")
        if long_raw < ZERO or short_raw < ZERO:
            raise ValueError("HPRL target exposure cannot be negative")
        if self.policy.target_unit is HprlTargetUnit.NOTIONAL_EQUITY_RATIO:
            long_notional, short_notional = long_raw, short_raw
            long_margin = long_notional / self.policy.leverage
            short_margin = short_notional / self.policy.leverage
        else:
            long_margin, short_margin = long_raw, short_raw
            long_notional = long_margin * self.policy.leverage
            short_notional = short_margin * self.policy.leverage
        return long_margin, short_margin, long_notional, short_notional

    def adapt(
        self,
        intent: "PlannedExecutionIntent",
        *,
        sequence: int,
        observed_at: datetime,
        now: datetime,
        previous: HprlTargetProjection | None = None,
    ) -> HprlTargetProjection:
        from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent

        if not isinstance(intent, PlannedExecutionIntent):
            raise TypeError("intent must be PlannedExecutionIntent")
        if sequence < 0:
            raise ValueError("sequence must be nonnegative")
        observed = _aware(observed_at, field="observed_at")
        current = _aware(now, field="now")
        long_margin, short_margin, long_notional, short_notional = self._raw_ratios(intent)
        confidence = _d(intent.confidence, field="confidence")
        model = ModelTarget(
            sequence=sequence,
            observed_at=observed,
            long_ratio=long_margin,
            short_ratio=short_margin,
            confidence=confidence,
            risk_budget_multiplier=ONE,
        )
        previous_model = None
        if previous is not None:
            previous_model = ModelTarget(
                sequence=previous.sequence,
                observed_at=previous.observed_at,
                long_ratio=previous.long_margin_ratio,
                short_ratio=previous.short_margin_ratio,
                confidence=previous.confidence,
                risk_budget_multiplier=ONE,
            )
        decision: ModelTargetDecision = validate_model_target(
            model,
            now=current,
            previous=previous_model,
            policy=self.policy.model_policy(),
        )
        reasons = list(decision.reasons)
        # The generic model-target policy bounds increases symmetrically.  HPRL explicitly
        # allows fast de-risking, so only an *increase* beyond the configured transition
        # envelope is a production violation.
        if previous is not None:
            reasons = [
                reason for reason in reasons
                if not (
                    reason == "MODEL_TARGET_LONG_JUMP"
                    and long_margin <= previous.long_margin_ratio
                )
                and not (
                    reason == "MODEL_TARGET_SHORT_JUMP"
                    and short_margin <= previous.short_margin_ratio
                )
            ]
        accepted = not reasons
        if not accepted:
            if previous is None:
                long_margin = short_margin = long_notional = short_notional = ZERO
            else:
                long_margin = previous.long_margin_ratio
                short_margin = previous.short_margin_ratio
                long_notional = previous.long_notional_ratio
                short_notional = previous.short_notional_ratio
        source_hash = self._source_hash(
            sequence=sequence,
            observed_at=observed,
            symbol=intent.symbol,
            model_id=intent.model_id,
            long=_d(intent.target_long_exposure, field="target_long_exposure"),
            short=_d(intent.target_short_exposure, field="target_short_exposure"),
            confidence=confidence,
            unit=self.policy.target_unit,
        )
        return HprlTargetProjection(
            sequence=sequence,
            observed_at=observed,
            symbol=intent.symbol,
            model_id=intent.model_id,
            long_margin_ratio=long_margin,
            short_margin_ratio=short_margin,
            long_notional_ratio=long_notional,
            short_notional_ratio=short_notional,
            confidence=confidence,
            accepted=accepted,
            reasons=tuple(dict.fromkeys(reasons)),
            source_sha256=source_hash,
        )

    def planner_profile(self, base: PlannerConfig | None = None) -> HprlPlannerProfile:
        base = base or PlannerConfig()
        leg = self.policy.max_leg_notional_ratio
        gross = self.policy.max_gross_notional_ratio
        config = replace(
            base,
            long_enabled=True,
            short_enabled=True,
            core_wallet_exposure_long=leg,
            core_wallet_exposure_short=leg,
            tactical_wallet_exposure_long=ZERO,
            tactical_wallet_exposure_short=ZERO,
            max_wallet_exposure_long=leg,
            max_wallet_exposure_short=leg,
            max_gross_wallet_exposure=gross,
            target_net_wallet_exposure=ZERO,
        )
        payload = {
            "leverage": str(self.policy.leverage),
            "max_leg_margin_ratio": str(self.policy.max_leg_margin_ratio),
            "max_gross_margin_ratio": str(self.policy.max_gross_margin_ratio),
            "max_abs_net_margin_ratio": str(self.policy.max_abs_net_margin_ratio),
            "target_unit": self.policy.target_unit.value,
            "planner": repr(config),
        }
        digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return HprlPlannerProfile(config, leg, gross, digest)

    def _scales(self, projection: HprlTargetProjection) -> tuple[Decimal, Decimal]:
        cap = self.policy.max_leg_notional_ratio
        if cap <= ZERO:  # pragma: no cover - policy rejects this
            raise ValueError("invalid HPRL leg cap")
        long_scale = projection.long_notional_ratio / cap
        short_scale = projection.short_notional_ratio / cap
        if not ZERO <= long_scale <= ONE or not ZERO <= short_scale <= ONE:
            raise ValueError("projection exceeds configured HPRL planner profile")
        return long_scale, short_scale

    def to_signal_event(
        self,
        projection: HprlTargetProjection,
        *,
        allow_new_risk: bool | None = None,
    ) -> SignalEvent:
        long_scale, short_scale = self._scales(projection)
        allowed = projection.accepted if allow_new_risk is None else bool(allow_new_risk and projection.accepted)
        # Confidence is diagnostic here, not a second sizing multiplier.  HPRL already chose
        # the hard tier.  Applying confidence again would silently move the target off-grid.
        return SignalEvent(
            timestamp=projection.observed_at,
            symbol=projection.symbol,
            long_signal=long_scale,
            short_signal=short_scale,
            target_net=None,
            target_net_ratio=None,
            confidence=ONE,
            risk_scale=ONE,
            long_exposure_scale=long_scale,
            short_exposure_scale=short_scale,
            allow_new_risk=allowed,
            regime="HPRL",
            reason=(
                "HPRL_EXACT_DUAL_LEG"
                if projection.accepted
                else "HPRL_TARGET_REJECTED:" + ",".join(projection.reasons)
            )[:256],
            model_version=projection.model_id[:128],
        )

    def signal_snapshot_kwargs(
        self,
        projection: HprlTargetProjection,
        *,
        timeframe: str,
        candle_close_time: datetime,
        feature_timestamp: datetime,
        allow_new_risk: bool | None = None,
    ) -> dict[str, object]:
        """Build the canonical live SignalSnapshot payload without importing Freqtrade.

        Keeping this pure makes the model/planner semantic gate runnable in a minimal
        source-validation environment.  ``to_signal_snapshot`` performs the actual
        runtime import when the complete Freqtrade dependency set is present.
        """
        if not timeframe.strip():
            raise ValueError("timeframe cannot be empty")
        long_scale, short_scale = self._scales(projection)
        allowed = projection.accepted if allow_new_risk is None else bool(allow_new_risk and projection.accepted)
        return {
            "symbol": projection.symbol,
            "timeframe": timeframe,
            "candle_close_time": _aware(candle_close_time, field="candle_close_time"),
            "feature_timestamp": _aware(feature_timestamp, field="feature_timestamp"),
            "long_score": long_scale,
            "short_score": short_scale,
            "target_net": None,
            "target_net_ratio": None,
            "confidence": ONE,
            "risk_scale": ONE,
            "long_exposure_scale": long_scale,
            "short_exposure_scale": short_scale,
            "allow_new_risk": allowed,
            "regime": "HPRL",
            "reason": (
                "HPRL_EXACT_DUAL_LEG"
                if projection.accepted
                else "HPRL_TARGET_REJECTED:" + ",".join(projection.reasons)
            )[:256],
            "strategy_reason": "HPRL_V3_PRODUCTION_ADAPTER",
            "model_version": projection.model_id[:128],
        }

    def to_signal_snapshot(
        self,
        projection: HprlTargetProjection,
        *,
        timeframe: str,
        candle_close_time: datetime,
        feature_timestamp: datetime,
        allow_new_risk: bool | None = None,
    ) -> "SignalSnapshot":
        from freqtrade.hedge.integration.signal_provider import SignalSnapshot

        return SignalSnapshot(
            **self.signal_snapshot_kwargs(
                projection,
                timeframe=timeframe,
                candle_close_time=candle_close_time,
                feature_timestamp=feature_timestamp,
                allow_new_risk=allow_new_risk,
            )
        )

    def apply_to_context(
        self,
        context: PlanningContext,
        projection: HprlTargetProjection,
        *,
        allow_new_risk: bool | None = None,
    ) -> tuple[PlanningContext, str]:
        """Apply one projected HPRL target using the exact live/replay sizing semantics.

        HPRL tiers are already risk-projected.  This bridge does not quantize again and
        does not collapse LONG/SHORT into a net target.  It converts the planner profile
        to notional/equity units once, then uses the same StrategyDirective scaling path
        consumed by replay and the live planning-context builder.
        """
        from freqtrade.hedge.strategies.contract import (
            StrategyDirective,
            planner_config_for_directive,
        )
        from freqtrade.hedge.symbols import raw_symbol

        if not isinstance(context, PlanningContext):
            raise TypeError("context must be PlanningContext")
        if raw_symbol(context.market.symbol) != raw_symbol(projection.symbol):
            raise ValueError("HPRL projection symbol does not match planning context")
        if context.wallet.leverage != self.policy.leverage:
            raise ValueError(
                "planning wallet leverage does not match HPRL production policy"
            )
        profile = self.planner_profile(context.config)
        event = self.to_signal_event(projection, allow_new_risk=allow_new_risk)
        directive = StrategyDirective(
            long_score=event.long_signal,
            short_score=event.short_signal,
            target_net_quantity=event.target_net,
            target_net_ratio=event.target_net_ratio,
            confidence=event.confidence,
            risk_scale=event.risk_scale,
            long_exposure_scale=event.long_exposure_scale,
            short_exposure_scale=event.short_exposure_scale,
            allow_new_risk=event.allow_new_risk,
            regime=event.regime,
            reason=event.reason,
            model_version=event.model_version,
        )
        effective = planner_config_for_directive(profile.planner_config, directive)
        adapted = replace(
            context,
            config=effective,
            long_signal=directive.long_score,
            short_signal=directive.short_score,
            target_net_quantity=None,
        )
        return adapted, profile.semantic_sha256
