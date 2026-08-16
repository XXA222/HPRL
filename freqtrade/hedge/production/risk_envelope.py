"""Account-level Cross Margin post-trade risk envelope."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import StrEnum

from .contracts import Decision

ZERO = Decimal("0")
ONE = Decimal("1")


def D(value: Decimal | str | int | float) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("risk numeric value must be finite")
    return result


class Side(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class RiskDirection(StrEnum):
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"


@dataclass(frozen=True, slots=True)
class AccountRiskView:
    equity: Decimal
    available_balance: Decimal
    long_notional: Decimal
    short_notional: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    pending_long_notional: Decimal = ZERO
    pending_short_notional: Decimal = ZERO
    pending_long_reduce_notional: Decimal = ZERO
    pending_short_reduce_notional: Decimal = ZERO
    funding_reserve: Decimal = ZERO
    fee_slippage_reserve: Decimal = ZERO

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = D(getattr(self, name))
            if value < ZERO:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.equity <= ZERO:
            raise ValueError("equity must be positive")
        if self.pending_long_reduce_notional > self.long_notional:
            raise ValueError("pending_long_reduce_notional exceeds executed LONG position")
        if self.pending_short_reduce_notional > self.short_notional:
            raise ValueError("pending_short_reduce_notional exceeds executed SHORT position")

    @property
    def gross(self) -> Decimal:
        return self.long_notional + self.short_notional

    @property
    def net(self) -> Decimal:
        return self.long_notional - self.short_notional


@dataclass(frozen=True, slots=True)
class CandidateIntent:
    side: Side
    direction: RiskDirection
    notional: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal

    def __post_init__(self) -> None:
        for name in ("notional", "initial_margin", "maintenance_margin"):
            value = D(getattr(self, name))
            if value < ZERO:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.initial_margin > self.notional or self.maintenance_margin > self.notional:
            raise ValueError("margin cannot exceed intent notional")


@dataclass(frozen=True, slots=True)
class CrossRiskLimits:
    max_gross_exposure_ratio: Decimal = Decimal("1.50")
    max_abs_net_exposure_ratio: Decimal = Decimal("0.75")
    max_margin_utilization: Decimal = Decimal("0.65")
    min_maintenance_buffer_ratio: Decimal = Decimal("0.25")
    min_available_balance_ratio: Decimal = Decimal("0.15")
    max_single_intent_ratio: Decimal = Decimal("0.20")
    clip_quantum: Decimal = Decimal("0.01")
    max_stress_loss_ratio: Decimal = Decimal("0.20")
    min_stressed_equity_ratio: Decimal = Decimal("0.70")

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = D(getattr(self, name))
            if value <= ZERO:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PostTradeProjection:
    long_notional: Decimal
    short_notional: Decimal
    gross_ratio: Decimal
    net_ratio: Decimal
    margin_utilization: Decimal
    maintenance_buffer_ratio: Decimal
    available_balance_ratio: Decimal


@dataclass(frozen=True, slots=True)
class StressScenario:
    """Adverse account scenario expressed as loss fractions of each side notional."""

    name: str
    long_loss_ratio: Decimal
    short_loss_ratio: Decimal
    extra_cost_ratio: Decimal = ZERO

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stress scenario name is required")
        for field_name in ("long_loss_ratio", "short_loss_ratio", "extra_cost_ratio"):
            value = D(getattr(self, field_name))
            if value < ZERO or value > ONE:
                raise ValueError(f"{field_name} must be in [0,1]")
            object.__setattr__(self, field_name, value)


DEFAULT_STRESS_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario("LONG_DOWN_12PCT", Decimal("0.12"), ZERO, Decimal("0.002")),
    StressScenario("SHORT_UP_12PCT", ZERO, Decimal("0.12"), Decimal("0.002")),
    StressScenario("BOTH_LEGS_COST_5PCT", Decimal("0.05"), Decimal("0.05"), Decimal("0.003")),
)


@dataclass(frozen=True, slots=True)
class StressResult:
    scenario: str
    loss: Decimal
    stressed_equity: Decimal
    loss_ratio: Decimal
    stressed_equity_ratio: Decimal
    passed: bool


@dataclass(frozen=True, slots=True)
class RiskEnvelopeDecision:
    decision: Decision
    reasons: tuple[str, ...]
    requested_notional: Decimal
    approved_notional: Decimal
    projection: PostTradeProjection
    stress_results: tuple[StressResult, ...] = ()

    @property
    def risk_reducing(self) -> bool:
        return self.decision in {Decision.APPROVE, Decision.CLIP} and not self.reasons


def _project(view: AccountRiskView, intent: CandidateIntent, notional: Decimal) -> PostTradeProjection:
    scale = ZERO if intent.notional == ZERO else notional / intent.notional
    long_n = view.long_notional + view.pending_long_notional
    short_n = view.short_notional + view.pending_short_notional
    signed = notional if intent.direction is RiskDirection.INCREASE else -notional
    if intent.side is Side.LONG:
        long_n = max(ZERO, long_n + signed)
    else:
        short_n = max(ZERO, short_n + signed)
    margin_delta = intent.initial_margin * scale
    maintenance_delta = intent.maintenance_margin * scale
    if intent.direction is RiskDirection.REDUCE:
        margin_delta = -margin_delta
        maintenance_delta = -maintenance_delta
    projected_initial = max(ZERO, view.initial_margin + margin_delta)
    projected_maintenance = max(ZERO, view.maintenance_margin + maintenance_delta)
    reserved = view.funding_reserve + view.fee_slippage_reserve
    available = max(ZERO, view.available_balance - max(margin_delta, ZERO) - reserved)
    return PostTradeProjection(
        long_notional=long_n,
        short_notional=short_n,
        gross_ratio=(long_n + short_n) / view.equity,
        net_ratio=(long_n - short_n) / view.equity,
        margin_utilization=projected_initial / view.equity,
        maintenance_buffer_ratio=max(ZERO, (view.equity - projected_maintenance - reserved) / view.equity),
        available_balance_ratio=available / view.equity,
    )


def _violations(proj: PostTradeProjection, limits: CrossRiskLimits) -> tuple[str, ...]:
    out: list[str] = []
    if proj.gross_ratio > limits.max_gross_exposure_ratio:
        out.append("GROSS_EXPOSURE_LIMIT")
    if abs(proj.net_ratio) > limits.max_abs_net_exposure_ratio:
        out.append("NET_EXPOSURE_LIMIT")
    if proj.margin_utilization > limits.max_margin_utilization:
        out.append("MARGIN_UTILIZATION_LIMIT")
    if proj.maintenance_buffer_ratio < limits.min_maintenance_buffer_ratio:
        out.append("MAINTENANCE_BUFFER_LIMIT")
    if proj.available_balance_ratio < limits.min_available_balance_ratio:
        out.append("AVAILABLE_BALANCE_LIMIT")
    return tuple(out)


def _stress_results(
    view: AccountRiskView,
    projection: PostTradeProjection,
    limits: CrossRiskLimits,
    scenarios: tuple[StressScenario, ...],
) -> tuple[StressResult, ...]:
    results: list[StressResult] = []
    for scenario in scenarios:
        gross = projection.long_notional + projection.short_notional
        loss = (
            projection.long_notional * scenario.long_loss_ratio
            + projection.short_notional * scenario.short_loss_ratio
            + gross * scenario.extra_cost_ratio
            + view.funding_reserve
            + view.fee_slippage_reserve
        )
        stressed_equity = max(ZERO, view.equity - loss)
        loss_ratio = loss / view.equity
        equity_ratio = stressed_equity / view.equity
        passed = (
            loss_ratio <= limits.max_stress_loss_ratio
            and equity_ratio >= limits.min_stressed_equity_ratio
        )
        results.append(
            StressResult(
                scenario.name,
                loss,
                stressed_equity,
                loss_ratio,
                equity_ratio,
                passed,
            )
        )
    return tuple(results)


def evaluate_post_trade_risk(
    view: AccountRiskView,
    intent: CandidateIntent,
    limits: CrossRiskLimits,
    *,
    stress_scenarios: tuple[StressScenario, ...] = DEFAULT_STRESS_SCENARIOS,
) -> RiskEnvelopeDecision:
    requested = intent.notional
    current_gross = view.gross
    requested_projection = _project(view, intent, requested)
    requested_violations = _violations(requested_projection, limits)

    # Controlled reduction is bounded to the actually closeable side so the risk layer
    # itself cannot authorize a position flip and then rely on Execution to save it.
    if intent.direction is RiskDirection.REDUCE:
        # Only *executed* position notional is closeable.  Pending increases belong in
        # the worst-case exposure projection, but they may never fill and therefore must
        # never expand the quantity a reduce order is allowed to close.
        if intent.side is Side.LONG:
            closeable = max(
                ZERO, view.long_notional - view.pending_long_reduce_notional
            )
        else:
            closeable = max(
                ZERO, view.short_notional - view.pending_short_reduce_notional
            )
        approved = min(requested, closeable)
        projection = _project(view, intent, approved)
        if approved <= ZERO:
            return RiskEnvelopeDecision(
                Decision.REJECT,
                ("NO_CLOSEABLE_POSITION",),
                requested,
                ZERO,
                projection,
            )
        if projection.long_notional + projection.short_notional <= current_gross + view.pending_long_notional + view.pending_short_notional:
            decision = Decision.APPROVE if approved == requested else Decision.CLIP
            reasons = () if decision is Decision.APPROVE else ("REDUCE_CLIPPED_TO_CLOSEABLE",)
            return RiskEnvelopeDecision(
                decision, reasons, requested, approved, projection
            )
        return RiskEnvelopeDecision(
            Decision.REJECT,
            ("REDUCE_ACTION_INCREASES_GROSS",),
            requested,
            ZERO,
            requested_projection,
        )

    single_cap = view.equity * limits.max_single_intent_ratio
    capped_request = min(requested, single_cap)
    requested_stress = _stress_results(view, requested_projection, limits, stress_scenarios)
    stress_violations = tuple(
        f"STRESS_LIMIT:{item.scenario}" for item in requested_stress if not item.passed
    )
    if not requested_violations and not stress_violations and requested <= single_cap:
        return RiskEnvelopeDecision(
            Decision.APPROVE, (), requested, requested, requested_projection, requested_stress
        )

    # Monotone binary search for the largest safe notional.  Fixed iterations preserve
    # deterministic Decimal behavior and avoid data-dependent runaway loops.
    low, high = ZERO, capped_request
    safe = ZERO
    for _ in range(48):
        mid = (low + high) / Decimal("2")
        mid_projection = _project(view, intent, mid)
        violations = _violations(mid_projection, limits)
        stress_bad = any(
            not item.passed
            for item in _stress_results(view, mid_projection, limits, stress_scenarios)
        )
        if violations or stress_bad:
            high = mid
        else:
            safe = mid
            low = mid
    quantum = limits.clip_quantum
    safe = (safe / quantum).to_integral_value(rounding=ROUND_DOWN) * quantum
    projection = _project(view, intent, safe)
    stress_results = _stress_results(view, projection, limits, stress_scenarios)
    if safe > ZERO:
        reasons = tuple(
            dict.fromkeys(
                (*requested_violations, *stress_violations, "INTENT_CLIPPED_TO_RISK_ENVELOPE")
            )
        )
        return RiskEnvelopeDecision(
            Decision.CLIP, reasons, requested, safe, projection, stress_results
        )
    reasons = requested_violations or stress_violations or ("NO_SAFE_NOTIONAL",)
    return RiskEnvelopeDecision(
        Decision.REJECT, reasons, requested, ZERO, projection, stress_results
    )
