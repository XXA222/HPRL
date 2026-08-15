"""Hedge-native Hyperopt parameter contracts and multi-objective loss.

This module is Optuna-compatible but keeps imports optional so source audit and dry-run
installations do not require the Hyperopt dependency graph at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
from json import dumps
from typing import Any, Callable, Iterable, Mapping, Protocol

from .backtest import HedgeBacktestArtifact, HedgeBacktestMetrics
from .models import ONE, ZERO, finite_decimal


class TrialLike(Protocol):
    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float: ...
    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int: ...
    def suggest_categorical(self, name: str, choices: Iterable[Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class DecimalSpace:
    name: str
    low: Decimal
    high: Decimal
    step: Decimal | None = None
    log: bool = False

    def __post_init__(self) -> None:
        low = finite_decimal(self.low, field_name=f"{self.name}.low")
        high = finite_decimal(self.high, field_name=f"{self.name}.high")
        if low >= high:
            raise ValueError(f"{self.name}: low must be below high")
        step = None if self.step is None else finite_decimal(self.step, field_name=f"{self.name}.step")
        if step is not None and step <= ZERO:
            raise ValueError(f"{self.name}: step must be positive")
        if self.log and (low <= ZERO or step is not None):
            raise ValueError(f"{self.name}: logarithmic space requires positive bounds and no step")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "step", step)

    def suggest(self, trial: TrialLike) -> Decimal:
        kwargs: dict[str, Any] = {"log": self.log}
        if self.step is not None:
            kwargs["step"] = float(self.step)
        return Decimal(str(trial.suggest_float(self.name, float(self.low), float(self.high), **kwargs)))


@dataclass(frozen=True, slots=True)
class IntegerSpace:
    name: str
    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.low, bool) or self.low > self.high or self.step <= 0:
            raise ValueError(f"invalid integer space {self.name}")

    def suggest(self, trial: TrialLike) -> int:
        return int(trial.suggest_int(self.name, self.low, self.high, step=self.step))


@dataclass(frozen=True, slots=True)
class CategoricalSpace:
    name: str
    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"{self.name}: categorical choices cannot be empty")

    def suggest(self, trial: TrialLike) -> Any:
        return trial.suggest_categorical(self.name, self.choices)


Space = DecimalSpace | IntegerSpace | CategoricalSpace


@dataclass(frozen=True, slots=True)
class HedgeHyperoptSpace:
    """Default search space for planner, exits, gross/net exposure and execution."""

    spaces: tuple[Space, ...] = field(default_factory=lambda: (
        DecimalSpace("grid_spacing", Decimal("0.001"), Decimal("0.03"), Decimal("0.001")),
        DecimalSpace("grid_spacing_growth", Decimal("1"), Decimal("2"), Decimal("0.05")),
        DecimalSpace("grid_qty_growth", Decimal("0.5"), Decimal("2"), Decimal("0.05")),
        IntegerSpace("max_grid_layers", 1, 12),
        IntegerSpace("take_profit_layers", 1, 12),
        DecimalSpace("core_wallet_exposure_long", Decimal("0.02"), Decimal("0.50"), Decimal("0.01")),
        DecimalSpace("core_wallet_exposure_short", Decimal("0.02"), Decimal("0.50"), Decimal("0.01")),
        DecimalSpace("tactical_wallet_exposure_long", Decimal("0.01"), Decimal("0.30"), Decimal("0.01")),
        DecimalSpace("tactical_wallet_exposure_short", Decimal("0.01"), Decimal("0.30"), Decimal("0.01")),
        DecimalSpace("max_gross_wallet_exposure", Decimal("0.10"), Decimal("1.50"), Decimal("0.05")),
        DecimalSpace("take_profit_spacing", Decimal("0.001"), Decimal("0.05"), Decimal("0.001")),
        DecimalSpace(
            "unstuck_trigger_gross_exposure",
            Decimal("0.01"),
            Decimal("0.30"),
            Decimal("0.01"),
        ),
        DecimalSpace("stoploss", Decimal("-0.40"), Decimal("-0.01"), Decimal("0.01")),
        DecimalSpace("roi_0", Decimal("0.002"), Decimal("0.15"), Decimal("0.002")),
        DecimalSpace("trailing_positive", Decimal("0.001"), Decimal("0.05"), Decimal("0.001")),
        DecimalSpace("trailing_offset", Decimal("0.002"), Decimal("0.10"), Decimal("0.002")),
        CategoricalSpace("trailing_enabled", (False, True)),
        DecimalSpace("max_fill_ratio_per_order", Decimal("0.05"), ONE, Decimal("0.05")),
    ))

    def sample(self, trial: TrialLike) -> dict[str, Any]:
        values = {space.name: space.suggest(trial) for space in self.spaces}
        # Enforce coherent exposure and trailing relationships deterministically.
        core_long = Decimal(str(values.get("core_wallet_exposure_long", ZERO)))
        core_short = Decimal(str(values.get("core_wallet_exposure_short", ZERO)))
        tactical_long = Decimal(str(values.get("tactical_wallet_exposure_long", ZERO)))
        tactical_short = Decimal(str(values.get("tactical_wallet_exposure_short", ZERO)))
        minimum_gross = core_long + core_short + tactical_long + tactical_short
        if Decimal(str(values.get("max_gross_wallet_exposure", ZERO))) < minimum_gross:
            values["max_gross_wallet_exposure"] = minimum_gross
        positive = Decimal(str(values.get("trailing_positive", ZERO)))
        offset = Decimal(str(values.get("trailing_offset", ZERO)))
        if offset < positive:
            values["trailing_offset"] = positive
        return values

    @staticmethod
    def apply(base_config: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
        """Return a deep-enough copy with params mapped to planner/native-exit sections."""

        config = dict(base_config)
        hedge = dict(config.get("hedge", {}))
        planner = dict(hedge.get("planner", {}))
        native = dict(hedge.get("native_convergence", {}))
        paper = dict(hedge.get("paper", {}))
        exits = dict(native.get("exits", {}))
        planner_names = {
            "grid_spacing", "grid_spacing_growth", "grid_qty_growth",
            "max_grid_layers", "take_profit_layers",
            "core_wallet_exposure_long", "core_wallet_exposure_short",
            "tactical_wallet_exposure_long", "tactical_wallet_exposure_short",
            "max_gross_wallet_exposure", "take_profit_spacing",
            "unstuck_trigger_gross_exposure",
        }
        for name, value in params.items():
            serialized = str(value) if isinstance(value, Decimal) else value
            if name in planner_names:
                planner[name] = serialized
            elif name == "stoploss":
                config["stoploss"] = serialized
            elif name == "roi_0":
                config["minimal_roi"] = {"0": serialized}
            elif name == "trailing_enabled":
                config["trailing_stop"] = bool(value)
            elif name == "trailing_positive":
                config["trailing_stop_positive"] = serialized
            elif name == "trailing_offset":
                config["trailing_stop_positive_offset"] = serialized
            elif name == "max_fill_ratio_per_order":
                paper[name] = serialized
        native["exits"] = exits
        hedge["planner"] = planner
        hedge["paper"] = paper
        hedge["native_convergence"] = native
        config["hedge"] = hedge
        return config


@dataclass(frozen=True, slots=True)
class HedgeHyperoptWeights:
    return_reward: Decimal = Decimal("1")
    drawdown_penalty: Decimal = Decimal("2")
    liquidation_penalty: Decimal = Decimal("5")
    funding_penalty: Decimal = Decimal("0.5")
    fee_penalty: Decimal = Decimal("0.25")
    gross_penalty: Decimal = Decimal("0.2")
    turnover_penalty: Decimal = Decimal("0.05")
    inactivity_penalty: Decimal = Decimal("0.2")

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_decimal(getattr(self, name), field_name=name)
            if value < ZERO:
                raise ValueError("hyperopt loss weights cannot be negative")
            object.__setattr__(self, name, value)


class HedgeMultiObjectiveLoss:
    """Minimization loss balancing return, drawdown, margin and trading costs."""

    def __init__(self, weights: HedgeHyperoptWeights | None = None) -> None:
        self.weights = weights or HedgeHyperoptWeights()

    def __call__(self, result: HedgeBacktestArtifact | HedgeBacktestMetrics) -> float:
        metrics = result.metrics if isinstance(result, HedgeBacktestArtifact) else result
        start = max(metrics.starting_balance, Decimal("0.00000001"))
        cost_ratio = (abs(metrics.fees) + abs(metrics.funding)) / start
        gross_ratio = metrics.max_gross_notional / start
        inactivity = ONE if metrics.fill_count == 0 else ZERO
        liquidation_proxy = max(metrics.max_margin_utilization - Decimal("0.80"), ZERO)
        turnover_proxy = Decimal(metrics.fill_count) / Decimal("1000")
        score = (
            -metrics.total_return * self.weights.return_reward
            + metrics.max_drawdown * self.weights.drawdown_penalty
            + liquidation_proxy * self.weights.liquidation_penalty
            + cost_ratio * (self.weights.funding_penalty + self.weights.fee_penalty)
            + gross_ratio * self.weights.gross_penalty
            + turnover_proxy * self.weights.turnover_penalty
            + inactivity * self.weights.inactivity_penalty
        )
        return float(score)


@dataclass(frozen=True, slots=True)
class HedgeHyperoptTrialResult:
    number: int
    params: Mapping[str, Any]
    loss: float
    artifact_hash: str
    user_attrs: Mapping[str, Any] = field(default_factory=dict)


class HedgeHyperoptRunner:
    """Dependency-light runner; callers may supply Optuna or a deterministic trial source."""

    def __init__(
        self,
        *,
        space: HedgeHyperoptSpace | None = None,
        loss: HedgeMultiObjectiveLoss | None = None,
    ) -> None:
        self.space = space or HedgeHyperoptSpace()
        self.loss = loss or HedgeMultiObjectiveLoss()

    def evaluate(
        self,
        trial: TrialLike,
        *,
        number: int,
        base_config: Mapping[str, Any],
        backtest: Callable[[Mapping[str, Any]], HedgeBacktestArtifact],
    ) -> HedgeHyperoptTrialResult:
        params = self.space.sample(trial)
        config = self.space.apply(base_config, params)
        artifact = backtest(config)
        payload = artifact.to_dict()
        loss = self.loss(artifact)
        digest = payload["result_sha256"]
        attrs = {
            "return": str(artifact.metrics.total_return),
            "max_drawdown": str(artifact.metrics.max_drawdown),
            "max_margin_utilization": str(artifact.metrics.max_margin_utilization),
            "fills": artifact.metrics.fill_count,
        }
        return HedgeHyperoptTrialResult(number, params, loss, digest, attrs)

    @staticmethod
    def manifest(results: Iterable[HedgeHyperoptTrialResult]) -> dict[str, Any]:
        rows = [
            {
                "number": item.number,
                "loss": item.loss,
                "params": {k: str(v) if isinstance(v, Decimal) else v for k, v in item.params.items()},
                "artifact_hash": item.artifact_hash,
                "user_attrs": dict(item.user_attrs),
            }
            for item in results
        ]
        rows.sort(key=lambda item: (item["loss"], item["number"]))
        canonical = dumps(rows, sort_keys=True, separators=(",", ":"))
        return {
            "schema": "hedge-hyperopt-result-v1",
            "trials": rows,
            "best_trial": None if not rows else rows[0],
            "sha256": sha256(canonical.encode()).hexdigest(),
        }
